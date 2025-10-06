# model.py
"""
Robust gaze + reread + fixation detector using MediaPipe FaceMesh + OpenCV.
Features:
- Robust pupil detection (adaptive thresholding + shape filtering)
- Smoothing (Kalman filter) to reduce jitter
- 2D polynomial calibration mapping gaze -> screen coordinates
- Reread detector (progress/regression heuristic) with confidence
- Fixation detector (dispersion-based)
- Clean API: GazeSystem.process_frame(bgr_frame) -> dict
"""

import time
from collections import deque
import numpy as np
import cv2
import mediapipe as mp

# 3rd-party libs required:
# pip install mediapipe opencv-python numpy

mp_face = mp.solutions.face_mesh

# ---------- Utilities ----------
def now_ts():
    return float(time.time())

def l2(a, b):
    a = np.array(a); b = np.array(b)
    return np.linalg.norm(a - b)

# ---------- Kalman2D (simple constant-velocity KF) ----------
class Kalman2D:
    def __init__(self, x0=0.5, y0=0.5, dt=1/8.0, process_var=1e-4, meas_var=1e-2):
        # state [x, y, vx, vy]
        self.dt = dt
        self.x = np.array([x0, y0, 0.0, 0.0], dtype=np.float64)
        self.P = np.eye(4) * 1e-2
        self.F = np.array([[1,0,dt,0],
                           [0,1,0,dt],
                           [0,0,1,0],
                           [0,0,0,1]], dtype=np.float64)
        self.Q = np.eye(4) * process_var
        self.H = np.array([[1,0,0,0],
                           [0,1,0,0]], dtype=np.float64)
        self.R = np.eye(2) * meas_var
        self.initialized = True

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, meas):
        z = np.array([meas[0], meas[1]], dtype=np.float64)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        y = z - (self.H @ self.x)
        self.x = self.x + (K @ y)
        self.P = (np.eye(4) - K @ self.H) @ self.P

    def step(self, meas):
        if meas is None:
            # predict only
            self.predict()
            return (self.x[0], self.x[1])
        self.predict()
        self.update(meas)
        return (float(self.x[0]), float(self.x[1]))

# ---------- Pupil detection inside an eye ROI ----------
def detect_pupil_in_eye(eye_img):
    """
    Input: eye_img BGR
    Returns: (nx, ny, score) where nx,ny in [0,1] relative to eye_img size, or None
    """
    if eye_img is None or eye_img.size == 0:
        return None

    h, w = eye_img.shape[:2]
    gray = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
    # Improve contrast
    gray = cv2.equalizeHist(gray)
    # Blur
    blur = cv2.GaussianBlur(gray, (7,7), 0)
    # Invert adaptive threshold to isolate dark pupil on bright sclera
    th = cv2.adaptiveThreshold(blur, 255,
                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 11, 5)
    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
    # Remove small specks
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    candidates = []
    area_img = float(w*h)
    for c in contours:
        area = cv2.contourArea(c)
        if area <= 3 or area < 0.001 * area_img:
            continue
        per = cv2.arcLength(c, True)
        if per <= 0:
            continue
        circularity = 4 * np.pi * (area / (per*per))
        M = cv2.moments(c)
        if M.get("m00", 0) == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        # penalize blobs near box edge (likely eyelid or noise)
        edge_pen = min(cx, w-cx)/w
        score = circularity * (0.5 + 0.5 * edge_pen) * (area / area_img)
        candidates.append((score, cx, cy, area, circularity))

    if not candidates:
        return None

    # choose best candidate by score
    candidates.sort(key=lambda x: x[0], reverse=True)
    best = candidates[0]
    score, cx, cy, area, circularity = best
    nx = float(cx / max(1, w))
    ny = float(cy / max(1, h))
    return (nx, ny, float(score))

# ---------- FaceMesh Eyebox extractor ----------
class FaceMeshEyeBoxes:
    """
    Uses MediaPipe FaceMesh to compute stable eye bounding boxes and some facial landmarks.
    """
    def __init__(self, static_image_mode=False, max_faces=1, refine_landmarks=True,
                 detection_conf=0.6, tracking_conf=0.6):
        self.fm = mp_face.FaceMesh(static_image_mode=static_image_mode,
                                   max_num_faces=max_faces,
                                   refine_landmarks=refine_landmarks,
                                   min_detection_confidence=detection_conf,
                                   min_tracking_confidence=tracking_conf)

    def get_landmarks(self, bgr_frame):
        h, w = bgr_frame.shape[:2]
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        res = self.fm.process(rgb)
        if not res.multi_face_landmarks:
            return None
        lm = res.multi_face_landmarks[0]
        # convert to array of (x,y) in pixel coords
        pts = np.array([[int(p.x * w), int(p.y * h)] for p in lm.landmark], dtype=np.int32)
        return pts

    def eye_boxes_from_landmarks(self, lm_pts):
        """
        Returns: dict with left_box=(x1,y1,x2,y2), right_box=(...), eye_centers=(cx,cy)
        Uses conservative landmark indices known to bracket the eyes in MediaPipe FaceMesh.
        """
        # these indices are standard approximations used widely with Mediapipe FaceMesh
        left_idx = [33, 133, 159, 145, 153, 154, 155, 133]
        right_idx = [362, 263, 386, 374, 388, 387, 386, 263]

        def bbox_from_idx(idxs):
            pts = lm_pts[idxs]
            minx = int(np.min(pts[:,0])); maxx = int(np.max(pts[:,0]))
            miny = int(np.min(pts[:,1])); maxy = int(np.max(pts[:,1]))
            # pad a bit
            w = maxx - minx; h = maxy - miny
            pad_x = int(0.25 * w) + 4
            pad_y = int(0.45 * h) + 4
            x1 = max(0, minx - pad_x); x2 = max(1, maxx + pad_x)
            y1 = max(0, miny - pad_y); y2 = max(1, maxy + pad_y)
            return (x1, y1, x2, y2)

        left_box = bbox_from_idx(left_idx)
        right_box = bbox_from_idx(right_idx)
        # centers
        lcx = (left_box[0] + left_box[2]) / 2.0
        lcy = (left_box[1] + left_box[3]) / 2.0
        rcx = (right_box[0] + right_box[2]) / 2.0
        rcy = (right_box[1] + right_box[3]) / 2.0
        return {
            "left_box": left_box,
            "right_box": right_box,
            "left_center": (lcx, lcy),
            "right_center": (rcx, rcy)
        }

# ---------- Calibration mapper ----------
class CalibrationMapper:
    """
    Fits a 2D quadratic polynomial mapping raw gaze (gx,gy) -> screen coordinates (sx, sy).
    Model terms: [1, gx, gy, gx^2, gx*gy, gy^2]
    Use normalized screen coords in [0,1] for targets.
    """
    def __init__(self):
        self.Ax = None  # coefficients for x
        self.Ay = None
        self.screen_w = 1.0
        self.screen_h = 1.0
        self.fitted = False

    def _design(self, gx, gy):
        return np.array([1.0, gx, gy, gx*gx, gx*gy, gy*gy], dtype=np.float64)

    def fit(self, gaze_samples, screen_samples, screen_size=(1.0,1.0)):
        """
        gaze_samples: list of (gx, gy) where gx,gy normalized [0,1]
        screen_samples: list of (sx, sy) where sx,sy normalized [0,1] (relative to screen)
        Need at least 6 samples (preferably 9 or more).
        """
        if len(gaze_samples) < 6:
            self.fitted = False
            return False
        X = np.vstack([self._design(gx,gy) for (gx,gy) in gaze_samples])
        sx = np.array([s for (s,_) in screen_samples], dtype=np.float64)
        sy = np.array([t for (_,t) in screen_samples], dtype=np.float64)
        # least squares
        try:
            coeff_x, *_ = np.linalg.lstsq(X, sx, rcond=None)
            coeff_y, *_ = np.linalg.lstsq(X, sy, rcond=None)
        except Exception:
            self.fitted = False
            return False
        self.Ax = coeff_x.flatten()
        self.Ay = coeff_y.flatten()
        self.screen_w, self.screen_h = screen_size
        self.fitted = True
        return True

    def map(self, gx, gy):
        if not self.fitted or self.Ax is None:
            # fallback: return same normalized gaze
            return (gx, gy)
        d = self._design(gx, gy)
        sx = float(np.dot(self.Ax, d))
        sy = float(np.dot(self.Ay, d))
        # clamp
        return (min(max(sx, 0.0), 1.0), min(max(sy, 0.0), 1.0))

# ---------- Detectors: fixation and reread ----------
class FixationDetector:
    """
    Dispersion-based fixation detection.
    Keep sliding window of gaze samples, measure dispersion (max-min).
    """
    def __init__(self, max_window_seconds=2.5, fps=8, dispersion_thresh=0.04, min_duration=0.9):
        self.max_len = int(max_window_seconds * fps)
        self.buf = deque(maxlen=self.max_len)  # (t, gx, gy)
        self.dispersion_thresh = dispersion_thresh
        self.min_duration = min_duration

    def feed(self, t, gx, gy):
        self.buf.append((t, gx, gy))
        return self.is_fixation()

    def is_fixation(self):
        if len(self.buf) < 3:
            return False, 0.0
        arr = np.array(self.buf)
        times = arr[:,0]
        duration = float(times[-1] - times[0])
        gxs = arr[:,1]; gys = arr[:,2]
        disp = float((np.max(gxs)-np.min(gxs)) + (np.max(gys)-np.min(gys)))
        if duration >= self.min_duration and disp <= self.dispersion_thresh:
            return True, duration
        return False, duration

class RereadDetector:
    """
    Detect regression in x while vertical stays approximately in same line.
    Heuristic:
    - Use recent window (~1.0-1.5s). Split into earlier/later halves.
    - If earlier half median x is >> later half median x (user moved right then back left)
      and vertical spread small -> treat as reread.
    - Returns (bool, confidence)
    """
    def __init__(self, fps=8, window_seconds=1.4, x_back_thresh=0.12, vertical_tol=0.06):
        self.window_len = int(window_seconds * fps)
        self.buf = deque(maxlen=max(8, self.window_len))
        self.x_back_thresh = x_back_thresh
        self.vertical_tol = vertical_tol

    def feed(self, t, gx, gy):
        self.buf.append((t, gx, gy))
        return self.check()

    def check(self):
        if len(self.buf) < 8:
            return False, 0.0
        arr = np.array(self.buf)
        gxs = arr[:,1]; gys = arr[:,2]
        half = max(3, len(gxs)//2)
        first = gxs[:half]; second = gxs[half:]
        max_first = float(np.max(first))
        min_second = float(np.min(second))
        diff = max_first - min_second
        vert_spread = float(np.max(gys) - np.min(gys))
        if diff > self.x_back_thresh and vert_spread <= self.vertical_tol:
            # confidence proportional to diff and stability
            conf = np.clip((diff - self.x_back_thresh) / 0.25 + 0.2, 0.05, 0.99)
            return True, conf
        return False, 0.0

# ---------- Main System ----------
class GazeSystem:
    def __init__(self, frame_w=640, frame_h=480, fps=8):
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.fps = fps
        self.eye_extractor = FaceMeshEyeBoxes()
        self.kalman = Kalman2D(dt=1.0/max(1.0, fps))
        self.calibrator = CalibrationMapper()
        self.fix_detector = FixationDetector(fps=fps)
        self.reread_detector = RereadDetector(fps=fps)
        self.history = deque(maxlen=1000)  # (t, gx, gy, conf)
        # tuning params
        self.pupil_weight = 1.0  # weight when combining left/right pupils
        self.last_valid = None

    def process_frame(self, frame_bgr, ts=None):
        """
        Input: BGR image (numpy)
        Returns dict:
        {
          "raw_gaze": (gx, gy) normalized [0,1] relative to camera frame,
          "screen_gaze": (sx, sy) normalized [0,1] (after calibration),
          "fixation": bool, "fixation_duration": float,
          "reread": bool, "reread_confidence": float,
          "confidence": 0..1,
          "debug": { optional debug info },
          "ts": timestamp
        }
        """
        if ts is None:
            ts = now_ts()
        h, w = frame_bgr.shape[:2]
        pts = self.eye_extractor.get_landmarks(frame_bgr)
        if pts is None:
            # no face detected
            self.history.append((ts, None, None, 0.0))
            return {
                "raw_gaze": (None, None),
                "screen_gaze": (None, None),
                "fixation": False,
                "fixation_duration": 0.0,
                "reread": False,
                "reread_confidence": 0.0,
                "confidence": 0.0,
                "ts": ts
            }

        boxes = self.eye_extractor.eye_boxes_from_landmarks(pts)
        lx1, ly1, lx2, ly2 = boxes["left_box"]
        rx1, ry1, rx2, ry2 = boxes["right_box"]

        left_eye = frame_bgr[ly1:ly2, lx1:lx2].copy()
        right_eye = frame_bgr[ry1:ry2, rx1:rx2].copy()

        left_p = detect_pupil_in_eye(left_eye)
        right_p = detect_pupil_in_eye(right_eye)

        # compute normalized "eye-centered gaze" offsets [-1..1] where 0 is center
        def pupil_to_offset(pupil, box):
            if pupil is None:
                return None
            nx, ny, score = pupil
            # Convert to offset: (nx - 0.5) * scale
            bx1, by1, bx2, by2 = box
            bw = bx2 - bx1; bh = by2 - by1
            ox = (nx - 0.5) * 2.0  # -1..1
            oy = (ny - 0.5) * 2.0
            return (ox, oy, score, bw, bh)

        lo = pupil_to_offset(left_p, boxes["left_box"])
        ro = pupil_to_offset(right_p, boxes["right_box"])

        conf_score = 0.0
        raw_gx = None; raw_gy = None

        # combine offsets to estimate gaze point relative to frame
        if lo is None and ro is None:
            # fallback: no pupil detection
            raw = None
            conf_score = 0.0
        else:
            # compute frame space coords from each eye if available
            candidates = []
            if lo is not None:
                ox, oy, score, bw, bh = lo
                # eye center in frame coords
                ecx, ecy = boxes["left_center"]
                # translate offset into pixels: use bw as horizontal scale
                px = ecx + ox * (bw * 0.48)
                py = ecy + oy * (bh * 0.48)
                candidates.append((px, py, score))
            if ro is not None:
                ox, oy, score, bw, bh = ro
                ecx, ecy = boxes["right_center"]
                px = ecx + ox * (bw * 0.48)
                py = ecy + oy * (bh * 0.48)
                candidates.append((px, py, score))
            # weighted average by score
            if candidates:
                total_w = sum([c[2] for c in candidates]) + 1e-6
                avg_x = sum([c[0]*c[2] for c in candidates]) / total_w
                avg_y = sum([c[1]*c[2] for c in candidates]) / total_w
                raw_gx = float(avg_x / max(1.0, w))
                raw_gy = float(avg_y / max(1.0, h))
                conf_score = min(0.99, float(min(1.0, total_w)))  # rough confidence from score sum

        if raw_gx is None:
            # fallback to last valid if available
            if self.last_valid:
                raw_gx, raw_gy = self.last_valid
                conf_score = 0.15
            else:
                self.history.append((ts, None, None, 0.0))
                return {
                    "raw_gaze": (None, None),
                    "screen_gaze": (None, None),
                    "fixation": False, "fixation_duration": 0.0,
                    "reread": False, "reread_confidence": 0.0,
                    "confidence": 0.0,
                    "ts": ts
                }

        # smoothing via Kalman
        sm_gx, sm_gy = self.kalman.step((raw_gx, raw_gy))
        self.last_valid = (sm_gx, sm_gy)

        # map to screen coords
        sx, sy = self.calibrator.map(sm_gx, sm_gy)

        # feed detectors
        self.fix_detector.feed(ts, sm_gx, sm_gy)
        fixation, fix_dur = self.fix_detector.is_fixation()
        reread, reread_conf = self.reread_detector.feed(ts, sm_gx, sm_gy)

        self.history.append((ts, sm_gx, sm_gy, conf_score))

        return {
            "raw_gaze": (float(sm_gx), float(sm_gy)),
            "screen_gaze": (float(sx), float(sy)),
            "fixation": bool(fixation),
            "fixation_duration": float(fix_dur),
            "reread": bool(reread),
            "reread_confidence": float(reread_conf),
            "confidence": float(conf_score),
            "ts": ts,
            "debug": {
                "left_pupil": left_p,
                "right_pupil": right_p,
                "left_box": boxes["left_box"],
                "right_box": boxes["right_box"]
            }
        }

    # ---------- calibration helpers ----------
    def calibrate_from_samples(self, gaze_samples, screen_samples, screen_size=(1.0,1.0)):
        """
        gaze_samples: list of (gx,gy) raw normalized gaze from camera (0..1)
        screen_samples: list of (sx,sy) normalized screen positions (0..1)
        """
        return self.calibrator.fit(gaze_samples, screen_samples, screen_size=screen_size)

    # utility to collect a median gaze from short capture
    def capture_median_gaze(self, frames, ts_list=None):
        """
        frames: list of BGR frames (or call process_frame repeatedly)
        Return median raw gaze (gx,gy) across frames that had detection
        """
        samples = []
        for f in frames:
            r = self.process_frame(f)
            gx, gy = r["raw_gaze"]
            if gx is not None:
                samples.append((gx, gy))
        if not samples:
            return None
        arr = np.array(samples)
        return (float(np.median(arr[:,0])), float(np.median(arr[:,1])))

