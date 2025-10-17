from __future__ import division
import os
import cv2
import dlib
from .eye import Eye
from .calibration import Calibration
import time
from collections import deque
import numpy as np

# NEW: Class to detect fixations (staring)
class FixationDetector(object):
    """
    Detects fixations (staring) by checking if gaze points stay
    within a small area for a certain duration.
    """
    def __init__(self, min_duration=5.0, dispersion_threshold=50, buffer_size=150):
        self.min_duration_seconds = min_duration
        self.dispersion_threshold_pixels = dispersion_threshold
        self.buffer = deque(maxlen=buffer_size)
        self.is_fixating = False
        self.fixation_start_time = 0

    def feed(self, timestamp, gaze_point_pixels):
        """Adds a new gaze point to the buffer and checks for a fixation."""
        # If we can't see the pupils, reset the fixation
        if gaze_point_pixels is None:
            self.is_fixating = False
            self.buffer.clear()
            return

        self.buffer.append({'time': timestamp, 'point': gaze_point_pixels})

        # Calculate the duration of the gaze points currently in our buffer
        duration = self.buffer[-1]['time'] - self.buffer[0]['time']

        # If the duration is less than our 5-second requirement, it's not a fixation
        if duration < self.min_duration_seconds:
            self.is_fixating = False
            return

        # If the duration is long enough, check how spread out the points are
        points = np.array([item['point'] for item in self.buffer])
        max_x, max_y = np.max(points, axis=0)
        min_x, min_y = np.min(points, axis=0)
        
        # Calculate dispersion (the size of the bounding box around the points)
        dispersion = (max_x - min_x) + (max_y - min_y)

        # If the points are clustered tightly together, it's a fixation
        if dispersion < self.dispersion_threshold_pixels:
            if not self.is_fixating:
                self.fixation_start_time = timestamp
            self.is_fixating = True
        else:
            # If they are too spread out, it's not a fixation, so we clear the buffer
            self.is_fixating = False
            self.buffer.clear()

class GazeTracking(object):
    """
    This class tracks the user's gaze.
    """
    def __init__(self):
        self.frame = None
        self.eye_left = None
        self.eye_right = None
        self.calibration = Calibration()
        
        # Initialize the new fixation detector
        self.fixation_detector = FixationDetector()

        # Dlib face detector and landmark predictor
        self._face_detector = dlib.get_frontal_face_detector()
        cwd = os.path.dirname(__file__)
        model_path = os.path.join(cwd, "trained_models/shape_predictor_68_face_landmarks.dat")
        self._predictor = dlib.shape_predictor(model_path)

    @property
    def pupils_located(self):
        """Check that the pupils have been located"""
        try:
            int(self.eye_left.pupil.x)
            int(self.eye_left.pupil.y)
            int(self.eye_right.pupil.x)
            int(self.eye_right.pupil.y)
            return True
        except Exception:
            return False

    def _analyze(self):
        """Detects the face and analyzes it"""
        frame_gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        faces = self._face_detector(frame_gray)

        if len(faces) > 0:
            landmarks = self._predictor(self.frame, faces[0])
            self.eye_left = Eye(self.frame, landmarks, 0, self.calibration)
            self.eye_right = Eye(self.frame, landmarks, 1, self.calibration)
        else:
            self.eye_left = None
            self.eye_right = None

        # Feed data to our fixation detector on every frame
        timestamp = time.time()
        gaze_point_pixels = self.gaze_coords()
        self.fixation_detector.feed(timestamp, gaze_point_pixels)

    def refresh(self, frame):
        """Refreshes the frame and analyzes it."""
        self.frame = frame
        self._analyze()

    def pupil_left_coords(self):
        """Returns the coordinates of the left pupil"""
        if self.pupils_located:
            x = self.eye_left.origin[0] + self.eye_left.pupil.x
            y = self.eye_left.origin[1] + self.eye_left.pupil.y
            return (x, y)

    def pupil_right_coords(self):
        """Returns the coordinates of the right pupil"""
        if self.pupils_located:
            x = self.eye_right.origin[0] + self.eye_right.pupil.x
            y = self.eye_right.origin[1] + self.eye_right.pupil.y
            return (x, y)

    def gaze_coords(self):
        """Returns the average coordinates of the two pupils"""
        if self.pupils_located:
            left_x, left_y = self.pupil_left_coords()
            right_x, right_y = self.pupil_right_coords()
            return (int((left_x + right_x) / 2), int((left_y + right_y) / 2))
        return None

    def horizontal_ratio(self):
        """Returns a number between 0.0 and 1.0 that indicates the
        horizontal direction of the gaze. The extreme right is 0.0,
        the center is 0.5 and the extreme left is 1.0
        """
        if self.pupils_located:
            pupil_left = self.eye_left.pupil.x / (self.eye_left.center[0] * 2 - 1)
            pupil_right = self.eye_right.pupil.x / (self.eye_right.center[0] * 2 - 1)
            return (pupil_left + pupil_right) / 2

    def vertical_ratio(self):
        """Returns a number between 0.0 and 1.0 that indicates the
        vertical direction of the gaze. The extreme top is 0.0,
        the center is 0.5 and the extreme bottom is 1.0
        """
        if self.pupils_located:
            pupil_left = self.eye_left.pupil.y / (self.eye_left.center[1] * 2 - 1)
            pupil_right = self.eye_right.pupil.y / (self.eye_right.center[1] * 2 - 1)
            return (pupil_left + pupil_right) / 2

    def is_right(self):
        """Returns true if the user is looking to the right"""
        if self.pupils_located:
            return self.horizontal_ratio() <= 0.35

    def is_left(self):
        """Returns true if the user is looking to the left"""
        if self.pupils_located:
            return self.horizontal_ratio() >= 0.65

    def is_center(self):
        """Returns true if the user is looking to the center"""
        if self.pupils_located:
            return self.is_right() is not True and self.is_left() is not True

    def is_blinking(self):
        """Returns true if the user closes his eyes"""
        if self.pupils_located:
            blinking_ratio = (self.eye_left.blinking + self.eye_right.blinking) / 2
            return blinking_ratio > 3.8

    def is_staring(self):
        """Returns true if the user is staring at a single point."""
        return self.fixation_detector.is_fixating

    def annotated_frame(self):
        """Returns the main frame with pupils highlighted"""
        frame = self.frame.copy()
        if self.pupils_located:
            color = (0, 255, 0)
            x_left, y_left = self.pupil_left_coords()
            x_right, y_right = self.pupil_right_coords()
            cv2.line(frame, (x_left - 5, y_left), (x_left + 5, y_left), color)
            cv2.line(frame, (x_left, y_left - 5), (x_left, y_left + 5), color)
            cv2.line(frame, (x_right - 5, y_right), (x_right + 5, y_right), color)
            cv2.line(frame, (x_right, y_right - 5), (x_right, y_right + 5), color)
        return frame