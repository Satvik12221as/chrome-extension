from __future__ import division
import os
import cv2
try:
    import dlib  # type: ignore[import]
except Exception:
    dlib = None
from .eye import Eye
from .calibration import Calibration
import time  
from collections import deque 

class FixationDetector(object):
    """
    Detects fixations (staring) by checking if gaze points stay
    within a small area for a certain duration.
    """
    def __init__(self, min_duration=5.0, dispersion_threshold=0.03, buffer_size=300):
        self.min_duration_seconds = min_duration
        self.dispersion_threshold = dispersion_threshold # How spread out the points can be
        self.buffer = deque(maxlen=buffer_size) # Stores recent (timestamp, gaze_point)
        self.is_fixating = False

    def feed(self, timestamp, gaze_point):
        """Adds a new gaze point to the buffer and checks for a fixation."""
        if gaze_point is None:
            self.is_fixating = False
            return

        self.buffer.append((timestamp, gaze_point))

        # Not enough data to be a fixation yet
        if len(self.buffer) < 10:
            self.is_fixating = False
            return

        # Check the duration of points in the buffer
        duration = self.buffer[-1][0] - self.buffer[0][0]
        if duration < self.min_duration_seconds:
            self.is_fixating = False
            return

        # If duration is long enough, check the dispersion (how spread out the points are)
        points = [p[1] for p in self.buffer]
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]

        dispersion = (max(xs) - min(xs)) + (max(ys) - min(ys))

        if dispersion < self.dispersion_threshold:
            self.is_fixating = True
        else:
            self.is_fixating = False

class GazeTracking(object):
    """
    This class tracks the user's gaze.
    It provides useful information like the position of the eyes
    """
    def __init__(self):
        self.frame = None
        self.eye_left = None
        self.eye_right = None
        self.calibration = Calibration()
        
        # This creates the new fixation detector
        self.fixation_detector = FixationDetector()

        # This creates the face detector
        self._face_detector = dlib.get_frontal_face_detector()

        # This loads the facial landmark model
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
        """Detects the face and initialize Eye objects"""
        frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        faces = self._face_detector(frame)

        try:
            landmarks = self._predictor(frame, faces[0])
            self.eye_left = Eye(frame, landmarks, 0, self.calibration)
            self.eye_right = Eye(frame, landmarks, 1, self.calibration)
        
        

        except IndexError:
            self.eye_left = None
            self.eye_right = None

    def refresh(self, frame):
        """Refreshes the frame and analyzes it.

        Arguments:
            frame (numpy.ndarray): The frame to analyze
        """
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

    def horizontal_ratio(self):
        """Returns a number between 0.0 and 1.0 that indicates the
        horizontal direction of the gaze. The extreme right is 0.0,
        the center is 0.5 and the extreme left is 1.0
        """
        if self.pupils_located:
            pupil_left = self.eye_left.pupil.x / (self.eye_left.center[0] * 2 - 10)
            pupil_right = self.eye_right.pupil.x / (self.eye_right.center[0] * 2 - 10)
            return (pupil_left + pupil_right) / 2

    def vertical_ratio(self):
        """Returns a number between 0.0 and 1.0 that indicates the
        vertical direction of the gaze. The extreme top is 0.0,
        the center is 0.5 and the extreme bottom is 1.0
        """
        if self.pupils_located:
            pupil_left = self.eye_left.pupil.y / (self.eye_left.center[1] * 2 - 10)
            pupil_right = self.eye_right.pupil.y / (self.eye_right.center[1] * 2 - 10)
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
        """
        Returns true if the user is staring at a single point.
        """
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
