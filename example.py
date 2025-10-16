import cv2
from gaze_tracking import GazeTracking

gaze = GazeTracking()
webcam = cv2.VideoCapture(0)

# We create a named window that can be manipulated
WINDOW_NAME = "Gaze Tracking (Press 'f' to toggle fullscreen, 'ESC' to exit)"
cv2.namedWindow(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

is_fullscreen = True

while True:
    # We get a new frame from the webcam
    _, frame = webcam.read()
    if frame is None:
        print("Error: Unable to capture video")
        break

    # we send this frame to GazeTracking to analyze it
    gaze.refresh(frame)

    frame = gaze.annotated_frame()
    text = ""

    if gaze.is_staring():
        text = 'Staring detected'

    if gaze.is_blinking():
        text = "Blinking"
    elif gaze.is_right():
        text = "Looking right"
    elif gaze.is_left():
        text = "Looking left"
    elif gaze.is_center():
        text = "Looking center"

    cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()
    cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

    cv2.imshow("WINDOW_NAME", frame)

    # Check for key presses for fullscreen toggle and exit 
    key = cv2.waitKey(1) & 0xFF

    # Press 'f' to toggle fullscreen mode
    if key == ord('f'):
        is_fullscreen = not is_fullscreen
        if is_fullscreen:
            cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

    # Press ESC (key code 27) to exit
    elif key == 27:
        break

webcam.release()
cv2.destroyAllWindows()