import cv2
from gaze_tracking import GazeTracking

gaze = GazeTracking()
webcam = cv2.VideoCapture(0)

while True:
    # We get a new frame from the webcam
    _, frame = webcam.read()

    # We send this frame to GazeTracking to analyze it
    gaze.refresh(frame)

    frame = gaze.annotated_frame()
    text = ""
    color = (147, 58, 31) # Default text color

    # --- NEW: Check for staring first, as it's the most important event ---
    if gaze.is_staring():
        text = "Staring Detected!"
        color = (0, 0, 255) # Red for staring
        
        # This is the "popup" signal: draw a red box around the gaze point
        gaze_point = gaze.gaze_coords()
        if gaze_point:
            x, y = gaze_point
            cv2.rectangle(frame, (x - 40, y - 40), (x + 40, y + 40), color, 2)

    elif gaze.is_blinking():
        text = "Blinking"
    elif gaze.is_right():
        text = "Looking right"
    elif gaze.is_left():
        text = "Looking left"
    elif gaze.is_center():
        text = "Looking center"

    # Display the main text (e.g., "Staring Detected!")
    cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, color, 2)

    # Display pupil coordinates
    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()
    cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

    cv2.imshow("Gaze Tracking", frame)

    if cv2.waitKey(1) == 27:
        break
   
webcam.release()
cv2.destroyAllWindows()