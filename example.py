import cv2
from gaze_tracking import GazeTracking

# --- 1. Global variables to manage the reading mode state ---
reading_mode_enabled = False
WINDOW_NAME = "Gaze Tracking - Press ESC to Exit"

# --- 2. A function to handle mouse clicks for the button ---
def toggle_reading_mode(event, x, y, flags, param):
    """Toggles the reading mode when the button is clicked."""
    global reading_mode_enabled
    # The button will be in the top-right corner, 250px wide and 50px tall.
    # We check if the left mouse button was clicked inside these coordinates
    if event == cv2.EVENT_LBUTTONDOWN:
        frame_width = int(webcam.get(cv2.CAP_PROP_FRAME_WIDTH))
        if (frame_width - 250) < x < frame_width and 10 < y < 60:
            reading_mode_enabled = not reading_mode_enabled
            print(f"Reading Mode Toggled: {'ON' if reading_mode_enabled else 'OFF'}")

# --- Setup the gaze tracker and webcam ---
gaze = GazeTracking()
webcam = cv2.VideoCapture(0)

# --- 3. Create a window and link our mouse click function to it ---
cv2.namedWindow(WINDOW_NAME)
cv2.setMouseCallback(WINDOW_NAME, toggle_reading_mode)

while True:
    _, frame = webcam.read()
    if frame is None:
        break

    # --- 4. The main logic: Only track gaze if reading mode is ON ---
    if reading_mode_enabled:
        # Send the frame to GazeTracking to analyze it
        gaze.refresh(frame)

        # Get the annotated frame with pupil crosses
        frame = gaze.annotated_frame()
        text = ""

        if gaze.is_blinking():
            text = "Blinking"
        elif gaze.is_right():
            text = "Looking right"
        elif gaze.is_left():
            text = "Looking left"
        elif gaze.is_center():
            text = "Looking center"

        # Display the gaze direction text
        cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

        # Display pupil coordinate
        left_pupil = gaze.pupil_left_coords()
        right_pupil = gaze.pupil_right_coords()
        cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
        cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

    # --- 5. Draw the button on the screen in every frame ---
    # The button's color and text change based on the mode.
    frame_width = int(webcam.get(cv2.CAP_PROP_FRAME_WIDTH))
    button_top_left = (frame_width - 250, 10)
    button_bottom_right = (frame_width - 10, 60)

    if reading_mode_enabled:
        # Green button for "ON" state
        button_color = (0, 180, 0)
        button_text = "Reading Mode: ON"
    else:
        # Red button for "OFF" state
        button_color = (0, 0, 180)
        button_text = "Reading Mode: OFF"
    
    cv2.rectangle(frame, button_top_left, button_bottom_right, button_color, -1)
    cv2.putText(frame, button_text, (button_top_left[0] + 10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.imshow(WINDOW_NAME, frame)

    if cv2.waitKey(1) == 27:
        break

webcam.release()
cv2.destroyAllWindows()