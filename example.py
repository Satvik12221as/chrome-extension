import cv2
from gaze_tracking import GazeTracking

# --- 1. Global variables to manage the state ---
reading_mode_enabled = False
WINDOW_NAME = "Gaze Tracking - Press ESC to Exit"

# --- 2. Mouse click handler to toggle the mode ---
def toggle_reading_mode(event, x, y, flags, param):
    """Toggles the reading mode when the button is clicked."""
    global reading_mode_enabled
    frame_width = int(webcam.get(cv2.CAP_PROP_FRAME_WIDTH))
    if event == cv2.EVENT_LBUTTONDOWN:
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

    # --- 4. Main Logic: Only track gaze if reading mode is ON ---
    if reading_mode_enabled:
        gaze.refresh(frame)
        frame = gaze.annotated_frame() # Get frame with green crosses
        text = ""
        color = (147, 58, 31)

        if gaze.is_staring():
            text = "Staring!"
            color = (0, 0, 255)
        elif gaze.is_blinking():
            text = "Blinking"
        elif gaze.is_right():
            text = "Looking right"
        elif gaze.is_left():
            text = "Looking left"
        elif gaze.is_center():
            text = "Looking center"
        
        cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, color, 2)

        # --- NEW: Get and display pupil coordinates ---
        left_pupil = gaze.pupil_left_coords()
        right_pupil = gaze.pupil_right_coords()
        cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
        cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
        # ------------------------------------------------

    # --- 5. Draw the button on the screen in every frame ---
    frame_width = int(webcam.get(cv2.CAP_PROP_FRAME_WIDTH))
    button_top_left = (frame_width - 250, 10)
    button_bottom_right = (frame_width - 10, 60)

    if reading_mode_enabled:
        button_color = (0, 180, 0) # Green for ON
        button_text = "Reading Mode: ON"
    else:
        button_color = (0, 0, 180) # Red for OFF
        button_text = "Reading Mode: OFF"
    
    cv2.rectangle(frame, button_top_left, button_bottom_right, button_color, -1)
    cv2.putText(frame, button_text, (button_top_left[0] + 10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.imshow(WINDOW_NAME, frame)

    if cv2.waitKey(1) == 27: # Press ESC to exit
        break

webcam.release()
cv2.destroyAllWindows()