# client.py
import cv2
import websockets
import asyncio
import base64
import json

reading_mode_enabled = False
WINDOW_NAME = "Gaze Tracking Client - Press ESC to Exit"

def toggle_reading_mode(event, x, y, flags, param):
    global reading_mode_enabled
    frame_width = param.get('frame_width', 1280)
    if event == cv2.EVENT_LBUTTONDOWN:
        if (frame_width - 250) < x < frame_width and 10 < y < 60:
            reading_mode_enabled = not reading_mode_enabled
            print(f"CLIENT: Reading Mode Toggled: {'ON' if reading_mode_enabled else 'OFF'}")

async def run_gaze_client():
    global reading_mode_enabled
    uri = "ws://localhost:8000/ws"
    
    try:
        async with websockets.connect(uri) as websocket:
            print("✅ CLIENT: Connected to the server.")
            webcam = cv2.VideoCapture(0)
            frame_width = int(webcam.get(cv2.CAP_PROP_FRAME_WIDTH))
            cv2.namedWindow(WINDOW_NAME)
            cv2.setMouseCallback(WINDOW_NAME, toggle_reading_mode, {'frame_width': frame_width})
            
            while True:
                ret, frame = webcam.read()
                if not ret: break

                if reading_mode_enabled:
                    _, buffer = cv2.imencode('.jpg', frame)
                    base64_image = "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')
                    await websocket.send(base64_image)
                    
                    response = await websocket.recv()
                    results = json.loads(response)

                    text = ""
                    color = (147, 58, 31)
                    if results.get("is_staring"): text, color = "Staring!", (0, 0, 255)
                    elif results.get("is_blinking"): text = "Blinking"
                    elif results.get("is_right"): text = "Looking right"
                    elif results.get("is_left"): text = "Looking left"
                    elif results.get("is_center"): text = "Looking center"
                    cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, color, 2)

                button_top_left = (frame_width - 250, 10)
                button_bottom_right = (frame_width - 10, 60)
                button_color = (0, 180, 0) if reading_mode_enabled else (0, 0, 180)
                button_text = "Reading Mode: ON" if reading_mode_enabled else "Reading Mode: OFF"
                cv2.rectangle(frame, button_top_left, button_bottom_right, button_color, -1)
                cv2.putText(frame, button_text, (button_top_left[0] + 10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.imshow(WINDOW_NAME, frame)
                if cv2.waitKey(1) == 27: break
            
            webcam.release()
            cv2.destroyAllWindows()
    except Exception as e:
        print(f"\n❌ CLIENT: An error occurred: {e}")
        print("Please ensure the server is running and the model file is in the correct location.")

if __name__ == "__main__":
    asyncio.run(run_gaze_client())