# main.py
import cv2
import base64
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from gaze_tracking import GazeTracking
import os # Import the os library to check for files

# --- 1. CRITICAL FILE CHECK ---
# Before we do anything else, let's verify that the model file exists.
# This will give a clear error instead of crashing the server.
model_path = "gaze_tracking/trained_models/shape_predictor_68_face_landmarks.dat"
if not os.path.exists(model_path):
    print("="*50)
    print(f"❌ ERROR: Dlib model not found at '{model_path}'")
    print("Please ensure the 'trained_models' folder is INSIDE the 'gaze_tracking' folder.")
    print("="*50)
    exit() # Stop the script if the model is missing.
# -----------------------------

# If the check passes, we proceed to create the app.
app = FastAPI()
gaze = GazeTracking()
print("✅ Dlib model found. Initializing Gaze Tracker...")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    print("✅ SERVER: Client connected.")
    await websocket.accept()
    
    try:
        while True:
            base64_image_data = await websocket.receive_text()
            
            try:
                header, encoded_data = base64_image_data.split(",", 1)
                image_bytes = base64.b64decode(encoded_data)
                image_array = np.frombuffer(image_bytes, dtype=np.uint8)
                frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

                if frame is None: continue

                gaze.refresh(frame)
                results = {
                    "is_blinking": gaze.is_blinking(), "is_right": gaze.is_right(),
                    "is_left": gaze.is_left(), "is_center": gaze.is_center(),
                    "is_staring": gaze.is_staring(), "left_pupil": gaze.pupil_left_coords(),
                    "right_pupil": gaze.pupil_right_coords(),
                }
                
                await websocket.send_json(results)
            except Exception as e:
                print(f"❌ SERVER: Error processing frame: {e}")
    except WebSocketDisconnect:
        print("🔌 SERVER: Client disconnected.")