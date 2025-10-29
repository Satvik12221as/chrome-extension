# main.py
import cv2
import base64
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from gaze_tracking import GazeTracking

# -- 1. Initialize FastAPI --
# This is the line that was missing or incorrect.
# It creates the application object named "app".
app = FastAPI()

# --- 2. Initialize the GazeTracking model ---
# We create one instance to be used by the server.
gaze = GazeTracking()

# --- 3. Define the WebSocket Endpoint ---
# A client will connect to this address: ws://localhost:8000/ws
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    print("Accepting client connection...")
    await websocket.accept()
    
    try:
        # Loop forever to receive frames and send back results
        while True:
            # Receive image data (as a Base64 string) from the client
            base64_image_data = await websocket.receive_text()
            
            # Decode the Base64 string back into an image
            header, encoded_data = base64_image_data.split(",", 1)
            image_bytes = base64.b64decode(encoded_data)
            image_array = np.frombuffer(image_bytes, dtype=np.uint8)
            frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

            # Process the image with your GazeTracking logic
            gaze.refresh(frame)

            # Create a dictionary of results to send back
            results = {
                "is_blinking": gaze.is_blinking(),
                "is_right": gaze.is_right(),
                "is_left": gaze.is_left(),
                "is_center": gaze.is_center(),
                "is_staring": gaze.is_staring(),
                "left_pupil": gaze.pupil_left_coords(),
                "right_pupil": gaze.pupil_right_coords(),
            }
            
            # Send the results back to the client as JSON
            await websocket.send_json(results)
            
    except WebSocketDisconnect:
        print("Client disconnected.")