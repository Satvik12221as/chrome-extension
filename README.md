## AI-Powered Gaze & Attention Tracking System
This project is an advanced, high-precision system for real-time gaze and attention tracking using a standard webcam. It's designed to understand a user's cognitive state by detecting events like rereading or staring, and then providing AI-powered assistance.

# Project Architecture
1. The system operates on a real-time data processing pipeline that transforms raw video from the webcam into helpful AI-generated explanations.

2. The webcam captures the video feed and A Gaze Tracking Module using OpenCV and Dlib processes each frame to detect the user's face, identify facial landmarks, and calculate the precise (x, y) gaze coordinates on the screen.

3. The system detects Reread Events (when the user's eyes move backward on a line) and Fixation Events (when the user stares at one point for an extended time).

# Core Features
Real-Time Performance : Analyzes webcam feeds with minimal latency.

Pupil Tracking : Accurately locates the (x, y) coordinates of both pupils.

Gaze Direction : Determines if the user is looking left, right, or center.

Blink Detection : Reliably detects when the user's eyes are closed.

Auto-Calibration : Adapts to user's lighting conditions to improve pupil detection robustness.

# Tech Stack & Models
Language - Python 

Computer Vision - OpenCV

Face & Landmark - Dlib

Numerical Ops - NumPy

Dependencies - CMake & C++ Build Tools

Model Used - Dlib Facial Landmark Model (.dat)

# Getting Started
Prerequisites
This project relies on Dlib, which needs to be compiled from source. You must have CMake and C++ build tools installed on your system.

CMake: Download CMake (ensure you add it to your system PATH during installation).

C++ Build Tools: Download Visual Studio Build Tools (select the "Desktop development with C++" workload in the installer).

Installation
The most reliable way to set up the environment is by using Anaconda.

Clone the repository:

Bash

git clone <your-repository-url>
cd <your-repository-name>
Create the Conda Environment:
This command will read the environment.yml file and automatically install the correct versions of all libraries, including the complex Dlib dependency.

Bash

conda env create -f environment.yml
Activate the Environment:

Bash

conda activate GazeTracking
Running the Demo
With the environment activated, run the example script to start the webcam and see the gaze tracking in action.

Bash

python example.py

# How It Works
The system analyzes each frame from the webcam in a multi-step pipeline:

Find Your Face: Dlib’s pre-trained face detector locates a human face in the frame.

Map Key Points: A facial landmark model places 68 specific points on the face, precisely identifying the corners of the eyes, nose, etc.

Isolate the Eyes: The system crops the frame to create two small, isolated images of just the left and right eyes.

Calibrate for Lighting: An automatic calibration routine finds the optimal black-and-white threshold to make the pupil stand out.

Detect the Pupil: OpenCV finds the center of the dark pupil in the isolated eye images.

Interpret Gaze: The pupil's position within the eye frame is used to determine the final gaze direction.

