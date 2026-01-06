# JoySeeker-AI-Drone

An AI-powered drone system that automatically captures joyful moments by tracking smiles through lightweight vision models and offline voice control.

## Project Overview

JoySeeker-AI-Drone is an innovative project that combines computer vision, speech recognition, and drone control technologies. The system can automatically recognize facial expressions, adjust the drone's position for shooting when stable smiles are detected, and supports Chinese voice command control, providing users with a convenient intelligent photography experience.

## Key Features

- **Intelligent Smile Tracking**: Real-time facial expression detection using MobileNetV3 model to recognize smile states
- **Automatic Positioning Photography**: Drone automatically adjusts position and captures photos when stable smiles are detected
- **Chinese Voice Control**: Offline Chinese voice command control for drone flight
- **Real-time Status Monitoring**: Provides real-time display of drone status
- **Dual Control Modes**: Supports both voice control and keyboard control

## Technical Architecture

### Core Modules

1. **Main Control Module** (`main.py`)
   - Coordinates all modules
   - Handles user input and system state switching

2. **Speech Recognition Module** (`asr_module.py`)
   - Offline Chinese speech recognition based on Vosk
   - Supports specific command set recognition

3. **Drone Control Module** (`tello_controller.py`)
   - Drone control based on DJI Tello SDK
   - Implements flight control and status acquisition

4. **Vision Processing Module** (`frame_parser.py`)
   - Expression recognition based on MobileNetV3
   - Implements smile detection and target positioning

5. **User Interface Module** (`UI.py`)
   - Real-time display of drone status and video stream
   - Provides screenshot functionality

### Technical Features

- **Temporal Anti-shake Algorithm**: Ensures only continuous smiles trigger photography, avoiding false triggers
- **Spatial Anti-shake Algorithm**: Smoothly processes target position changes, preventing drastic drone movements
- **Lightweight Model**: Uses MobileNetV3 for real-time expression recognition
- **Offline Speech Recognition**: Enables voice control without network connection

## Installation and Usage

### Requirements

- Python 3.8+
- DJI Tello drone
- CUDA-enabled GPU (optional, for accelerating model inference)

### Installation Steps

1. Clone the project repository
```bash
git clone [repository URL]
cd JoySeeker-AI-Drone
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Ensure resource files are complete
   - `resource/best_emotion_model.pth`: Trained expression recognition model
   - `resource/vosk-model-small-cn-0.22`: Chinese speech recognition model

### Usage

1. Start the program
```bash
python main.py
```

2. Control methods
   - **Keyboard Control**:
     - Q: Exit program
     - Y: Toggle voice control
     - U: Toggle auto-tracking
     - W/A/S/D: Forward/Left/Backward/Right
     - R/F: Up/Down
     - J: Hover
     - K: Takeoff
     - L: Land

   - **Voice Control** (press Y to enable first):
     - "上" (up), "下" (down), "左" (left), "右" (right), "前" (forward), "后" (back): Control drone movement
     - "起飞" (takeoff), "降落" (land), "悬停" (hover): Control drone status

## Project Structure

```
JoySeeker-AI-Drone/
├── main.py                 # Main program entry
├── asr_module.py           # Speech recognition module
├── tello_controller.py     # Drone control module
├── frame_parser.py         # Vision processing module
├── UI.py                   # User interface module
├── requirements.txt        # Project dependencies
├── resource/               # Resource folder
│   ├── best_emotion_model.pth  # Expression recognition model
│   └── vosk-model-small-cn-0.22 # Chinese speech recognition model
└── training/               # Training related files
```

## Technical Details

### Expression Recognition Model

Uses a deep learning model based on MobileNetV3 architecture, trained on the FER2013 dataset, capable of recognizing 7 basic emotions (anger, disgust, fear, happiness, sadness, surprise, neutral), with special optimization for smile detection.

### Speech Recognition System

Adopts Vosk lightweight Chinese speech recognition model, supporting offline operation, recognizing specific command sets, including directional control commands and status control commands.

### Drone Control Algorithm

Implements smooth control algorithms, combining temporal and spatial anti-shake technologies to ensure stable drone flight and precise positioning.

## Development Team

This project was developed by Team 7 of the Deep Learning and Artificial Intelligence course.

## License

This project is open-source, see LICENSE file for details.