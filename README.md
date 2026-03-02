# Real-Time Person Following System using YOLOv8

## Overview

This project implements a real-time person following system using computer vision and control theory. It combines object detection, visual tracking, state estimation, and PID control to generate smooth yaw and forward movement commands based on a live webcam feed.

The system detects a person using YOLOv8, locks onto the target using a CSRT tracker, smooths motion with a Kalman filter, and computes control outputs using PID controllers.

This project is intended for robotics, autonomous systems, or intelligent camera platforms.

---

## Features

- YOLOv8 person detection
- CSRT object tracking for stable target lock
- Kalman filter for motion smoothing and prediction
- PID controller for yaw control
- PID controller for distance regulation
- Real-time webcam processing
- Geometric distance estimation from bounding box height

---

## System Architecture

1. Person Detection  
   The system uses YOLOv8 to detect a person in the frame.

2. Target Lock  
   After detection, a CSRT tracker maintains the target between detection cycles.

3. State Estimation  
   A Kalman filter predicts the target center position to reduce jitter.

4. Error Calculation  
   - Horizontal displacement from frame center determines yaw error.
   - Change in bounding box height determines distance error.

5. Control Output  
   Two PID controllers compute:
   - Yaw command
   - Forward movement command

---

## Requirements

- Python 3.8 or higher
- Webcam
- CPU or GPU capable of running YOLOv8

---

## Installation

Clone the repository:
- git clone https://github.com/yourusername/person-follow.git


## Configuration Parameters

Inside `main.py`, the following parameters should be adjusted:

- `REAL_PERSON_HEIGHT`  
  Real height of the tracked person in meters.

- `FOCAL_LENGTH`  
  Camera focal length in pixels. This must be calibrated for accurate distance estimation.

- `TARGET_DISTANCE`  
  Desired following distance in meters.

- PID gains  
  `pid_yaw` and `pid_dist` parameters can be tuned for smoother behavior.

---

## Distance Estimation

Distance is estimated using geometric projection:


distance = (REAL_PERSON_HEIGHT * FOCAL_LENGTH) / bounding_box_height


Accurate results require proper focal length calibration.

---

## Known Limitations

- Single person tracking only
- Performance depends on lighting conditions
- Webcam resolution affects accuracy
- No multi-target reidentification
- No hardware control layer included

---

## Future Improvements

- Multi-person selection logic
- Hardware integration for mobile robots
- ROS integration
- Dynamic PID auto-tuning
- GPU optimization
- Model size benchmarking

---

## License

This project is provided for educational and research purposes.

