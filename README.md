# 🤖 AI-Based 5DOF Robotic Arm Control System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Arduino](https://img.shields.io/badge/Arduino-Uno-green)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-red)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Hand%20Tracking-orange)
![ESP32-CAM](https://img.shields.io/badge/ESP32-CAM-yellow)
![Status](https://img.shields.io/badge/Status-Active-success)

**AI-Assisted Teleoperation, Vision-Guided Control, Forward & Inverse Kinematics, and Future Digital Twin Integration**

</div>

---

# 📌 Overview

This project is a **5DOF intelligent robotic arm platform** designed for:

* Real-time teleoperation
* Computer vision-based control
* Forward Kinematics (FK)
* Inverse Kinematics (IK)
* Hand gesture tracking
* Wireless camera monitoring
* Future autonomous manipulation

The system uses **MediaPipe + OpenCV** to track hand motion from a laptop webcam and convert it into robotic arm commands. These commands are transmitted through **USB Serial** to an **Arduino Uno**, which drives the arm using a **PCA9685 servo controller**.

An **ESP32-CAM** mounted on the robotic arm provides a real-time end-effector view over WiFi.

---

# 🎯 Key Features

## Robotic Arm Control

* Forward Kinematics (FK)
* Inverse Kinematics (IK)
* Real-time servo control
* Workspace visualization
* Collision-aware arm simulation
* End-effector positioning

## Computer Vision

* MediaPipe hand tracking
* Hand landmark detection
* Hand centroid tracking
* Gesture recognition
* Pinch detection for gripper control
* Depth estimation using hand size

## User Interface

* Neon Dark Theme Dashboard
* FK / IK Mode Switching
* Real-time arm visualization
* Live camera feeds
* KP gain tuning
* Tracking pause/resume

## Communication

* USB Serial Communication
* WiFi Video Streaming
* PCA9685 Servo Control
* Expandable wireless architecture

---

# 🦾 Robotic Arm Specifications

## Degrees of Freedom (DOF)

| Joint    | Function            |
| -------- | ------------------- |
| Base     | Rotation            |
| Shoulder | Vertical Motion     |
| Elbow    | Reach Control       |
| Wrist    | Orientation         |
| Gripper  | Object Manipulation |

Total: **5 Degrees of Freedom**

---

# 📏 Mechanical Dimensions

| Segment       | Length  |
| ------------- | ------- |
| Base Height   | 12.5 cm |
| Shoulder Link | 18.5 cm |
| Elbow Link    | 14.5 cm |
| Wrist Link    | 17 cm   |
| Gripper Reach | 14 cm   |

---

# ⚙️ Hardware Components

## Electronics

| Component                   | Quantity           |
| --------------------------- | ------------------ |
| Arduino Uno                 | 1                  |
| PCA9685 Servo Driver        | 1                  |
| ESP32-CAM                   | 1                  |
| ESP32 Dev Board             | 1 (Future Upgrade) |
| Servo Motors (180°)         | 5                  |
| External Servo Power Supply | 1                  |
| Breadboard / PCB            | 1                  |
| Jumper Wires                | Multiple           |

---

## Vision & Communication

| Component     | Purpose             |
| ------------- | ------------------- |
| Laptop Camera | Hand Tracking       |
| ESP32-CAM     | End-Effector View   |
| WiFi Network  | Video Streaming     |
| USB Serial    | Robot Communication |

---

# 🧠 Software Stack

## Languages

* Python
* Arduino C++

## Libraries

```bash
opencv-python
mediapipe
numpy
pyserial
matplotlib
tkinter
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 🖥️ System Architecture

```text
Laptop Camera
      │
      ▼
OpenCV + MediaPipe
      │
      ▼
Hand Landmark Detection
      │
      ▼
Centroid & Gesture Processing
      │
      ▼
Forward / Inverse Kinematics
      │
      ▼
Serial Communication
      │
      ▼
Arduino Uno
      │
      ▼
PCA9685 Servo Driver
      │
      ▼
Servo Motors
      │
      ▼
Robotic Arm Motion
```

---

# 📷 Vision System

## Laptop Camera

Used for:

* Hand tracking
* Gesture recognition
* Centroid calculation
* Depth estimation
* Gripper control

### MediaPipe Detects

* 21 Hand Landmarks
* Finger Positions
* Palm Orientation
* Hand Centroid
* Pinch Distance

---

## ESP32-CAM

Mounted near the robotic arm end-effector.

Provides:

* Robot-eye view
* Visual feedback
* Object monitoring
* Future AI vision integration

---

# 📡 ESP32-CAM Streaming Pipeline

```text
ESP32-CAM
     │
     ▼
MJPEG Stream over WiFi
     │
     ▼
Laptop OpenCV Client
     │
     ▼
Frame Decoding
     │
     ▼
Dashboard Display
```

Python receives the stream using:

```python
urllib.request
cv2.imdecode()
```

---

# 🎮 Gesture-Based Control

## Motion Mapping

| Hand Motion        | Robot Motion  |
| ------------------ | ------------- |
| Left / Right       | Base Rotation |
| Up / Down          | Shoulder      |
| Forward / Backward | Elbow         |

---

## Gripper Control

Pinch detection is used:

```text
Thumb ↔ Index Finger Distance
```

| Gesture      | Action        |
| ------------ | ------------- |
| Pinch Closed | Close Gripper |
| Pinch Open   | Open Gripper  |

---

# 🧮 Forward Kinematics (FK)

The FK dashboard allows direct servo control through sliders.

### Features

* Real-time joint control
* Servo angle adjustment
* Arm visualization
* Workspace monitoring
* Collision avoidance checks
* 3D rendering support

---

# 🔄 Inverse Kinematics (IK)

The IK dashboard allows direct end-effector positioning.

### Control Inputs

* X Coordinate
* Y Coordinate
* Z Coordinate

### Automatically Solves

* Base Angle
* Shoulder Angle
* Elbow Angle
* Wrist Angle

using geometrical inverse kinematics.

---

# 🖥️ Dashboard Features

Built using **Tkinter + OpenCV + Matplotlib**.

### Includes

* FK Mode
* IK Mode
* Real-time 3D Arm Visualization
* Live Webcam Feed
* Live ESP32-CAM Feed
* KP Gain Tuning
* Pause / Resume Tracking
* Dark Neon UI

---

# 🔌 Communication System

## Current Architecture

```text
Laptop
   │
 USB Serial
   │
   ▼
Arduino Uno
   │
 I2C
   │
   ▼
PCA9685
   │
 PWM
   │
   ▼
Servo Motors
```

---

# 🚀 Future Wireless Architecture

An ESP32 will be added alongside the Arduino Uno.

Supported technologies:

| Technology | Purpose                   |
| ---------- | ------------------------- |
| WiFi       | Long Range Control        |
| Bluetooth  | Mobile App Control        |
| ESP-NOW    | Low Latency Communication |
| NRF24L01   | RF Communication          |

### Planned Architecture

```text
Wireless Transmitter
        │
        ▼
ESP32 Receiver
        │
        ▼
Arduino Uno
        │
        ▼
PCA9685
        │
        ▼
Servo Motors
```

---

# 🧠 Future Roadmap

## Digital Twin

Create a real-time virtual replica of the robotic arm synchronized with physical motion.

---

## Object Detection

Integrate:

* YOLO
* OpenCV
* ESP32-CAM

for autonomous target recognition.

---

## Autonomous Pick & Place

Future capabilities:

* Object classification
* Grasp planning
* Path planning
* Obstacle avoidance

---

## ROS2 Integration

Planned integration with:

* ROS2 Humble
* RViz
* Gazebo

for simulation and advanced robotics development.

---

## VR / AR Teleoperation

Future support for:

* VR robotic control
* AR overlays
* Immersive teleoperation

---

# 🏭 Industrial Applications

## Manufacturing

* Pick and place automation
* Assembly line operations
* Precision handling

## Warehouse Automation

* Parcel sorting
* Inventory handling
* Automated logistics

## Hazardous Environments

* Chemical handling
* Remote inspection
* Disaster response

## Medical Robotics

* Assistive robotics
* Teleoperation systems
* Precision manipulation

## Research & Education

* Robotics learning
* AI experimentation
* Computer vision research

---

# 🎯 Project Vision

The goal of this project is to develop a low-cost intelligent robotic manipulation platform that combines:

* Robotics
* Artificial Intelligence
* Computer Vision
* Embedded Systems
* Wireless Communication
* Real-Time Control Systems

while serving as a foundation for future autonomous robotic systems.

---

# 👨‍💻 Developer

**Abhinav Goel**



GitHub: https://github.com/Humobot1812

---

⭐ If you find this project interesting, consider starring the repository.
