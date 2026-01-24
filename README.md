# ADAS: Overtaking Estimation System (OES)

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![YOLOv8](https://img.shields.io/badge/vision-YOLOv8-green.svg)
![Status](https://img.shields.io/badge/status-MVP_Development-orange.svg)

## 🇸🇪 Project Overview
Developed as a high-fidelity ADAS (Advanced Driver Assistance System) proof-of-concept. The goal is to assist drivers in making safe overtaking maneuvers on two-lane roads by analyzing oncoming traffic and estimating the "Safe-to-Pass" window.

This project focuses on **low-latency detection** and **deterministic risk assessment**.

## 🚀 Core Features (MVP Roadmap)
* **Object Detection:** Real-time vehicle detection using YOLOv8 (Inference optimized for edge cases).
* **Distance Estimation:** Monocular distance estimation using camera geometry and Inverse Perspective Mapping (IPM).
* **Relative Velocity Calculation:** Tracking objects over time to estimate closing speeds of oncoming traffic.
* **Overtaking Logic:** A decision-making engine that calculates the required time-to-collision (TTC) versus the overtaking window.
* **Visual Debugger:** Overlay showing safety zones (Green/Red) directly on the HUD/Video feed.

## 🛠 Tech Stack
- **Language:** Python 3.10+
- **Computer Vision:** OpenCV, Ultralytics YOLOv8
- **Math/Physics:** NumPy (Coordinate transformations, Kinematics)
- **Version Control:** Git (Feature-branch workflow)

## 📈 System Architecture
1. **Perception Layer:** Raw video input -> YOLOv8 Detection -> Bounding Box Filtering.
2. **Tracking Layer:** Temporal consistency (tracking IDs) to prevent flickering.
3. **Geometry Layer:** Mapping 2D image coordinates to 3D world coordinates (Ground Plane).
4. **Decision Layer:** Risk assessment based on distance, speed, and acceleration.

## 🛠 Installation & Usage
(To be updated as we progress)
```bash
git clone [https://github.com/net0045/ADAS_overtaking_estimatation](https://github.com/net0045/ADAS_overtaking_estimatation)
cd ADAS_overtaking_estimatation
pip install -r requirements.txt
python main.py