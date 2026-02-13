# ADAS: Overtaking Estimation System (OES)

![Python](https://img.shields.io/badge/python-3.12-blue.svg)
![YOLOv8](https://img.shields.io/badge/vision-YOLOv8-green.svg)
![Status](https://img.shields.io/badge/status-Experimental_POC-orange.svg)

## 📌 Project Disclaimer
This is an **abstract experimental project** created as a personal deep-dive into Computer Vision (CV) and Object Detection (OD) within the automotive sector. The primary goal is to understand the underlying principles of camera geometry, temporal object tracking, and decision-making logic. It is a proof-of-concept for educational purposes, not a production-ready safety system.

## Showcase videos
<div align="center">
  <h3>🎥 Project Demonstration</h3>
  <video src="https://github.com/user-attachments/assets/8a0c4eb5-ad54-4eda-a705-0e446379aa90" width="100%" controls muted autoplay loop>
    Configurator launcher
  </video>
</div>

## Project Objectives
* **Perspective Understanding:** Mapping 2D image coordinates to 3D world distances using monocular camera geometry.
* **Signal Processing:** Implementing recursive estimation (Kalman filters) to mitigate noise in distance and velocity measurements.
* **Motion Analysis:** Classifying object behavior through trajectory analysis rather than simple bounding box changes.

## Tech Stack
* **Computer Vision:** `ultralytics` (YOLOv8), `OpenCV`
* **GPU Acceleration:** `PyTorch` with **CUDA 11.8** support
* **Filtering & Math:** `NumPy`, Custom Kalman Filter implementation
* **GUI & UI:** `CustomTkinter` (Unified launcher and real-time dashboard)
* **Performance:** Optimized inference with **FP16 precision**

## Core Features (Implemented)
* **Dynamic Calibration:** Real-time adjustment of FOV and Horizon-line to align the geometric model with the video perspective.
    
* **Trajectory-based Classification:** A robust decider that uses centroid motion vectors ($dx, dy$) to distinguish between:
    * **Oncoming:** Vehicles with high lateral shift and negative relative velocity.
    * **Following:** Vehicles moving within the ego-lane trajectory.
    * **Stationary:** Roadside objects/signs filtered by absolute velocity consistency.
* **Hybrid Telemetry Filtering:** Multi-stage filtering using Low-pass and **Kalman filters** to eliminate "pixel jitter" from YOLO detections.
    
* **Monocular Distance Estimation:** Distance is calculated based on the vertical offset from the horizon line using the formula:
    $$d = \frac{f \cdot H}{\Delta y}$$
* **TTC Logic:** Real-time calculation of **Time-to-Collision** to assess the overtaking window safety.

## System Workflow
1.  **Configuration:** `ADASLauncher` initializes camera parameters (FOV, Horizon) and YOLO settings.
2.  **Perception:** YOLOv8 extracts bounding boxes, which are then filtered for automotive classes.
3.  **Tracking & Metrics:** `ObjectTracker` maintains temporal consistency and calculates $V_{rel}$ and $V_{real}$ in $m/s$.
4.  **Decision Engine:** The system evaluates the risk based on the classified direction of travel and the calculated distance gap.


## Future Roadmap: The "Sim-to-Real" Phase
The current version relies on estimated parameters from 2D dashboard videos. To advance the project, the next steps include:
* **CARLA Simulator Integration:** Transitioning to a high-fidelity simulation environment to obtain **ground-truth** physics data.
* **Accuracy Benchmarking:** Comparing CV-based distance and speed estimates against the simulator's internal engine to measure and reduce error margins.
* **Vehicle Dynamics:** Incorporating ego-vehicle CAN bus data (speed, steering angle) for more precise motion compensation.

## Installation
To run the system with GPU acceleration, ensure you have an NVIDIA GPU and follow these steps:

1. **Install PyTorch (CUDA 11.8):**
   ```bash
   pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)

2. **Install remaining dependencies:**
    ```bash
   pip install -r requirements.txt

3. **Launch the app:**
    ```bash
   python main.py