import numpy as np
import cv2
import torch
from pathlib import Path
from VideoProcessor import VideoProcessor
from helpers.ADASLauncher import ADASLauncher 

def start_adas(config):
    print("Launching ADAS Engine with selected config...")
    
    videoProcessor = VideoProcessor(config["video_path"]) 
    videoProcessor.set_initial_config(
        horizon_y=config["horizon"],
        fov=config["fov"],
        yolo_thrs=config["thresh"],
        yolo_imgsz=config["imgsz"])
    
    videoProcessor.run_video()

def main():
    print("Starting ADAS Configuration Launcher...")
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"CUDA is available! Using: {device_name}")
    else:
        print("CUDA not found. Running on CPU")
    
    app = ADASLauncher(start_callback=start_adas)
    app.mainloop()

if __name__ == "__main__":
    main()