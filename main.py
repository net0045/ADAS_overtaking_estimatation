import numpy as np
import cv2
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
    
    app = ADASLauncher(start_callback=start_adas)
    app.mainloop()

if __name__ == "__main__":
    main()