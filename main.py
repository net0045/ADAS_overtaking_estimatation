import numpy as np
import cv2
from pathlib import Path
from VideoProcessor import VideoProcessor

BASE_DIR = Path(__file__).resolve().parent
videopath = BASE_DIR / "assets" / "overtaking_1.MOV"


def main():
    print("Starting ADAS Overtaking Estimation...")
    print(f"Processing video file at: {videopath}")
    videoProcessor = VideoProcessor(videopath)
    videoProcessor.create_calibration_window()
    videoProcessor.run_video()

if __name__ == "__main__":
    main()
    