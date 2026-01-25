import numpy as np
import cv2
from pathlib import Path
from VideoProcessor import VideoProcessor

BASE_DIR = Path(__file__).resolve().parent
videopath = BASE_DIR / "assets" / "driving_clip.mp4"


def main():
    print("Starting ADAS Overtaking Estimation...")
    print(f"Processing video file at: {videopath}")
    videoProcessor = VideoProcessor(videopath)
    videoProcessor.run_video()

if __name__ == "__main__":
    main()
    