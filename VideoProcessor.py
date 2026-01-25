import cv2
import numpy as np
from ultralytics import YOLO
import time
from helpers.GUIManager import GUIManager
from helpers.ObjectTracker import ObjectTracker

class VideoProcessor:
    def __init__(self, video_path, model_name="yolov8n.pt"):
        self.video_path = video_path
        self.yolo_model = YOLO(model_name)
        self.allowed_objects = [2, 3, 5, 7]  # car, motorcycle, bus, truck
        self.gui_manager = GUIManager()
        self.object_tracker = ObjectTracker()

    def process_frame(self, frame, score_thresh=0.5, device='cpu'):
        model = self.yolo_model
        results = model.predict(
            frame,
            imgsz=320,
            conf=score_thresh,
            device=0 if 'cuda' in str(device) else 'cpu',
            verbose=False
        )

        frame_detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                class_id = int(box.cls[0]) # get the most probable class id
                if class_id in self.allowed_objects:
                    # Extract bounding box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = box.conf[0]
                    frame_detections.append((class_id, (x1, y1, x2, y2), conf))
        
        # Update tracked objects
        tracked_objects = self.object_tracker.update(frame_detections)

        # Draw bounding boxes around tracked objects
        for obj in tracked_objects:
            ox1, oy1, ox2, oy2 = obj.bbox
    
            # Update distance, speed, and TTC
            obj.update_metrics()

            # Set color based on TTC
            color = (255, 255, 111)
            if obj.ttc is not None:
                if obj.ttc < 20.0: 
                    color = (50, 50, 255) # car is closing in and TTC < 20s
                elif obj.ttc > 20.0: 
                    color = (0, 255, 255) # car is closing in but TTC > 20s

            label = f"ID {obj.object_id} Dist: {obj.distance}m V: {round(obj.speed*3.6, 1)}km/h TTC: {obj.ttc}s"
            cv2.rectangle(frame, (ox1, oy1), (ox2, oy2), color, 2)
            cv2.putText(frame, label, (ox1, oy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            

        return frame


    def run_video(self):
        # Open the video file 
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print("Error: Could not open video.")
            return

        # Get starttime of the calculation loop
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process the frame
            processed_frame = self.process_frame(frame)
            
            # Calculate FPS for real-time display
            current_time = time.time()
            delta_time = current_time - start_time
            start_time = current_time

            fps = 1 / delta_time if delta_time > 0 else 0

            # Vizualization
            cv2.putText(processed_frame, f"FPS: {int(fps)}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.line(frame, (0, 540), (frame.shape[1], 540), (255, 255, 255), 1)
            
            self.gui_manager.display_window(processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                break

        cap.release()
        self.gui_manager.close()