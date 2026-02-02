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
        self.allowed_objects = [2, 3, 5, 7]  # auto, motorka, autobus, náklaďák
        self.gui_manager_video = GUIManager(window_name="ADAS Overtaking Estimation", width=1280, height=720)
        self.gui_manager_calib = GUIManager(window_name="Calibration", width=600, height=400)  
        self.object_tracker = ObjectTracker()
        self.dt = 1.0 / 30  # Výchozí hodnota
        self.pov_speed = 0.0

    def process_frame(self, frame, score_thresh=0.45, device='cpu'):
        results = self.yolo_model.predict(
            frame, imgsz=640, conf=score_thresh,
            device=0 if 'cuda' in str(device) else 'cpu', verbose=False
        )

        frame_detections = []
        for r in results:
            for box in r.boxes:
                class_id = int(box.cls[0])
                if class_id in self.allowed_objects:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = box.conf[0]
                    frame_detections.append((class_id, (x1, y1, x2, y2), conf))
        
        # Tracking
        tracked_objects = self.object_tracker.update(frame_detections)

        focal_length = self.gui_manager_calib.get_trackbar_value("FocalLength")
        horizon_y = self.gui_manager_calib.get_trackbar_value("Horizon")
        cv2.line(frame, (0, horizon_y), (frame.shape[1], horizon_y), (255, 255, 255), 1)

        self.pov_speed = self.object_tracker.get_speed_median()

        for obj in tracked_objects:
            ox1, oy1, ox2, oy2 = obj.bbox
            raw_distance = obj.compute_distance(focal_length=focal_length, horizon_y=horizon_y)
            
            obj.update_metrics(raw_distance, self.dt, self.pov_speed)

            color = (255, 255, 111) 
            if obj.ttc is not None:
                if obj.ttc < 20.0: color = (50, 50, 255) 
                else: color = (0, 255, 255) 

            label = f"{obj.status} Dist: {obj.distance}m V_rel: {round(obj.real_speed*3.6, 1)}km/h TTC: {obj.ttc}s"
            cv2.rectangle(frame, (ox1, oy1), (ox2, oy2), color, 2)
            cv2.putText(frame, label, (ox1, oy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return frame

    @staticmethod
    def nothing(x): pass

    def create_calibration_window(self):
        self.gui_manager_calib.create_trackbar("Horizon", 540, 1080, self.nothing)
        self.gui_manager_calib.create_trackbar("FocalLength", 400, 2000, self.nothing)

    def run_video(self):
        cap = cv2.VideoCapture(self.video_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps == 0: video_fps = 30
        self.dt = 1.0 / video_fps 
        prev_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret: break

            if cv2.getWindowProperty("Calibration", cv2.WND_PROP_VISIBLE) >= 1:
                self.gui_manager_calib.display_window(np.zeros((100, 600, 3), np.uint8))

            processed_frame = self.process_frame(frame)
            
            curr_time = time.time()
            proc_dt = curr_time - prev_time
            prev_time = curr_time
            proc_fps = 1 / proc_dt if proc_dt > 0 else 0

            cv2.putText(processed_frame, f"PC Performance FPS: {int(proc_fps)} | Ego-Speed: {round(self.pov_speed*3.6, 1)} km/h", 
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            self.gui_manager_video.display_window(processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        cap.release()
        self.gui_manager_video.close()