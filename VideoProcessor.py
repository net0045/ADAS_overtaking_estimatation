import cv2
import numpy as np
from ultralytics import YOLO
import time
import torch 
from helpers.GUIManager import GUIManager
from helpers.ObjectTracker import ObjectTracker

class VideoProcessor:
    def __init__(self, video_path, model_name="yolov8n.pt"):
        self.video_path = video_path
        self.yolo_model = YOLO(model_name)
        self.allowed_objects = [2, 3, 5, 7]  # auto, motorka, autobus, náklaďák
        self.class_names = self.yolo_model.names
        self.gui_manager_video = GUIManager(window_name="ADAS Overtaking Estimation", width=1280, height=720) 
        self.object_tracker = ObjectTracker()
        self.dt = 1.0 / 30  
        self.pov_speed = 0.0
        self.focal_length = 400

    def set_initial_config(self, horizon_y, fov, yolo_thrs, yolo_imgsz):
        self.horizon_y = horizon_y
        self.fov = fov
        self.yolo_thrs = yolo_thrs
        self.yolo_imgsz = yolo_imgsz

    def process_frame(self, frame):
        chosen_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        results = self.yolo_model.predict(
            frame, imgsz=self.yolo_imgsz, conf=self.yolo_thrs,
            device=0 if 'cuda' in str(chosen_device) else 'cpu', verbose=False
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

        horizon_y = self.horizon_y
        focal_length = self.focal_length
        cv2.line(frame, (0, self.horizon_y), (frame.shape[1], self.horizon_y), (255, 255, 255), 1)

        #self.pov_speed = self.object_tracker.get_speed_median()

        for obj in tracked_objects:
            ox1, oy1, ox2, oy2 = obj.bbox
            raw_distance = obj.compute_distance(focal_length=focal_length, horizon_y=horizon_y)
            class_name = self.class_names.get(obj.class_id, "Unknown")
            
            obj.update_metrics(raw_distance, self.dt, self.pov_speed)

            color = (255, 255, 111) 
            if obj.ttc is not None:
                if obj.ttc < 20.0: color = (50, 50, 255) 
                else: color = (0, 255, 255) 

            top_label = f"{obj.status} {class_name}"
            label = f"Dist: {obj.distance}m V: {round((obj.real_speed)*3.6, 1)}km/h"
            ttc_label = f"TTC: {obj.ttc}s"
            cv2.rectangle(frame, (ox1, oy1), (ox2, oy2), color, 2)
            cv2.putText(frame, top_label, (ox1, oy1 - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 111) , 2)
            cv2.putText(frame, label, (ox1, oy1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, ttc_label, (ox1, oy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return frame

    @staticmethod
    def nothing(x): pass

    def run_video(self):
        
        cap = cv2.VideoCapture(self.video_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fov_rad = np.deg2rad(self.fov)
        self.focal_length = (width / 2) / np.tan(fov_rad / 2)

        if video_fps == 0: video_fps = 30
        self.dt = 1.0 / video_fps 
        prev_time = time.time()

        while self.gui_manager_video.running:
            ret, frame = cap.read()
            if not ret: break

            self.pov_speed = self.gui_manager_video.get_ego_speed()

            if cv2.getWindowProperty("Calibration", cv2.WND_PROP_VISIBLE) >= 1:
                self.gui_manager_calib.display_window(np.zeros((100, 600, 3), np.uint8))

            processed_frame = self.process_frame(frame)

            final_frame = self.draw_hud(processed_frame, self.pov_speed, self.object_tracker.tracked_objects)

            curr_time = time.time()
            proc_dt = curr_time - prev_time
            prev_time = curr_time
            proc_fps = 1 / proc_dt if proc_dt > 0 else 0
            cv2.putText(final_frame, f"PC Performance FPS: {int(proc_fps)} | Resolution: {width}x{height} | FOCAL_LENGTH: {round(self.focal_length, 2)}", 
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (125, 100, 25), 3)
           

            self.gui_manager_video.display_window(final_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        cap.release()
        self.gui_manager_video.close()

    def draw_hud(self, frame, speed_ms, tracked_objects):
        height, width = frame.shape[:2]
        
        overlay = frame.copy()
        BG_COLOR = (20, 20, 20)     
        ACCENT_COLOR = (155, 255, 25) 
        DANGER_COLOR = (50, 50, 255)  
        
        panel_h = 100
        cv2.rectangle(overlay, (0, height - panel_h), (width, height), BG_COLOR, -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # 2. LOGIKA NEBEZPEČÍ
        # Kontrolujeme, zda je nějaký objekt "ONCOMING" a má nízké TTC
        is_danger = any(obj.status == "ONCOMING" and obj.ttc is not None and obj.ttc < 10.0 
                        for obj in tracked_objects)
        
        status_text = "OVERTAKING DANGER" if is_danger else "SAFE TO OVERTAKE"
        status_color = DANGER_COLOR if is_danger else ACCENT_COLOR

        speed_kh = int(speed_ms * 3.6)
        max_speed = 180
        bar_w = 400
        bar_x = (width // 2) - (bar_w // 2)
        bar_y = height - 40 
        
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 8), (60, 60, 60), -1)
        
        fill_w = int((min(speed_kh, max_speed) / max_speed) * bar_w)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + 8), ACCENT_COLOR, -1)
        cv2.putText(frame, f"{speed_kh}", (width // 2 - 40, height - 60), 
                    cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2)
        cv2.putText(frame, "km/h", (width // 2 + 35, height - 65), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.putText(frame, status_text, (width // 2 - 110, height - 12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.polylines(frame, [np.array([
            [width // 2 - 250, height - 20],
            [width // 2 - 300, height - 60],
            [width // 2 - 450, height - 60]
        ])], False, ACCENT_COLOR, 2)
        
        cv2.polylines(frame, [np.array([
            [width // 2 + 250, height - 20],
            [width // 2 + 300, height - 60],
            [width // 2 + 450, height - 60]
        ])], False, ACCENT_COLOR, 2)

        return frame