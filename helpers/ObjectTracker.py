import numpy as np
from helpers.TrackedObject import TrackedObject

class ObjectTracker:
    def __init__(self, iou_threshold=0.3, max_missed_frames=5):
        self.tracked_objects = []
        self.next_object_id = 0
        self.iou_threshold = iou_threshold
        self.max_missed_frames = max_missed_frames

    def update(self, detections):
        matched_detections = set()

        for obj in self.tracked_objects:
            best_iou = 0
            best_det_index = -1
            for i, (_, bbox, _) in enumerate(detections):
                if i in matched_detections: continue
                calculated_iou = obj.calculate_IoU(bbox)
                if calculated_iou > best_iou:
                    best_iou, best_det_index = calculated_iou, i

            if best_iou >= self.iou_threshold:
                _, bbox, confidence = detections[best_det_index]
                obj.update(bbox, confidence)
                matched_detections.add(best_det_index)
            else:
                obj.missed_frames += 1

        self.tracked_objects = [obj for obj in self.tracked_objects if obj.missed_frames <= self.max_missed_frames]

        for i, det in enumerate(detections):
            if i not in matched_detections:
                class_id, bbox, confidence = det
                self.tracked_objects.append(TrackedObject(self.next_object_id, class_id, bbox, confidence))
                self.next_object_id += 1
        
        return self.tracked_objects

    def get_speed_median(self):
        speeds = [abs(obj.speed) for obj in self.tracked_objects 
                  if obj.speed is not None and obj.status == "FOLLOWING"]
        if not speeds: return 25.0 
        return float(np.median(speeds))