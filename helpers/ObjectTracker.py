import cv2
from helpers.TrackedObject import TrackedObject

class ObjectTracker:
    def __init__(self, iou_threshold=0.3, max_missed_frames=5):
        self.tracked_objects = [] # List of instances TrackedObject
        self.next_object_id = 0
        self.iou_threshold = iou_threshold
        self.max_missed_frames = max_missed_frames
    
    # Detections from YOLO model: list of tuples (class_id, (x1, y1, x2, y2), confidence)
    def update(self, detections):
        matched_detections = set()

        # Matching existing tracked objects with new detections
        for obj in self.tracked_objects:
            best_iou = 0
            best_det_index = -1

            for i, (_, bbox, _) in enumerate(detections):
                if i in matched_detections:
                    continue
                
                calculated_iou = obj.calculate_IoU(bbox)
                if calculated_iou > best_iou:
                    best_iou = calculated_iou
                    best_det_index = i

            if best_iou >= self.iou_threshold:
                _, bbox, confidence = detections[best_det_index]
                obj.update(bbox, confidence)
                matched_detections.add(best_det_index)
            else:
                obj.increment_missed_frames()
            
        # Remove objects with too many missed frames
        self.tracked_objects = [
            obj for obj in self.tracked_objects 
            if obj.missed_frames <= self.max_missed_frames
        ]    
            
        # Create new tracked objects for unmatched detections
        for i, det in enumerate(detections):
            if i not in matched_detections:
                class_id, bbox, confidence = det
                new_object = TrackedObject(self.next_object_id, class_id, bbox, confidence)
                self.tracked_objects.append(new_object)
                self.next_object_id += 1  
        
        return self.tracked_objects

        

                
                
