import time

class TrackedObject:
    def __init__(self, object_id, class_id, bbox, confidence, alpha_dist=0.2, alpha_speed=0.1):
        self.object_id = object_id
        self.class_id = class_id
        self.bbox = bbox  # (x1, y1, x2, y2)
        self.confidence = confidence
        self.centroid = self.calculate_centroid()

        self.missed_frames = 0
        self.history_boxes = []
        self.distance = None
        self.speed = 0
        self.ttc = None
        self.closing_in = False
        
        self.last_timestamp = None
       
        self.alpha_dist = alpha_dist
        self.alpha_speed = alpha_speed


    def update(self, bbox, confidence):
        self.history_boxes.append(self.bbox)
        self.bbox = bbox
        self.confidence = confidence
        self.centroid = self.calculate_centroid()
        self.missed_frames = 0
        

    def calculate_centroid(self):
        x1, y1, x2, y2 = self.bbox
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        return (cx, cy)
    
    def increment_missed_frames(self):
        self.missed_frames += 1
    
    def calculate_IoU(self, other_bbox):
        sx1, sy1, sx2, sy2 = self.bbox
        ox1, oy1, ox2, oy2 = other_bbox

        # Calculate intersection
        inter_x1 = max(sx1, ox1)
        inter_y1 = max(sy1, oy1)
        inter_x2 = min(sx2, ox2)
        inter_y2 = min(sy2, oy2)

        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

        # Calculate union area
        box_area = (sx2 - sx1) * (sy2 - sy1)
        other_area = (ox2 - ox1) * (oy2 - oy1)
        union_area = box_area + other_area - inter_area

        return inter_area / union_area

    def compute_distance(self, focal_length = 1000, camera_height = 1.2, horizon_y = 540):
        y_bottom = self.bbox[3] # y2 of the bounding box
        delta_y = y_bottom - horizon_y

        if delta_y <= 2:
            return 150.0  # Return a large distance if too close to horizon

        distance = (focal_length * camera_height) / delta_y
        # Above 150 meters, is just big assumation
        return round(min(distance, 150.0), 2)
    
    def compute_speed(self, new_dist, old_dist, delta_time):
        if delta_time <= 0 or old_dist is None:
            return 0.0
        
        delta_dist = new_dist - old_dist
        speed = delta_dist / delta_time
        return round(speed, 2)
    
    def compute_ttc(self, speed, distance):
        if speed < -0.1:
            return round(distance / abs(speed), 1)
        return None

    def _low_pass_filter(self, new_val, old_val, alpha):
        # Using Alpha filter to smooth the distance estimation
        if old_val is None:
            return new_val
        else:
            return round(alpha * new_val + (1 - alpha) * old_val, 2)
    
    
    def update_metrics(self):
        now = time.time()
        if self.last_timestamp is None:
            self.last_timestamp = now
            return
        
        delta_time = now - self.last_timestamp
        raw_distance = self.compute_distance()
        prev_distance = self.distance
        self.distance = self._low_pass_filter(raw_distance, self.distance, self.alpha_dist)

        if prev_distance is not None and delta_time > 0:
            raw_speed = self.compute_speed(self.distance, prev_distance, delta_time)
            self.speed = self._low_pass_filter(raw_speed, self.speed, self.alpha_speed)

        self.ttc = self.compute_ttc(self.speed, self.distance)
        self.last_timestamp = now
   
    
    

        

     


    
