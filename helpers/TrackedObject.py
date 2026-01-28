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
        self.is_oncoming = "UNKNOWN"
        
        self.last_timestamp = None
       
        self.alpha_dist = alpha_dist
        self.alpha_speed = alpha_speed

        self.prev_box_width = bbox[2] - bbox[0]
        self.width_box_growth = 1.0


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

    def compute_distance(self, focal_length = 400, camera_height = 1.2, horizon_y = 540):
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
        
    def compute_width_growth(self, actual_width, previous_width):
        if previous_width == 0:
            return 1.0  
        growth = actual_width / previous_width
        return growth

    
    def direction_decider(self, gwlt = 0.98, gwht = 1.02):
        # For now only based on width growth
        if self.width_box_growth < gwlt:
            return "FOLLOWING"
        elif self.width_box_growth > gwht:
            return "ONCOMING"
        else:
            return "STATIONARY"
    
    
    def update_metrics(self, raw_distance, delta_time):
        # For first calculation
        if self.distance is None:
            self.distance = raw_distance
            return
        
        prev_distance = self.distance
        self.distance = self._low_pass_filter(raw_distance, self.distance, self.alpha_dist)

        if prev_distance is not None and delta_time > 0:
            raw_speed = self.compute_speed(self.distance, prev_distance, delta_time)
            self.speed = self._low_pass_filter(raw_speed, self.speed, self.alpha_speed)

        prev_box_width = self.prev_box_width
        actual_box_width = self.bbox[2] - self.bbox[0]
        raw_box_with_growth = self.compute_width_growth(actual_box_width, prev_box_width)
        self.width_box_growth = self._low_pass_filter(raw_box_with_growth, self.width_box_growth, 0.3)
        self.prev_box_width = actual_box_width

        self.is_oncoming = self.direction_decider()

        self.ttc = self.compute_ttc(self.speed, self.distance)
   
    
    

        

     


    
