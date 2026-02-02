import numpy as np

class TrackedObject:
    def __init__(self, object_id, class_id, bbox, confidence, alpha_dist=0.2, alpha_speed=0.1):
        self.object_id = object_id
        self.class_id = class_id
        self.bbox = bbox  # (x1, y1, x2, y2)
        self.confidence = confidence
        
        self.missed_frames = 0
        self.distance = None
        self.speed = 0        # Relative speed
        self.real_speed = 0   # Absolut speed
        self.ttc = None
        self.status = "UNKNOWN"
        
        self.alpha_dist = alpha_dist
        self.alpha_speed = alpha_speed
        self.prev_box_width = bbox[2] - bbox[0]
        self.width_box_growth = 1.0

    def update(self, bbox, confidence):
        self.bbox = bbox
        self.confidence = confidence
        self.missed_frames = 0

    def calculate_IoU(self, other_bbox):
        sx1, sy1, sx2, sy2 = self.bbox
        ox1, oy1, ox2, oy2 = other_bbox
        inter_x1, inter_y1 = max(sx1, ox1), max(sy1, oy1)
        inter_x2, inter_y2 = min(sx2, ox2), min(sy2, oy2)
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        box_area = (sx2 - sx1) * (sy2 - sy1)
        other_area = (ox2 - ox1) * (oy2 - oy1)
        union_area = box_area + other_area - inter_area
        return inter_area / union_area if union_area > 0 else 0

    def compute_distance(self, focal_length=400, camera_height=1.2, horizon_y=540):
        y_bottom = self.bbox[3]
        delta_y = y_bottom - horizon_y
        if delta_y <= 2: return 150.0
        distance = (focal_length * camera_height) / delta_y
        return round(min(distance, 150.0), 2)

    def _low_pass_filter(self, new_val, old_val, alpha):
        if old_val is None: return new_val
        return round(alpha * new_val + (1 - alpha) * old_val, 2)

    def direction_decider(self, v_thresh= 1.5, growth_thresh=1.01):
        if self.real_speed < -v_thresh and self.width_box_growth > growth_thresh:
            return "ONCOMING"

        if self.real_speed > v_thresh:
            return "FOLLOWING"

        if abs(self.real_speed) <= v_thresh:
            return "STATIONARY"

        return "UNKNOWN"

    def update_metrics(self, raw_distance, delta_time, ego_speed):
        if self.distance is None:
            self.distance = raw_distance
            return
        
        prev_distance = self.distance
        self.distance = self._low_pass_filter(raw_distance, self.distance, self.alpha_dist)

        if delta_time > 0:
            raw_rel_speed = (self.distance - prev_distance) / delta_time
            self.speed = self._low_pass_filter(raw_rel_speed, self.speed, self.alpha_speed)
            
            raw_real_speed = ego_speed + self.speed
            self.real_speed = self._low_pass_filter(raw_real_speed, self.real_speed, self.alpha_speed)

        actual_width = self.bbox[2] - self.bbox[0]
        raw_growth = actual_width / self.prev_box_width if self.prev_box_width > 0 else 1.0
        self.width_box_growth = self._low_pass_filter(raw_growth, self.width_box_growth, 0.3)
        self.prev_box_width = actual_width

        self.status = self.direction_decider()
        if self.speed < -0.1:
            self.ttc = round(self.distance / abs(self.speed), 1)
        else:
            self.ttc = None