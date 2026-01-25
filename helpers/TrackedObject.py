class TrackedObject:
    def __init__(self, object_id, class_id, bbox, confidence, alpha=0.2):
        self.object_id = object_id
        self.class_id = class_id
        self.bbox = bbox  # (x1, y1, x2, y2)
        self.confidence = confidence
        self.centroid = self.calculate_centroid()
        self.missed_frames = 0
        self.history_boxes = []
        self.distance = None
        self.alpha = 0.2
        

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
    
    def update_distance(self, new_distance):
        # Using ALpha filter to smooth the distance estimation
        if self.distance is None:
            self.distance = new_distance
        else:
            self.distance = self.alpha * new_distance + (1 - self.alpha) * self.distance
        return round(self.distance, 2)
     


    
