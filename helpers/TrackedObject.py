class TrackedObject:
    def __init__(self, object_id, class_id, bbox, confidence):
        self.object_id = object_id
        self.class_id = class_id
        self.bbox = bbox  # (x1, y1, x2, y2)
        self.confidence = confidence
        self.centroid = self.calculate_centroid()
        self.missed_frames = 0
        self.history_boxes = []

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


    
