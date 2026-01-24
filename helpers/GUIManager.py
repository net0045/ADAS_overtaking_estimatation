import cv2 

class GUIManager:
    def __init__(self, window_name="ADAS Overtaking Estimation", width=1280, height=720):
        self.window_name = window_name
        self.width = width
        self.height = height
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.width, self.height)

    def display_window(self, frame):
        h, w = frame.shape[:2]
        if w != self.width or h != self.height:
            aspect_ratio = h / w
            new_height = int(self.width * aspect_ratio)
            frame = cv2.resize(frame, (self.width, new_height))

        cv2.imshow(self.window_name, frame)

    def close(self):
        cv2.destroyAllWindows()