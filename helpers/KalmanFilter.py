class KalmanFilter:
    def __init__(self, Q=0.01, R=0.1, initial_value=0):
        self.Q = Q # Process noise covariance - how much to change the value
        self.R = R # Measurement noise covariance - how much trust to YOLO detections
        self.x = initial_value # Estimated value
        self.P = 1.0          # Estimation error covariance

    def update(self, measurement):
        self.P += self.Q

        K = self.P / (self.P + self.R)
        self.x += K * (measurement - self.x)
        self.P = (1 - K) * self.P

        return self.x