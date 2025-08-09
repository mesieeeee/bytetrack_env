from ultralytics import YOLO
import cv2

class PoseEstimator:
    def __init__(self, model_path="yolov8n-pose.pt"):
        self.model = YOLO(model_path)

    def get_keypoints(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        cropped = frame[y1:y2, x1:x2]
        results = self.model.predict(cropped, conf=0.3)

        if not results or not results[0].keypoints or len(results[0].keypoints.xy) == 0:
            return []

        keypoints = results[0].keypoints.xy[0].cpu().numpy()
        keypoints[:, 0] += x1
        keypoints[:, 1] += y1
        return keypoints.tolist()
