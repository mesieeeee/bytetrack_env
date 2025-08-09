import cv2

class PoseDrawer:
    def __init__(self):
        self.skeleton = [
            (5, 7), (7, 9),     # left arm
            (6, 8), (8, 10),    # right arm
            (5, 6), (5, 11), (6, 12),  # torso
            (11, 13), (13, 15), # left leg
            (12, 14), (14, 16), # right leg
            (11, 12)
        ]

    def draw(self, frame, keypoints):
        for x, y in keypoints:
            cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)

        for start, end in self.skeleton:
            if start < len(keypoints) and end < len(keypoints):
                x1, y1 = keypoints[start]
                x2, y2 = keypoints[end]
                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

        return frame
