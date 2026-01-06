import cv2
import numpy as np
import time
from ultralytics import YOLO

UPPER_BODY_KP = [0,1,2,3,4,5,6,7,8,9,10]


class BodyTracker:
    def __init__(self, model_path):
        self.yolo = YOLO(model_path)
        self.yolo_tilt_start = None
        self.YOLO_SLEEP_TIME = 2.0

    def update(self, frame, now, face_lost):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.createCLAHE(2.0, (8, 8)).apply(gray)

        res = self.yolo(frame, verbose=False)[0]
        if res.keypoints is None or res.boxes is None:
            self.yolo_tilt_start = None
            return {
                "sleeping": False,
                "face_obstructed": False,
                "frame": gray
            }

        boxes = res.boxes.xyxy.cpu().numpy()
        idx = np.argmax((boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1]))
        kp = res.keypoints.xy[idx].cpu().numpy()

        le, re = kp[1], kp[2]
        ls, rs = kp[5], kp[6]
        nose = kp[0]

        shoulder_vec = rs - ls
        shoulder_angle = np.degrees(np.arctan2(shoulder_vec[1], shoulder_vec[0]))

        eye_vec = re - le
        eye_angle = np.degrees(np.arctan2(eye_vec[1], eye_vec[0]))

        roll = (eye_angle - shoulder_angle + 180) % 360 - 180

        state = "UPRIGHT"
        if roll > 12:
            state = "RIGHT"
        elif roll < -12:
            state = "LEFT"

        if state != "UPRIGHT":
            if self.yolo_tilt_start is None:
                self.yolo_tilt_start = now
        else:
            self.yolo_tilt_start = None

        sleeping = (
            self.yolo_tilt_start and
            (now - self.yolo_tilt_start) >= self.YOLO_SLEEP_TIME
        )

        face_obstructed = face_lost and state == "UPRIGHT"

        for i in UPPER_BODY_KP:
            x, y = kp[i]
            cv2.circle(gray, (int(x), int(y)), 4, 255, -1)

        return {
            "sleeping": sleeping,
            "face_obstructed": face_obstructed,
            "frame": gray
        }
