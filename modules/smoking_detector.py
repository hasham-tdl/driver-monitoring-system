import cv2
import numpy as np
from ultralytics import YOLO


class SmokingDetector:
    def __init__(self, model_path="best.pt", conf_thresh=0.5):
        """
        model_path : path to YOLO smoking model
        conf_thresh: confidence threshold for smoking detection
        """
        self.model = YOLO(model_path)
        self.conf_thresh = conf_thresh

    def update(self, frame):
        """
        Args:
            frame (np.ndarray): BGR frame

        Returns:
            dict:
                smoking (bool)
                confidence (float)
                overlay (np.ndarray or None)
        """

        if frame is None:
            return {
                "smoking": False,
                "confidence": 0.0,
                "overlay": None
            }

        # 1) Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 2) Convert grayscale → 3-channel (YOLO expects 3 channels)
        gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # 3) Run YOLO inference
        results = self.model(gray_3ch, verbose=False)

        smoking_detected = False
        max_conf = 0.0

        # 4) Parse detections
        for r in results:
            if r.boxes is None:
                continue

            for box in r.boxes:
                conf = float(box.conf[0])
                if conf >= self.conf_thresh:
                    smoking_detected = True
                    max_conf = max(max_conf, conf)

        # 5) Draw overlay (same size as frame)
        overlay = np.zeros_like(frame)
        if smoking_detected:
            overlay = results[0].plot()

        return {
            "smoking": smoking_detected,
            "confidence": max_conf,
            "overlay": overlay if smoking_detected else None
        }
