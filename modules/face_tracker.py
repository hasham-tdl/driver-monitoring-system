import cv2
import dlib
import numpy as np
import time
from collections import deque
from imutils import face_utils
from scipy.spatial import distance as dist


# ================= METRICS =================
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)


def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[13], mouth[19])
    B = dist.euclidean(mouth[14], mouth[18])
    C = dist.euclidean(mouth[15], mouth[17])
    D = dist.euclidean(mouth[12], mouth[16])
    return (A + B + C) / (3.0 * D)


# ================= FACE TRACKER =================
class FaceTracker:
    def __init__(self, predictor_path):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)

        # -------- THRESHOLDS (same as app) --------
        self.EYE_THRESH = 0.25
        self.MAR_THRESH = 0.70
        self.SLEEP_TIME = 1.5
        self.YAWN_TIME = 1.0
        
        self.UNCONSCIOUS_EAR = 0.2

        self.HEAD_TILT_THRESH = 15
        self.PERCLOS_THRESH = 0.2

        # -------- STATE --------
        self.state = None
        self.eyes_closed_start = None
        self.yawn_start = None

        # -------- TIMERS --------
        self.last_face_time = time.time()
        self.FACE_LOST_TIMEOUT = 0.5

        # -------- SMOOTHING --------
        self.ear_history = deque(maxlen=5)
        self.mar_history = deque(maxlen=5)
        self.perclos = deque(maxlen=900)

        # -------- LANDMARKS --------
        (self.lStart, self.lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (self.rStart, self.rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        (self.mStart, self.mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

    # ================= UPDATE =================
    def update(self, frame, now):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.createCLAHE(2.0, (8, 8)).apply(gray)
        overlay = np.zeros_like(gray)

        faces = self.detector(gray, 1)

        # ---------- NO FACE (IDENTICAL LOGIC) ----------
        if len(faces) == 0:
            face_lost = (now - self.last_face_time) > self.FACE_LOST_TIMEOUT

            if face_lost:
                self.state = None
                self.eyes_closed_start = None
                self.yawn_start = None
                self.perclos.clear()
                self.ear_history.clear()
                self.mar_history.clear()

            return self._output(
                face_found=False,
                face_lost=face_lost,
                frame=gray,
                overlay=overlay,
                debug_reason="face_not_detected"
            )

        # ---------- FACE PRESENT ----------
        self.last_face_time = now
        face = max(
            faces,
            key=lambda f: (f.right() - f.left()) * (f.bottom() - f.top())
        )

        shape = face_utils.shape_to_np(self.predictor(gray, face))

        # Draw landmarks
        for (x, y) in shape:
            cv2.circle(overlay, (x, y), 2, 255, -1)

        # ================= EAR =================
        ear_vals = []
        le = shape[self.lStart:self.lEnd]
        re = shape[self.rStart:self.rEnd]

        if np.linalg.norm(le[0] - le[3]) > 1:
            ear_vals.append(eye_aspect_ratio(le))
        if np.linalg.norm(re[0] - re[3]) > 1:
            ear_vals.append(eye_aspect_ratio(re))

        if not ear_vals:
            return self._output(
                face_found=True,
                face_lost=False,
                frame=gray,
                overlay=overlay,
                debug_reason="invalid_eye_geometry"
            )

        ear = min(ear_vals)
        self.ear_history.append(ear)
        ear = sum(self.ear_history) / len(self.ear_history)

        self.perclos.append(1 if ear < self.EYE_THRESH else 0)

        if ear < self.EYE_THRESH:
            if self.eyes_closed_start is None:
                self.eyes_closed_start = now
        else:
            self.eyes_closed_start = None

        # ================= MAR =================
        mar = mouth_aspect_ratio(shape[self.mStart:self.mEnd])
        self.mar_history.append(mar)
        mar = sum(self.mar_history) / len(self.mar_history)

        if mar > self.MAR_THRESH:
            if self.yawn_start is None:
                self.yawn_start = now
        else:
            self.yawn_start = None

        yawning = self.yawn_start and (now - self.yawn_start) > self.YAWN_TIME
        perclos_rate = sum(self.perclos) / len(self.perclos)

        # ================= HEAD / STATE LOGIC =================
        nose, chin = shape[33], shape[8]
        head_angle = np.degrees(np.arctan2(
            chin[1] - nose[1],
            chin[0] - nose[0]
        ))

        unconscious = (
            ear < self.UNCONSCIOUS_EAR and
            abs(head_angle) > self.HEAD_TILT_THRESH and
            self.eyes_closed_start and
            (now - self.eyes_closed_start) > 0.8
        )

        sleep_condition = (
            not yawning and (
                unconscious or
                (self.eyes_closed_start and (now - self.eyes_closed_start) > self.SLEEP_TIME)
            )
        )

        if sleep_condition:
            self.state = "SLEEPING"
        elif yawning or perclos_rate > self.PERCLOS_THRESH:
            self.state = "DROWSY"
        else:
            self.state = "AWAKE"

        return self._output(
            face_found=True,
            face_lost=False,
            frame=gray,
            overlay=overlay,
            state=self.state,
            ear=ear,
            mar=mar,
            perclos=perclos_rate,
            head_angle=head_angle
        )

    # ================= OUTPUT =================
    def _output(
        self,
        face_found,
        face_lost,
        frame,
        overlay,
        state=None,
        ear=None,
        mar=None,
        perclos=None,
        head_angle=None,
        debug_reason=None
    ):
        return {
            "face_found": face_found,
            "face_lost": face_lost,
            "state": state,
            "frame": frame,
            "overlay": overlay,
            "ear": ear,
            "mar": mar,
            "perclos": perclos,
            "head_angle": head_angle,
            "debug": {"reason": debug_reason}
        }
