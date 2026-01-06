
#3
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.popup import Popup
from kivy.uix.filechooser import FileChooserIconView
from kivy.clock import Clock
from kivy.graphics.texture import Texture

import cv2
import dlib
import numpy as np
from imutils import face_utils
from scipy.spatial import distance as dist
import time
from collections import deque

from ultralytics import YOLO

# ================= PATHS =================
predictor_path = r"shape_predictor\shape_predictor_68_face_landmarks.dat"
video_folder = r"videos"
YOLO_MODEL = "yolov8n-pose.pt"

UPPER_BODY_KP = [0,1,2,3,4,5,6,7,8,9,10]

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

# ================= APP =================
class DriverMonitoringApp(App):

    def build(self):
        self.face_obstructed = False
        self.capture = None
        self.video_path = None
        self.use_webcam = False

        # -------- MODELS --------
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)
        self.yolo = YOLO(YOLO_MODEL)

        # -------- THRESHOLDS --------
        self.EYE_THRESH = 0.24 # directly propotional to sensitivity 
        self.MAR_THRESH = 0.70 # inversly prop
        self.SLEEP_TIME = 1.5 # how long eyes have to be closed to to be considered sleepign
        self.YAWN_TIME = 1.0
        self.UNCONSCIOUS_EAR = 0.18
        self.HEAD_TILT_THRESH = 15
        self.PERCLOS_THRESH = 0.23

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

        self.yawn_events = deque()


        # -------- YOLO TEMPORAL SLEEP --------
        self.yolo_tilt_start = None
        self.YOLO_SLEEP_TIME = 2.0

        (self.lStart, self.lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (self.rStart, self.rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        (self.mStart, self.mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

        # -------- UI --------
        layout = BoxLayout(orientation="vertical")
        self.image = Image()
        layout.add_widget(self.image)

        controls = BoxLayout(size_hint_y=None, height=50)
        controls.add_widget(Button(text="Load Video", on_press=self.open_file_chooser))
        controls.add_widget(Button(text="Use Webcam", on_press=self.start_webcam))
        controls.add_widget(Button(text="Play", on_press=self.start))
        controls.add_widget(Button(text="Stop", on_press=self.stop))
        layout.add_widget(controls)

        return layout

    # ================= INPUT =================
    def open_file_chooser(self, _):
        chooser = FileChooserIconView(path=video_folder, filters=["*.mp4"])
        box = BoxLayout(orientation="vertical")
        box.add_widget(chooser)
        btn = Button(text="Select", size_hint_y=None, height=50)
        box.add_widget(btn)

        popup = Popup(title="Select Video", content=box, size_hint=(0.9, 0.9))

        def select(_):
            if chooser.selection:
                self.video_path = chooser.selection[0]
                self.use_webcam = False
                popup.dismiss()

        btn.bind(on_press=select)
        popup.open()

    def start_webcam(self, _):
        self.stop(None)
        self.use_webcam = True

        for i in range(5):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                self.capture = cap
                break

        if not self.capture:
            print("❌ Webcam not found")
            return

        Clock.schedule_interval(self.update, 1 / 30)

    def start(self, _):
        if self.use_webcam or not self.video_path:
            return
        self.stop(None)
        self.capture = cv2.VideoCapture(self.video_path)
        Clock.schedule_interval(self.update, 1 / 30)

    def stop(self, _):
        if self.capture:
            Clock.unschedule(self.update)
            self.capture.release()
            self.capture = None
            self.image.texture = None

    # ================= MAIN LOOP =================
    def update(self, dt):
        ret, frame = self.capture.read()
        if not ret:
            return

        now = time.time()

        # ---------- GLOBAL ILLUMINATION NORMALIZATION ----------
        gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_full = cv2.createCLAHE(2.0, (8, 8)).apply(gray_full)

        faces = self.detector(gray_full, 1)

        # ---------- NO FACE ----------
        if len(faces) == 0:
            face_lost = (now - self.last_face_time) > self.FACE_LOST_TIMEOUT

            if face_lost:
                self.state = None
                self.eyes_closed_start = None
                self.yawn_start = None
                self.perclos.clear()
                self.ear_history.clear()
                self.mar_history.clear()

            # Let YOLO decide posture
            self.process_yolo(frame, face_lost=face_lost)
            return

        # ---------- FACE PRESENT ----------
        self.last_face_time = now
        face = max(
            faces,
            key=lambda f: (f.right() - f.left()) * (f.bottom() - f.top())
        )

        shape = face_utils.shape_to_np(self.predictor(gray_full, face))
        # ---------- DRAW ALL LANDMARKS (FULL FRAME) ----------
        for (x, y) in shape:
            cv2.circle(gray_full, (int(x), int(y)), 2, 255, -1)


        # ======================================================
        # =============== VIEW-ANGLE NORMALIZATION =============
        # ======================================================

        # Eye centers
        left_eye = shape[self.lStart:self.lEnd].mean(axis=0)
        right_eye = shape[self.rStart:self.rEnd].mean(axis=0)

        # Angle between eyes
        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dy, dx))

        # Desired canonical eye distance
        desired_eye_dist = 80.0
        current_eye_dist = np.linalg.norm(right_eye - left_eye)
        scale = desired_eye_dist / (current_eye_dist + 1e-6)

        # Rotation matrix
        eyes_center = (
            float((left_eye[0] + right_eye[0]) / 2.0),
            float((left_eye[1] + right_eye[1]) / 2.0)
        )
        M = cv2.getRotationMatrix2D(tuple(eyes_center), angle, scale)

        # Warp original frame
        aligned = cv2.warpAffine(
            frame,
            M,
            (frame.shape[1], frame.shape[0]),
            flags=cv2.INTER_CUBIC
        )

        # Transform landmarks to aligned space
        ones = np.ones((shape.shape[0], 1))
        shape_h = np.hstack([shape, ones])
        aligned_shape = (M @ shape_h.T).T.astype(int)

        # Canonical crop (based on aligned landmarks)
        x_min = np.min(aligned_shape[:, 0])
        y_min = np.min(aligned_shape[:, 1])
        x_max = np.max(aligned_shape[:, 0])
        y_max = np.max(aligned_shape[:, 1])

        pad = 20
        x1 = max(0, x_min - pad)
        y1 = max(0, y_min - pad)
        x2 = min(aligned.shape[1], x_max + pad)
        y2 = min(aligned.shape[0], y_max + pad)

        face_cutout = aligned[y1:y2, x1:x2]

        # ---------- DRAW ALL LANDMARKS (CUTOUT SPACE) ----------
        for (x, y) in aligned_shape:
            cx = x - x1
            cy = y - y1
            if 0 <= cx < face_cutout.shape[1] and 0 <= cy < face_cutout.shape[0]:
                cv2.circle(face_cutout, (int(cx), int(cy)), 2, (255, 255, 255), -1)


        # ---------- CUTOUT NORMALIZATION ----------
        cut_gray = cv2.cvtColor(face_cutout, cv2.COLOR_BGR2GRAY)
        cut_gray = cv2.createCLAHE(2.0, (8, 8)).apply(cut_gray)

        # Gamma stabilization (robust to shadows)
        mean_intensity = np.mean(cut_gray) / 255.0
        gamma = 1.0 if mean_intensity == 0 else np.clip(0.5 / mean_intensity, 0.7, 1.5)
        lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype("uint8")
        cut_gray = cv2.LUT(cut_gray, lut)

        # Resize canonical face
        cut_gray = cv2.resize(cut_gray, (160, 160))
        cut_gray = cv2.flip(cut_gray, 0)  # ✅ FIX: flip cutout only


        # ======================================================
        # ================= EAR / MAR LOGIC ====================
        # ======================================================

        ear_vals = []
        le, re = shape[self.lStart:self.lEnd], shape[self.rStart:self.rEnd]
        if np.linalg.norm(le[0] - le[3]) > 1:
            ear_vals.append(eye_aspect_ratio(le))
        if np.linalg.norm(re[0] - re[3]) > 1:
            ear_vals.append(eye_aspect_ratio(re))

        if not ear_vals:
            self.render(gray_full)
            return

        ear = min(ear_vals)
        self.ear_history.append(ear)
        ear = sum(self.ear_history) / len(self.ear_history)

        self.perclos.append(1 if ear < self.EYE_THRESH else 0)

        if ear < self.EYE_THRESH:
            if self.eyes_closed_start is None:
                self.eyes_closed_start = now
        else:
            self.eyes_closed_start = None

        mar = mouth_aspect_ratio(shape[self.mStart:self.mEnd])
        self.mar_history.append(mar)
        mar = sum(self.mar_history) / len(self.mar_history)

        if mar > self.MAR_THRESH:
            if self.yawn_start is None:
                self.yawn_start = now
        else:
            self.yawn_start = None

        perclos_rate = sum(self.perclos) / len(self.perclos)
        yawning = self.yawn_start and (now - self.yawn_start) > self.YAWN_TIME
        is_active_yawn = bool(yawning)

        if yawning:
            if not self.yawn_events or (now - self.yawn_events[-1]) > 2.0:
                self.yawn_events.append(now)

        # Keep last 60 seconds
        self.yawn_events = deque(
            [t for t in self.yawn_events if now - t < 60],
            maxlen=10
        )


        nose, chin = shape[33], shape[8]
        head_angle = np.degrees(np.arctan2(chin[1] - nose[1], chin[0] - nose[0]))
        unconscious = (
            ear < self.UNCONSCIOUS_EAR and
            abs(head_angle) > self.HEAD_TILT_THRESH and
            self.eyes_closed_start and
            (now - self.eyes_closed_start) > 0.8
        )

        stable_eye_closure = (
            self.eyes_closed_start and
            (now - self.eyes_closed_start) > self.SLEEP_TIME and
            ear < self.UNCONSCIOUS_EAR
        )

        sleep_condition = (
            not is_active_yawn and
            (
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

        # ---------- VISUALIZATION ----------
        x1f, y1f, x2f, y2f = face.left(), face.top(), face.right(), face.bottom()
        cv2.rectangle(gray_full, (x1f, y1f), (x2f, y2f), 255, 2)

        cv2.putText(gray_full, f"EAR: {ear:.2f}", (30, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)
        cv2.putText(gray_full, f"PERCLOS: {perclos_rate:.2f}", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)

        if self.state == "DROWSY":
            cv2.putText(gray_full, "DROWSY", (30, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, 255, 4)
        elif self.state == "SLEEPING":
            cv2.putText(gray_full, "SLEEPING", (30, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.8, 255, 5)

        # ---------- OVERLAY NORMALIZED CUTOUT ----------
        h, w = gray_full.shape
        ch, cw = cut_gray.shape
        gray_full[h - ch - 10:h - 10, w - cw - 10:w - 10] = cut_gray

        self.render(gray_full)


    # ================= YOLO (VISUAL ONLY + TEMPORAL SLEEP) =================
    def process_yolo(self, frame, face_lost=False):
        now = time.time()

        res = self.yolo(frame, verbose=False)[0]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.createCLAHE(2.0, (8, 8)).apply(gray)

        if res.keypoints is None or res.boxes is None:
            self.yolo_tilt_start = None
            self.render(gray)
            return

        boxes = res.boxes.xyxy.cpu().numpy()
        areas = (boxes[:,2] - boxes[:,0]) * (boxes[:,3] - boxes[:,1])
        idx = np.argmax(areas)
        kp = res.keypoints.xy[idx].cpu().numpy()

        nose = kp[0]
        le, re = kp[1], kp[2]
        lear, rear = kp[3], kp[4]
        ls, rs = kp[5], kp[6]

        shoulder_mid = (ls + rs) / 2
        shoulder_vec = rs - ls
        shoulder_angle = np.degrees(np.arctan2(shoulder_vec[1], shoulder_vec[0]))
        shoulder_width = np.linalg.norm(shoulder_vec)

        eye_vec = re - le
        eye_angle = np.degrees(np.arctan2(eye_vec[1], eye_vec[0]))
        roll_angle = (eye_angle - shoulder_angle + 180) % 360 - 180

        eye_y = (le[1] + re[1]) / 2
        ear_y = (lear[1] + rear[1]) / 2
        face_plane_y = (eye_y + ear_y) / 2
        pitch_offset = (nose[1] - face_plane_y) / shoulder_width

        nose_vert = (nose[1] - shoulder_mid[1]) / shoulder_width

        state = "UPRIGHT"
        if roll_angle > 12:
            state = "RIGHT"
        elif roll_angle < -12:
            state = "LEFT"
        elif pitch_offset > 0.10:
            state = "FORWARD"
        elif pitch_offset < -0.05:
            state = "BACK"
        elif abs(roll_angle) < 6 and abs(pitch_offset) < 0.05 and abs(nose_vert) < 0.15:
            state = "UPRIGHT"

        self.face_obstructed = (
            face_lost and
            state == "UPRIGHT"
        )

        # -------- TEMPORAL SLEEP (YOLO) --------
        if state != "UPRIGHT":
            if self.yolo_tilt_start is None:
                self.yolo_tilt_start = now
        else:
            self.yolo_tilt_start = None

        yolo_sleeping = (
            self.yolo_tilt_start is not None and
            (now - self.yolo_tilt_start) >= self.YOLO_SLEEP_TIME
        )

        for i in UPPER_BODY_KP:
            x, y = kp[i]
            cv2.circle(gray, (int(x), int(y)), 4, 255, -1)

        cv2.putText(gray, f"HEAD: {state}",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    255,
                    3)

        if yolo_sleeping:
            cv2.putText(gray, "SLEEPING",
                        (30, 140),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.8,
                        255,
                        5)
            
        if self.face_obstructed:
            cv2.putText(
            gray,
            "FACE OBSTRUCTED",
            (30, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.4,
            255,
            4
        )

        self.render(gray)

    # ================= RENDER =================
    def render(self, gray):
        gray = cv2.flip(gray, 0)
        tex = Texture.create(
            size=(gray.shape[1], gray.shape[0]),
            colorfmt="luminance"
        )
        tex.blit_buffer(
            gray.tobytes(),
            colorfmt="luminance",
            bufferfmt="ubyte"
        )
        self.image.texture = tex



if __name__ == "__main__":
    DriverMonitoringApp().run()
