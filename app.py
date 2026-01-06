import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import cv2
import numpy as np

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.popup import Popup
from kivy.uix.filechooser import FileChooserIconView
from kivy.clock import Clock
from kivy.graphics.texture import Texture

# ================= MODULES =================
from modules.face_tracker import FaceTracker
from modules.body_tracker import BodyTracker
from modules.smoking_detector import SmokingDetector
from modules.phone_detector import PhoneDetector

# ================= PATHS =================
predictor_path = r"models\shape_predictor_68_face_landmarks.dat"
video_folder = r"videos"
YOLO_MODEL = "yolov8n-pose.pt"


def get_base(output):
    if output.get("base") is not None:
        return output.get("base")
    return output.get("frame")


class DriverMonitoringApp(App):

    def build(self):
        # -------- STATE --------
        self.debug_enabled = False
        self.capture = None
        self.video_path = None
        self.use_webcam = False

        self.last_face_out = None
        self.last_body_out = None
        self.last_smoke_out = None

        # -------- MODULES --------
        self.face_tracker = FaceTracker(predictor_path)
        self.body_tracker = BodyTracker(YOLO_MODEL)
        self.smoking_detector = SmokingDetector()
        self.phone_detector = PhoneDetector()

        # -------- UI --------
        layout = BoxLayout(orientation="vertical")
        self.image = Image()
        layout.add_widget(self.image)

        controls = BoxLayout(size_hint_y=None, height=50)
        controls.add_widget(Button(text="Load Video", on_press=self.open_file_chooser))
        controls.add_widget(Button(text="Use Webcam", on_press=self.start_webcam))
        controls.add_widget(Button(text="Play", on_press=self.start))
        controls.add_widget(Button(text="Stop", on_press=self.stop))
        controls.add_widget(Button(text="Debug", on_press=self.toggle_debug))
        layout.add_widget(controls)

        return layout

    # ================= INPUT =================
    def toggle_debug(self, _):
        self.debug_enabled = not self.debug_enabled

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

        self.capture = None
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
        if not self.capture:
            return

        ret, frame = self.capture.read()
        if not ret:
            return

        now = time.time()

        # ---------- FACE TRACKER ----------
        face_out = self.face_tracker.update(frame, now)

        # ---------- BODY TRACKER ----------
        body_out = self.body_tracker.update(
            frame,
            now,
            face_lost=face_out.get("face_lost", False)
        )

        self.last_face_out = face_out
        self.last_body_out = body_out

        # ---------- SMOKING ----------
        smoke_out = self.smoking_detector.update(frame)
        self.last_smoke_out = smoke_out

        # ---------- PHONE ----------
        self.phone_detector.update(frame)

        # ---------- BASE FRAME ----------
        base = (
            get_base(face_out)
            if face_out.get("face_found")
            else get_base(body_out)
        )

        if base is None:
            return

        final = base.copy()

        # ---------- OVERLAYS ----------
        if face_out.get("overlay") is not None:
            final = cv2.add(final, face_out["overlay"])

        if body_out.get("overlay") is not None:
            final = cv2.add(final, body_out["overlay"])

        if smoke_out.get("smoking"):
            cv2.putText(
                final,
                f"SMOKING ({smoke_out['confidence']:.2f})",
                (30, 180),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.4,
                255,
                4
            )

        # ---------- ALERT TEXT ----------
        if smoke_out.get("smoking"):
            cv2.putText(
                final,
                f"SMOKING ({smoke_out['confidence']:.2f})",
                (30, 180),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.4,
                255,
                4
            )

        if face_out.get("face_found"):
            if face_out.get("state") == "DROWSY":
                cv2.putText(final, "DROWSY", (30, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.4, 255, 4)
            elif face_out.get("state") == "SLEEPING":
                cv2.putText(final, "SLEEPING", (30, 140),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.8, 255, 5)
        else:
            if body_out.get("sleeping"):
                cv2.putText(final, "SLEEPING", (30, 140),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.8, 255, 5)

            if body_out.get("face_obstructed"):
                cv2.putText(final, "FACE OBSTRUCTED", (30, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.4, 255, 4)

        # ---------- DEBUG ----------
        if self.debug_enabled:
            self.draw_debug(final)

        self.render(final)

    # ================= DEBUG DRAW =================
    def draw_debug(self, frame):
        fo = self.last_face_out
        if not fo or not fo.get("face_found"):
            return

        y = 30

        cv2.putText(frame, f"EAR: {fo['ear']:.3f}",
                    (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)
        y += 22

        cv2.putText(frame, f"MAR: {fo['mar']:.3f}",
                    (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)
        y += 22

        cv2.putText(frame, f"PERCLOS: {fo['perclos']:.2f}",
                    (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)
        y += 22

        cv2.putText(frame, f"HEAD ANGLE: {fo['head_angle']:.1f}",
                    (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)

        if self.last_smoke_out:
            cv2.putText(
                frame,
                f"SMOKING: {self.last_smoke_out['smoking']} "
                f"({self.last_smoke_out['confidence']:.2f})",
                (30, y + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                255,
                2
            )

    # ================= RENDER =================
    def render(self, frame):
        frame = cv2.flip(frame, 0)

        if len(frame.shape) == 2:
            colorfmt = "luminance"
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            colorfmt = "rgb"

        tex = Texture.create(
            size=(frame.shape[1], frame.shape[0]),
            colorfmt=colorfmt
        )

        tex.blit_buffer(
            frame.tobytes(),
            colorfmt=colorfmt,
            bufferfmt="ubyte"
        )

        self.image.texture = tex


if __name__ == "__main__":
    DriverMonitoringApp().run()
