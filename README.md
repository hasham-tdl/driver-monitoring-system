# Driver Monitoring System (v1)

**Detect drowsiness, phone use, and smoking in real time on a $35 Raspberry Pi — because driver inattention kills and dedicated ADAS hardware costs thousands.**

> **v2 is a from-scratch rebuild with significantly higher accuracy** (97.8% eye-state CNN, ground-truth regression suite). See [dms2 / v2](https://github.com/hasham-tdl/v2).

## What It Does

Real-time driver behavior monitoring on a **Raspberry Pi 4 + IR camera**,
with a desktop GUI for development. Detects:

- **Drowsiness** — PERCLOS, yawning, with a per-driver calibrated eye threshold
- **Sleeping** — sustained eye closure, head-slump fast path, body-pose fallback
- **Phone distraction** — phone visible **and** head down/away (stock COCO model, no training needed)
- **Smoking** — custom YOLO11 cigarette model (`models/best.pt`)
- **Looking away / face obstructed**

Built with Python, OpenCV, **MediaPipe Face Mesh** (replaces dlib — ~10× faster
on ARM, no 100 MB landmark file), and Ultralytics YOLO with NCNN exports for
the Pi.

## Architecture

```
                 ┌────────────────────── every frame ──────────────────────┐
camera ──► CLAHE ├─► FaceTracker (MediaPipe): EAR/MAR/PERCLOS, pitch/yaw    │
 (IR ok)         ├─► PoseFallback (YOLO pose) ── only while face is lost   │
                 └─► ObjectWatch ── scheduled, not per-frame:              │
                       phone  every 0.7 s   (COCO 'cell phone' class)      │
                       smoking every 2 s    (best.pt cigarette model)      │
                                   │
                                   ▼
                           Fusion state machine
                  AWAKE / DROWSY / SLEEPING + distraction flags
                  (time-based confirmation + hysteresis everywhere)
                                   │
                     ┌─────────────┼──────────────┐
                     ▼             ▼              ▼
               screen overlay   event log    buzzer (optional,
               (HDMI / browser) + snapshots   wired later)
```

All thresholds live in [config.yaml](config.yaml).

## Project layout

| Path | Purpose |
|---|---|
| `dms/` | Core package: capture, face tracking, fusion, detectors, alerts |
| `web/` | Live dashboard (3D head pose, gaze zones, metrics, event feed) |
| `app.py` | Desktop GUI (Kivy) — development & threshold tuning |
| `pi_service.py` | Headless service + web dashboard (`:8080`) |
| `scripts/eval_video.py` | Run the pipeline over a video, print the state timeline |
| `scripts/export_ncnn.py` | One-time NCNN export of all models for Pi speed |
| `deploy/dms.service` | systemd unit to start on boot |

## Dashboard

`pi_service.py` serves a dashboard at `http://<host>:8080/` (works in any
browser — phone, laptop, or an HDMI screen on the Pi):

- **Live 3D face mesh** — the driver's actual 478 MediaPipe landmarks rendered
  as a point-cloud with eye/lip/oval contours, so blinks, yawns, and head
  turns are real data, not an animation (ghost mannequin when tracking drops)
- **Gaze zone map** — a windshield grid showing where the driver is looking
  (ROAD AHEAD / mirrors / PHONE-LAP), green on road, red off road
- Live camera view, EAR/PERCLOS/MAR gauges with personal thresholds,
  head angles, phone/cigarette detection chips, scrolling event log
- Three.js is vendored in `web/` — no internet needed in the car

Bench demo without a camera:

```bash
python pi_service.py --video "videos/Driver sleeps for a full minute while driving.mp4"
```

## Models (`models/`)

| File | Role | Source |
|---|---|---|
| `face_landmarker.task` | face/eye/head tracking | [MediaPipe models](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task) (3.6 MB) |
| `best.pt` | smoking (1 class: cigarette) | your Colab-trained YOLO11m |
| `yolo11n.pt` | phone (COCO class 67) | [Ultralytics releases](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt) (5.4 MB) |
| `yolov8n-pose.pt` | body fallback | Ultralytics |

The runtime automatically prefers a `*_ncnn_model` folder next to each `.pt`
when present (created by `scripts/export_ncnn.py`).

## Desktop setup (Windows)

```bash
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements-desktop.txt
python app.py                       # GUI
python scripts/eval_video.py "videos/clip.mp4"   # headless tuning
```

## Raspberry Pi setup (Pi 4, Raspberry Pi OS Bookworm 64-bit Lite)

```bash
sudo apt update
sudo apt install -y python3-picamera2 python3-opencv python3-pip
pip install -r requirements-pi.txt --break-system-packages

# copy models/ (with *_ncnn_model exports) onto the Pi, then:
python3 pi_service.py               # open http://<pi>:8080/ on your phone
```

Start on boot: see [deploy/dms.service](deploy/dms.service).

**Camera:** OV5647 NoIR "night vision" board (CSI ribbon). Focus the lens at
driver distance (~50–70 cm) by rotating the barrel. The IR LED pods handle
night illumination; the whole pipeline runs on CLAHE-enhanced grayscale, so
day and night frames look alike to the models.

**Power (demo):** any 15 W+ USB-C power bank runs the Pi 4 + camera for hours.

## Test footage (`videos/samples/`)

Freely accessible clips pulled from public GitHub repos for detector testing
(not redistributed here — see `.gitignore`):

| File | Tests | Source |
|---|---|---|
| `smoking_test_1..4.mp4` | smoking detector | [Realtime-Smoking-Detection](https://github.com/Paransaik/Realtime-Smoking-Detection), [Smoking-Detection](https://github.com/AarnoStormborn/Smoking-Detection) |
| `distracted_driving_demo.mp4` | drowsiness, long-run stability | [Distracted-Driver-Detection](https://github.com/HarshineeSriram/Distracted-Driver-Detection) |
| `driver_action_recognition.mp4` | phone usage, distraction | [intel-iot-devkit/sample-videos](https://github.com/intel-iot-devkit/sample-videos) |
| `head_pose_test.mp4` | head angles / gaze zones | [intel-iot-devkit/sample-videos](https://github.com/intel-iot-devkit/sample-videos) |

Run any of them: `python scripts/eval_video.py videos/samples/<file>`
Quickest live phone-detector test: `python app.py` → *Use Webcam* → hold up
your phone and look down at it for ~3 s.

## How detection works

- **Calibration** — first ~25 s of each session learns the driver's open-eye
  EAR; threshold = 75th-percentile × 0.75 (clamped). Press *Recalibrate* (GUI)
  when the driver changes.
- **DROWSY** — PERCLOS over a rolling 60 s window crosses 0.20 (clears at
  0.15), or a sustained yawn.
- **SLEEPING** — eyes closed > 1.5 s, or closed > 0.8 s with the head slumped
  > 20°; body-pose tilt fallback when the face is hidden. If the face vanishes
  *while* SLEEPING (head slumps out of view), the state is held for up to 8 s
  rather than reset.
- **Phone** — phone seen in ≥3 of the last 5 detector runs **and** head
  down/away ≥ 1.5 s.
- **Smoking** — cigarette in ≥2 of the last 4 runs.
- Every alert appends a row to `runs/events.csv` plus a JPEG snapshot —
  use these to tune thresholds per test subject.

## Roadmap

- [ ] Retrain one 2-class YOLO11n (cigarette + phone) with grayscale/IR
      augmentation — replaces both detector models, biggest accuracy win
- [ ] GPIO buzzer + acknowledge button (`BuzzerSink` is already wired for it)
- [ ] Permanent install power: 12 V→5 V buck + read-only filesystem

## Disclaimer

Research/educational prototype. **Not** certified for safety-critical or
commercial automotive use.

## My Role

Solo end-to-end project. Designed the multi-signal fusion state machine, integrated MediaPipe Face Mesh (replacing dlib for ~10× ARM speedup), trained a custom YOLO11 cigarette model in Colab, built the Kivy desktop GUI for threshold tuning, and wrote the live browser dashboard served from the Pi.

## The Hard Technical Challenge

Running real-time multi-model inference on a Pi 4 (no GPU, 4 GB RAM). Solved by: replacing dlib with MediaPipe (10× faster on ARM, no 100 MB landmark file), scheduling YOLO detectors at sub-frame intervals (phone every 0.7 s, smoke every 2 s in a background thread) instead of per-frame, and exporting all models to NCNN format via `scripts/export_ncnn.py`.

## Constraint

Single Raspberry Pi 4 with IR camera, running headless with no internet. All Three.js dependencies are vendored locally so the dashboard works offline in the car.

## Author

Hasham — https://github.com/hasham-tdl
