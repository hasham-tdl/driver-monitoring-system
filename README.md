=====================================================================
                         DRIVER MONITORING SYSTEM
=====================================================================

Real-time driver behavior monitoring using Computer Vision & Deep Learning

Built with:
Python | OpenCV | Kivy | YOLOv8

---------------------------------------------------------------------
OVERVIEW
---------------------------------------------------------------------

This Driver Monitoring System (DMS) is a real-time application designed
to analyze driver behavior and detect unsafe conditions such as:

- Drowsiness
- Sleeping
- Face obstruction
- Smoking
- Phone usage

The system supports both webcam and video input and displays results
through a Kivy-based GUI with live overlays and debug metrics.

---------------------------------------------------------------------
KEY FEATURES
---------------------------------------------------------------------

[ FACE-BASED MONITORING ]
- Face detection and tracking
- Eye Aspect Ratio (EAR)
- Mouth Aspect Ratio (MAR)
- PERCLOS (percentage of eye closure)
- Head pose / head angle estimation
- Drowsiness and sleeping detection

[ BODY & POSE FALLBACK ]
- YOLOv8 pose detection when face is lost
- Sleeping detection via head tilt
- Face obstruction detection

[ SMOKING DETECTION ]
- YOLO-based smoking detection
- Grayscale preprocessing for robustness
- Confidence score output

[ PHONE USAGE DETECTION ]
- YOLO-based phone usage detection

[ USER INTERFACE ]
- Kivy graphical interface
- Webcam and video file support
- Real-time overlays
- Debug panel with live metrics

---------------------------------------------------------------------
PROJECT STRUCTURE
---------------------------------------------------------------------

app.py
|
+-- modules/
|    |-- face_tracker.py
|    |-- body_tracker.py
|    |-- smoking_detector.py
|    |-- phone_detector.py
|
+-- models/        (local only, not committed)
+-- videos/
+-- README.md

Each module:
- Is independent
- Handles its own preprocessing
- Returns structured output dictionaries
- Can be modified without breaking the pipeline

---------------------------------------------------------------------
REQUIREMENTS
---------------------------------------------------------------------

- Python 3.8+
- OpenCV
- Kivy
- NumPy
- dlib
- imutils
- SciPy
- ultralytics (YOLOv8)

---------------------------------------------------------------------
INSTALLATION
---------------------------------------------------------------------

1) Clone the repository

   git clone https://github.com/hasham-tdl/driver-monitoring-system.git
   cd driver-monitoring-system

2) Create a virtual environment (recommended)

   conda create -n dms python=3.9
   conda activate dms

3) Install dependencies

   pip install opencv-python kivy numpy imutils scipy ultralytics

NOTE:
dlib may require CMake and Visual Studio Build Tools on Windows.

---------------------------------------------------------------------
MODEL FILES (NOT INCLUDED)
---------------------------------------------------------------------

Model weights are NOT included in this repository.

Place the following files locally:

models/
- shape_predictor_68_face_landmarks.dat
- yolov8n-pose.pt
- smoking_model.pt
- phone_model.pt

Update model paths inside the modules if required.

---------------------------------------------------------------------
RUNNING THE APPLICATION
---------------------------------------------------------------------

python app.py

Controls:
- Load Video   -> Select an MP4 file
- Use Webcam   -> Start live camera feed
- Play / Stop
- Debug        -> Toggle live metrics overlay

---------------------------------------------------------------------
DEBUG METRICS
---------------------------------------------------------------------

When debug mode is enabled, the following metrics are displayed:

- EAR
- MAR
- PERCLOS
- Head Angle
- Smoking status and confidence

Useful for:
- Threshold tuning
- Model validation
- Research and experimentation

---------------------------------------------------------------------
LIMITATIONS
---------------------------------------------------------------------

- Performance depends on lighting conditions
- Smoking detection accuracy depends on training data
- CPU-only inference may reduce FPS
- Not certified for commercial automotive deployment

---------------------------------------------------------------------
FUTURE IMPROVEMENTS
---------------------------------------------------------------------

- Temporal smoothing for smoking detection
- Mouth ROI cropping before inference
- Multi-driver support
- Audio or visual alerts
- Cloud logging and analytics
- Embedded / edge deployment

---------------------------------------------------------------------
DISCLAIMER
---------------------------------------------------------------------

This project is intended for research and educational purposes only.
It is NOT certified for safety-critical or commercial automotive use.

---------------------------------------------------------------------
AUTHOR
---------------------------------------------------------------------

Hasham
GitHub: https://github.com/hasham-tdl

---------------------------------------------------------------------
ACKNOWLEDGEMENTS
---------------------------------------------------------------------

- OpenCV
- dlib
- Kivy
- Ultralytics YOLO
- Academic research on Driver Monitoring Systems

---------------------------------------------------------------------
CONTRIBUTIONS
---------------------------------------------------------------------

Contributions are welcome.
For major changes, please open an issue first.

=====================================================================
