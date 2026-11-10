# Video Object Tracker (Qt Quick app)

Desktop app that tracks a user-selected object in a live webcam feed or a video file, using the **C-BIoU** and **McByte** trackers from this repository. The UI is Qt Quick (QML) on PySide6 and opens as a Python project in Qt Creator.

## Features

- **Video feed pane**: pick a webcam (live feed) or a media file, then Start, Pause/Resume or Close.
- **Click to track**: a left click draws a 30 × 30 px gate centred on the click; that gate is the target followed through the following frames. While hovering, a yellow outline previews the gate.
- **Adjustable gate**: change the gate width and height at any time; an active target is re-initialised at the new size.
- **Switch target**: click another object while tracking and the tracker restarts on the new gate.
- **Stop Tracking** breaks the current track.
- **Save** writes the processed (annotated) video to `.mp4` or `.avi` from that moment until *Stop Saving*, the source is closed, or the file ends.
- **Tracker choice**: C-BIoU or McByte. McByte can also use its SAM + Cutie mask pipeline when the `trackers[mask]` extra is installed.
- **Candidate source choice**: *Template matching* (default; any object, CPU only) or *YOLO detector* (pretrained YOLOv8 / RF-DETR via `inference-models`; only the detector's classes, such as people and vehicles). Switch at any time in the Tracking panel; an active target is re-selected at its current position.

## How it works

C-BIoU and McByte are tracking-by-detection trackers. They associate per-frame detections over time but do not find objects themselves. Because the target is an arbitrary clicked object, the app generates per-frame detections from the gate:

1. On click, the pixels inside the gate become an appearance template, and the tracker is reset and seeded with the gate as a confidence-1.0 detection. The track ID it returns is the target.
2. On each new frame, `TemplateCandidateGenerator` searches a window around the target's Kalman-predicted position with normalized cross-correlation. It returns the strongest few non-overlapping peaks as gate-sized boxes whose confidence is the correlation score.
3. The selected tracker associates those candidates with its tracks: high-scoring peaks in the first pass, weaker ones in the second (buffered IoU for C-BIoU, clear-match locking plus optional masks and camera-motion compensation for McByte). The box matched to the target ID is the tracked position.
With **YOLO detector** selected, steps 1 and 2 change: the detector runs on every frame, a click selects the detection that contains the click (the smallest one, so a person in front of a bus wins) or else the one overlapping the gate the most, and later frames only pass detections of that class to the tracker. The tracked box is the detector's box, so the gate size only sets the click area. If nothing is detected under the click, that target is tracked with template matching instead and a message says so.

4. If the target is not matched, it is shown as **LOST** and the search window grows. After a few lost frames, a new confidently matched track (score ≥ 0.8) is adopted as the target. The template slowly adapts to the target's appearance while it is tracked with a high score.

```
apps/qt_tracker/
├── main.py                  # entry point
├── qml/                     # UI: Main, SourcePane, TrackingPane, SavePane, StatusPane, VideoView, MessageBar
├── tracker_app/
│   ├── candidates.py        # template-matching candidate generator (no Qt)
│   ├── detector.py          # YOLO / RF-DETR detector candidate source (no Qt)
│   ├── engine.py            # single-target engine over C-BIoU / McByte (no Qt)
│   ├── worker.py            # capture → track → annotate → save loop in a QThread
│   ├── controller.py        # QObject exposed to QML
│   └── video_item.py        # QQuickPaintedItem that shows frames and maps clicks to image pixels
└── tests/                   # engine tests on synthetic video (no Qt needed)
```

## Run

From the repository root:

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r apps/qt_tracker/requirements.txt
python apps/qt_tracker/main.py
```

Optional extras, also from the repository root:

- YOLO detector candidates: `pip install -e ".[detection]"`. The model weights (default `yolov8n-640`; `yolov8s-640`, `yolov8m-640` and `rfdetr-nano` are offered, and any `inference-models` ID can be typed) download on first use.
- McByte's mask pipeline (downloads SAM and Cutie weights, a GPU is recommended): `pip install -e ".[mask]"`.

### Qt Creator

*File → Open File or Project…* and select `apps/qt_tracker/pyproject.toml` (Qt Creator 14+) or `apps/qt_tracker/qt_tracker.pyproject` (older versions). Choose the virtual environment above as the Python interpreter and press Run.

## Tests

```bash
pip install pytest
pytest apps/qt_tracker/tests
```
