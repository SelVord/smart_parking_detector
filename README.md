# Parking Detection

A Streamlit web app for real-time parking space occupancy detection. Mark parking spaces manually on photos or videos, and a fine-tuned MobileNetV2 CNN classifies each space as **free** or **occupied** — with confidence-based coloring to flag uncertain predictions.

---

## Features

### Zone Marking
- **Rectangle mode** — drag to draw a bounding box over each parking space; instant, no interruptions between zones
- **4-point mode** — click 4 corners for each space, enabling accurate perspective-corrected analysis on angled or wide-angle shots
- **Save / Load zones** — export zone layouts to JSON and reload them for the same camera angle later (no re-marking needed)
- **Remove last** button for quick corrections in both modes
- **Clear all** to start over

### Photo Analysis
- Upload any JPEG/PNG parking lot photo
- Mark spaces with either tool
- Click **Analyze** — every space is classified instantly
- Overlay shows: green (free), red (occupied), yellow (uncertain)
- Occupancy metrics and progress bar shown below the image

### Video Analysis
- Upload MP4/AVI/MOV parking lot footage
- Mark zones once on the first frame — zones apply to all frames
- **Frame scrubber** with navigation buttons (−100 / −10 / −1 / +1 / +10 / +100) for frame-by-frame inspection
- Every seeked frame is analyzed automatically with zone overlay
- **Batch processing** — process the entire video in a background thread (non-blocking UI)
  - Live preview of the latest analyzed frame
  - Real-time occupancy counter
  - Event log of status changes with timestamps (e.g. `Space #3 → occupied at 01:24`)
  - Configurable check interval (20–500 ms) and stability filter (1–10 frames)
  - **Cancel** button to stop mid-run

### Classifier
- **MobileNetV2** fine-tuned on the PKLot dataset (497K+ annotated parking spaces)
- **Perspective warp** — each zone is warped to a 200×300 canonical view before classification, correcting for camera angle
- **Confidence threshold** slider (0.50–0.99) — predictions below the threshold are shown in yellow instead of green/red
- **HSV fallback** — if the CNN weights are missing, falls back to an HSV color/texture heuristic so the app always works

---

## Tech Stack

| Component | Library |
|---|---|
| UI / web app | Streamlit 1.38.0 |
| Zone drawing | streamlit-drawable-canvas |
| CNN classifier | PyTorch + MobileNetV2 |
| Image processing | OpenCV, Pillow |
| Background processing | Python threading |

---

## Project Structure

```
parking/
├── app/
│   ├── app.py               # Streamlit entry point
│   ├── classifier.py        # CNN loading, perspective warp, inference
│   ├── drawing.py           # Zone drawing widget (rect + 4-point modes)
│   ├── ui.py                # CSS, header, metrics, zone overlay renderer
│   └── video_processor.py   # Background video processing (threading)
├── models/
│   └── cnn_classifier.pth   # Trained weights (download separately — see Setup)
└── requirements.txt
```

---

## Setup

### 1. Clone and create a virtual environment

```bash
git clone https://github.com/your-username/parking-detection.git
cd parking-detection
python -m venv venv
```

Activate:
- Windows: `venv\Scripts\activate`
- Linux/Mac: `source venv/bin/activate`

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

For GPU support (CUDA 12.8):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

### 3. Download the pre-trained model

A pre-trained MobileNetV2 checkpoint (fine-tuned on PKLot) is provided so you can run the app immediately — no dataset download or training required.

Download `cnn_classifier.pth` from the [Releases page](https://github.com/your-username/parking-detection/releases) and place it in the `models/` directory:

```
parking/
└── models/
    └── cnn_classifier.pth   ← place it here
```

> **No model?** The app will automatically fall back to an HSV color/texture heuristic so it always runs, even without the weights file.

### 4. Run the app

```bash
streamlit run app/app.py
```

Open `http://localhost:8501` in your browser.

---

## How It Works

1. **Zone marking** — the user draws rectangles or places 4 corner points on the parking image
2. **Perspective warp** — each zone's quadrilateral is warped to a standard 200×300 view using `cv2.getPerspectiveTransform`, correcting for any camera angle
3. **CNN inference** — the warped patch is resized to 64×64, normalized with ImageNet stats, and passed through MobileNetV2; softmax gives a probability for each class
4. **Confidence filtering** — if the winning class probability is below the configured threshold, the space is shown in yellow (uncertain) instead of green/red
5. **Video stability** — for video processing, a configurable number of consecutive frames must agree before a status change is recorded, filtering out single-frame noise

---

## Color Legend

| Color | Meaning |
|---|---|
| Green | Space is free |
| Red | Space is occupied |
| Yellow | Prediction confidence is below threshold (uncertain) |
| Cyan | Space not yet analyzed |

---

## Training from Scratch (Optional)

If you want to retrain the model yourself, the training script uses the [PKLot dataset](https://public.roboflow.com/object-detection/pklot) — a large benchmark with 497K+ annotated parking spaces from multiple parking lots and weather conditions.

<details>
<summary>Expand training instructions</summary>

### Download PKLot

```bash
pip install kagglehub
python - <<'EOF'
import kagglehub
kagglehub.dataset_download("ammarnassanalhajali/pklot-dataset")
EOF
```

Place the downloaded data at `data/pklot/` with the structure:
```
data/pklot/
├── train/
│   ├── _annotations.coco.json
│   └── *.jpg
├── valid/
│   ├── _annotations.coco.json
│   └── *.jpg
```

### Run training

```bash
python scripts/train_on_pklot.py
```

Training settings (editable at the top of the script):

| Setting | Default | Description |
|---|---|---|
| `EPOCHS` | 20 | Training epochs |
| `BATCH_SIZE` | 64 | Batch size |
| `MAX_PER_CLASS` | 30 000 | Max PKLot samples per class |
| `LR` | 0.001 | Learning rate (cosine decay) |
| `PATCH_SIZE` | 64 | Input image size |

The best checkpoint (by validation accuracy) is saved to `models/cnn_classifier.pth`.

### Custom data

Drop additional labeled patch images into `data/patches/empty/` and `data/patches/occupied/`. The training script automatically picks them up and adds them to the training set.

</details>

---

## Requirements

- Python 3.9+
- See `requirements.txt` for full dependency list
- GPU optional but recommended for video batch processing
