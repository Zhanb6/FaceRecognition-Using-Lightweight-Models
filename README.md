# Energy-Efficient Face Recognition System

A lightweight real-time face recognition system built with MobileFaceNet/FaceNet and OpenCV, designed for energy-efficient deployment on CPU-based devices.

## Project Structure
```
Diploma/
├── main.py            # Entry point
├── camera.py          # Main camera loop, tracking, enrollment
├── embeddings.py      # Model loading, embedding extraction, recognition
├── database.py        # Face database load/save operations
├── config.py          # All configurable parameters
├── benchmark.py       # Performance & Energy efficiency testing tool
├── faces_database.pkl # Stored face embeddings (auto-generated)
└── README.md
```

## Benchmarking (Research)

To compare models for energy efficiency and performance:
```bash
python benchmark.py
```

**Current Baseline (FaceNet / VGGFace2):**
- **Parameters:** 27.91 M
- **Latency:** ~12.10 ms (Inference only)
- **CPU Usage:** ~90.9%
- **Memory Usage:** ~481.05 MB
- **Estimated Model FPS:** ~82.7

*Wait, real camera FPS is lower (~10-15) because it includes detection (MTCNN), tracking, and rendering.*

## Requirements

- Python 3.11
- macOS / Linux / Windows

## Installation

**1. Clone or download the project**
```bash
cd ~/Downloads/Diploma
```

**2. Create virtual environment with Python 3.11**
```bash
python3.11 -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows
```

**3. Install dependencies**
```bash
pip install torch==2.2.2 torchvision==0.17.2
pip install facenet-pytorch
pip install "numpy>=1.24.0,<2.0.0"
pip install "opencv-contrib-python==4.8.1.78"
pip install Pillow
```

## Usage

**Run the system**
```bash
python main.py
```

**Controls**

| Key | Action |
|-----|--------|
| `E` | Enroll a new face (type name in terminal) |
| `Q` | Quit |

**Enrolling a face**
1. Run `python main.py`
2. Make sure your face is visible in the camera
3. Press `E` in the camera window
4. Type your name in the terminal and press Enter
5. The box around your face will turn green

## Configuration

All parameters are in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `FRAME_SKIP` | `5` | Run face detection every N frames |
| `RECOG_INTERVAL` | `5` | Re-recognize every N frames |
| `THRESHOLD` | `0.7` | Minimum cosine similarity to confirm identity |
| `MIN_FACE_SIZE` | `80` | Minimum face size in pixels to detect |
| `MODEL_NAME` | `vggface2` | Pretrained weights to use |
| `DB_FILE` | `faces_database.pkl` | Path to face database file |

## How It Works
```
Camera frame
     │
     ▼
MTCNN (face detection) — runs every FRAME_SKIP frames
     │
     ▼
Face crop → resize 160×160
     │
     ▼
FaceNet (embedding extraction) — outputs 512-dimensional vector
     │
     ▼
Cosine similarity vs database — compare against all stored vectors
     │
     ▼
Identity + confidence score — displayed on screen
```

## Performance

Tested on MacBook Pro (Apple M-series), CPU only:

| Metric | Value |
|--------|-------|
| FPS | ~10–15 (CPU) |
| Detection model | MTCNN |
| Recognition model | InceptionResnetV1 (VGGFace2) |
| Embedding size | 512-d float32 |
| Database format | Python pickle |

## Known Limitations

- Face database is stored as a local `.pkl` file — not suitable for production
- No GPU acceleration (CPU only in current setup)
- Recognition accuracy decreases with poor lighting or extreme head angles
- `pickle` format has no encryption — database file should not be shared

## Future Work

- [ ] Migrate to Django web interface
- [ ] Replace pickle database with PostgreSQL
- [ ] Swap FaceNet for MobileFaceNet for better energy efficiency
- [ ] Add GPU support via CoreML (Apple Silicon) or CUDA
- [ ] REST API for remote recognition requests
- [ ] Logging and analytics dashboard

## Author

## Start project

```bash
source venv/bin/activate && python main.py
```

