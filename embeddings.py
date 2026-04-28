"""
embeddings.py — Model loading, embedding extraction, recognition.

Supports multiple backbones (selected via BACKBONE in config.py):
    - facenet           → InceptionResnetV1, 160×160, 512-d
    - mobilefacenet     → MobileFaceNet,     112×112, 128-d
    - efficientnet_lite0→ EfficientNet-Lite0, 112×112, 512-d
"""

import torch
import numpy as np
import logging
from PIL import Image
from facenet_pytorch import MTCNN
from config import THRESHOLD, MODEL_NAME, MIN_FACE_SIZE, BACKBONE

logger = logging.getLogger(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using: {device}")

# ─── Face detector (shared by all backbones) ─────────────────────────────
mtcnn = MTCNN(keep_all=True, device=device, min_face_size=MIN_FACE_SIZE)

# ─── Load recognition model based on BACKBONE ───────────────────────────
INPUT_SIZE = 160  # default for FaceNet

if BACKBONE == 'facenet':
    from models.facenet import FaceNet
    model = FaceNet(pretrained=MODEL_NAME).eval().to(device)
    INPUT_SIZE = 160
    print(f"Backbone: FaceNet (InceptionResnetV1, {MODEL_NAME})")

elif BACKBONE == 'mobilefacenet':
    from models.mobilefacenet import MobileFaceNet
    model = MobileFaceNet(embedding_size=128, input_size=112).eval().to(device)
    INPUT_SIZE = 112
    print("Backbone: MobileFaceNet (128-d, 112×112)")

elif BACKBONE == 'efficientnet_lite0':
    from models.efficientnet_lite import EfficientNetLite0Face
    model = EfficientNetLite0Face(embedding_size=512, pretrained=True).eval().to(device)
    INPUT_SIZE = 112
    print("Backbone: EfficientNet-Lite0 (512-d, 112×112)")

else:
    raise ValueError(
        f"Unknown BACKBONE='{BACKBONE}'. "
        f"Choose from: facenet, mobilefacenet, efficientnet_lite0"
    )


# ─── Helpers ─────────────────────────────────────────────────────────────

def normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def get_embedding_from_crop(face_crop_pil):
    """Extract embedding from a PIL face crop."""
    try:
        face        = face_crop_pil.resize((INPUT_SIZE, INPUT_SIZE))
        arr         = np.array(face).astype(np.float32)
        arr         = (arr / 255.0 - 0.5) / 0.5  # normalize to [-1, 1]
        face_tensor = torch.tensor(arr).permute(2, 0, 1)
        with torch.no_grad():
            emb = model(face_tensor.unsqueeze(0).to(device))
        return normalize(emb[0].cpu().numpy())
    except Exception:
        logger.exception("Embedding extraction failed")
        return None


def recognize(embedding, database):
    """Compare embedding against all stored vectors using cosine similarity."""
    best_name, best_score = "Unknown", 0.0
    query = normalize(np.array(embedding).flatten())
    for name, vectors in database.items():
        if name.startswith('__'):
            continue
        for vec in vectors:
            stored = normalize(np.array(vec).flatten())
            score  = float(np.dot(query, stored))
            if score > best_score:
                best_score, best_name = score, name
    if best_score > THRESHOLD:
        return best_name, best_score
    return "Unknown", best_score
