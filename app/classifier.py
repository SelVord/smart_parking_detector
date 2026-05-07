"""CNN model loading and occupancy inference."""

import os

import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

_BASE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CNN_PATH  = os.path.join(_BASE, "models", "cnn_classifier.pth")
_DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"

_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_cnn():
    """Return (model, device, error_string).  error_string is None on success."""
    if not os.path.exists(_CNN_PATH):
        return None, None, f"CNN weights not found: {_CNN_PATH}"
    try:
        device = torch.device(_DEVICE)
        cnn = models.mobilenet_v2(weights=None)
        cnn.classifier[1] = torch.nn.Linear(1280, 2)
        cnn.load_state_dict(torch.load(_CNN_PATH, map_location=device, weights_only=True))
        cnn.eval()
        cnn = cnn.to(device)
        return cnn, device, None
    except Exception as exc:
        return None, None, str(exc)


def extract_zone(image_rgb: np.ndarray, points) -> np.ndarray:
    """Warp a quadrilateral region into a 200x300 rectangle."""
    pts = np.array(points, dtype=np.float32)
    dst = np.array([[0, 0], [200, 0], [200, 300], [0, 300]], dtype=np.float32)
    M   = cv2.getPerspectiveTransform(pts, dst)
    return cv2.warpPerspective(image_rgb, M, (200, 300))


def is_occupied(image_rgb: np.ndarray, points, model, device, threshold: float = 45):
    """Return (occupied: bool, confidence: float)."""
    patch = extract_zone(image_rgb, points)

    if model is not None:
        try:
            patch_pil = Image.fromarray(patch).convert("RGB")
            tensor    = _transform(patch_pil).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(tensor)
                prob   = torch.softmax(output, dim=1)
                pred   = output.argmax(1).item()
                conf   = prob[0][pred].item()
            return pred == 1, conf
        except Exception:
            pass

    # HSV fallback when model unavailable
    hsv   = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
    score = float(np.std(hsv[:, :, 2]) * 0.7 + np.mean(hsv[:, :, 1]) * 0.3)
    return score > threshold, score
