from ultralytics import YOLO
from PIL import Image
import cv2

# Load the model once when the container starts
model = YOLO("best.pt")          # HF resolves this path inside the repo

def predict(image, conf: float = 0.25, iou: float = 0.45):
    """
    Args:
        image: raw bytes or PIL.Image provided by the API
        conf : confidence threshold (default 0.25)
        iou  : IoU threshold for NMS (default 0.45)

    Returns:
        PIL.Image with bounding boxes drawn.
    """
    # Make sure we have a PIL.Image
    if not isinstance(image, Image.Image):
        image = Image.open(image)

    # Run inference
    results = model(image, conf=conf, iou=iou)[0]

    # Ultralytics returns a BGR NumPy array from .plot()
    annotated = results.plot()
    annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)   # BGR ➜ RGB

    return Image.fromarray(annotated)