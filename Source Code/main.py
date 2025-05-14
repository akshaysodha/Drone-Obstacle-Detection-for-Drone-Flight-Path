from flask import Flask, request, jsonify
import cv2
import numpy as np
from google.cloud import storage
import os
import tempfile
import logging

# Initialize Flask app
app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define class mapping for segmentation
CLASS_MAPPING = {
    'ultra_thin': 255,
    'thin_structures': 240,
    'small_mesh': 220,
    'large_mesh': 200,
    'trees': 180,
    'buildings': 160,
    'vehicles': 140,
    'animals': 100,
    'other': 80
}

def process_image(input_path, segmentation_path):
    """Processes an image and segmentation file to generate YOLO annotations."""
    storage_client = storage.Client()

    # Parse the GCS bucket and file paths
    bucket_name, input_blob_name = input_path.replace("gs://", "").split("/", 1)
    input_bucket = storage_client.bucket(bucket_name)
    input_blob = input_bucket.blob(input_blob_name)

    segmentation_bucket_name, segmentation_blob_name = segmentation_path.replace("gs://", "").split("/", 1)
    segmentation_bucket = storage_client.bucket(segmentation_bucket_name)
    segmentation_blob = segmentation_bucket.blob(segmentation_blob_name)

    # Check if files exist
    if not input_blob.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if not segmentation_blob.exists():
        raise FileNotFoundError(f"Segmentation file not found: {segmentation_path}")

    # Download images from GCS to temporary local files
    with tempfile.NamedTemporaryFile(delete=False) as temp_input:
        input_blob.download_to_filename(temp_input.name)
        base_img = cv2.imread(temp_input.name)

    with tempfile.NamedTemporaryFile(delete=False) as temp_segmentation:
        segmentation_blob.download_to_filename(temp_segmentation.name)
        segmentation_img = cv2.imread(temp_segmentation.name, cv2.IMREAD_GRAYSCALE)

    # Generate YOLO annotations
    annotations = []
    for label, value in CLASS_MAPPING.items():
        mask = (segmentation_img == value).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > 50:  # Ignore small contours
                x, y, w, h = cv2.boundingRect(contour)
                x_center = (x + w / 2) / base_img.shape[1]
                y_center = (y + h / 2) / base_img.shape[0]
                width = w / base_img.shape[1]
                height = h / base_img.shape[0]
                class_id = list(CLASS_MAPPING.keys()).index(label)
                annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    return annotations

@app.route("/", methods=["POST"])
def annotate_images():
    """Endpoint to process images and generate annotations."""
    input_paths = [
        ("gs://newbucket010/test/neighbourhood/image.png", "gs://newbucket010/test/neighbourhood/segmentation.png"),
        ("gs://newbucket010/test/park/image.png", "gs://newbucket010/test/park/segmentation.png")
    ]
    output_bucket_name = "newbucket010"
    output_prefix = "test/annotations"

    storage_client = storage.Client()
    output_bucket = storage_client.bucket(output_bucket_name)

    for input_path, segmentation_path in input_paths:
        try:
            annotations = process_image(input_path, segmentation_path)
            output_blob_name = f"{output_prefix}/{os.path.basename(input_path).replace('.png', '.txt').replace('.jpg', '.txt')}"
            output_blob = output_bucket.blob(output_blob_name)
            output_blob.upload_from_string("\n".join(annotations))
        except Exception as e:
            logging.error(f"Error processing {input_path} and {segmentation_path}: {e}")

    return jsonify({"message": "Annotation pipeline completed."}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
