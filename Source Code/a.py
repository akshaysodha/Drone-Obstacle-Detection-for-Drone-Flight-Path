import os
import cv2
import numpy as np
import json
import logging
from tqdm import tqdm
import shutil

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Paths for the test set
source_base_path = r"E:\DATA 298 A project\Final\DDOS\data\test"  # Update to point to the test set
destination_base_path = r"E:\DATA 298 A project\Annotated Data\EfficientDet\test"  # Destination for EfficientDet

# Class Mapping for segmentation
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

# Create COCO-like dataset structure
coco_data = {
    "info": {
        "description": "EfficientDet Dataset",
        "version": "1.0",
        "year": 2024,
        "contributor": "Your Name",
        "date_created": "2024-11-29"
    },
    "licenses": [],
    "categories": [
        {"id": i, "name": label, "supercategory": "none"}
        for i, label in enumerate(CLASS_MAPPING.keys())
    ],
    "images": [],
    "annotations": []
}

annotation_id = 1  # Unique ID for each annotation

def convert_to_coco_format(image_shape, bbox):
    """
    Convert bounding box coordinates to COCO format.
    For COCO, the format is [x, y, width, height].
    """
    x_min, y_min = bbox[0], bbox[1]
    width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
    return [int(x_min), int(y_min), int(width), int(height)]

def annotate_images(flight_folder, destination_folder):
    """Annotate images and save annotations in COCO-compatible format."""
    global annotation_id

    image_folder = os.path.join(flight_folder, "image")
    segmentation_folder = os.path.join(flight_folder, "segmentation")

    if not os.path.exists(image_folder):
        logging.warning(f"Image folder not found in {flight_folder}. Skipping.")
        return

    if not os.path.exists(segmentation_folder):
        logging.warning(f"Segmentation folder not found in {flight_folder}. Skipping.")
        return

    # Create a destination folder for the flight
    flight_id = os.path.basename(flight_folder)
    flight_dest_folder = os.path.join(destination_folder, flight_id)
    os.makedirs(flight_dest_folder, exist_ok=True)

    for file in tqdm(os.listdir(image_folder), desc=f"Annotating images in {flight_id}", leave=False):
        if file.endswith(".png") or file.endswith(".jpg"):
            image_path = os.path.join(image_folder, file)
            segmentation_path = os.path.join(segmentation_folder, file)

            if not os.path.exists(segmentation_path):
                logging.warning(f"Segmentation file not found for {file}. Skipping.")
                continue

            # Read the base image and segmentation mask
            base_img = cv2.imread(image_path)
            segmentation_img = cv2.imread(segmentation_path, cv2.IMREAD_GRAYSCALE)

            if base_img is None or segmentation_img is None:
                logging.error(f"Failed to read image or segmentation for {file}. Skipping.")
                continue

            img_height, img_width = base_img.shape[:2]
            image_id = len(coco_data["images"]) + 1

            # Add image metadata to COCO dataset
            coco_data["images"].append({
                "id": image_id,
                "file_name": file,
                "height": img_height,
                "width": img_width
            })

            # Iterate over each class in CLASS_MAPPING to find regions and draw bounding boxes
            for label, value in CLASS_MAPPING.items():
                mask = (segmentation_img == value).astype(np.uint8)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for contour in contours:
                    if cv2.contourArea(contour) > 50:  # Filter small contours
                        x, y, w, h = cv2.boundingRect(contour)
                        bbox = convert_to_coco_format(base_img.shape, (x, y, x + w, y + h))
                        category_id = list(CLASS_MAPPING.keys()).index(label)

                        # Add annotation to COCO dataset
                        coco_data["annotations"].append({
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": category_id,
                            "bbox": bbox,
                            "area": w * h,
                            "iscrowd": 0
                        })

                        annotation_id += 1

            # Save the corresponding image in the flight destination folder
            destination_image_path = os.path.join(flight_dest_folder, file)
            shutil.copy(image_path, destination_image_path)

    logging.info(f"Saved annotations and images for {flight_folder} to {flight_dest_folder}")

def main():
    locations = ["neighbourhood", "park"]
    for location in locations:
        location_path = os.path.join(source_base_path, location)
        destination_path = os.path.join(destination_base_path, location)

        for flight_id in os.listdir(location_path):
            flight_folder = os.path.join(location_path, flight_id)
            annotate_images(flight_folder, destination_path)

    # Save COCO annotations
    annotation_file = os.path.join(destination_base_path, "annotations.json")
    with open(annotation_file, "w") as f:
        json.dump(coco_data, f, indent=4)

    logging.info("Annotation process completed for EfficientDet.")

if __name__ == "__main__":
    main()
