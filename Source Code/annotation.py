import os
import cv2
import numpy as np
import json
import logging
from tqdm import tqdm
import shutil

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Paths for the training set
source_base_path = r"E:\DATA 298 A project\Final\DDOS\data\train"  # Update to point to the training set
destination_base_path = r"E:\DATA 298 A project\Annotated Data\Faster R CNN\train"  # Updated destination for Faster R-CNN format

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

def convert_to_faster_rcnn_format(image_shape, bbox):
    """
    Convert bounding box coordinates to Faster R-CNN format.
    For Faster R-CNN, the format is [x_min, y_min, x_max, y_max].
    """
    x_min, y_min = bbox[0], bbox[1]
    x_max, y_max = bbox[0] + bbox[2], bbox[1] + bbox[3]  # x + width, y + height
    return [x_min, y_min, x_max, y_max]

def annotate_images(flight_folder, destination_folder):
    """Annotate images and save annotations in Faster R-CNN compatible format."""
    image_folder = os.path.join(flight_folder, "image")
    segmentation_folder = os.path.join(flight_folder, "segmentation")

    if not os.path.exists(image_folder):
        logging.warning(f"Image folder not found in {flight_folder}. Skipping.")
        return

    if not os.path.exists(segmentation_folder):
        logging.warning(f"Segmentation folder not found in {flight_folder}. Skipping.")
        return

    annotations = []  # Store all annotations for this flight

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
            image_annotations = {
                "file_name": file,
                "height": img_height,
                "width": img_width,
                "annotations": []
            }

            # Iterate over each class in CLASS_MAPPING to find regions and draw bounding boxes
            for label, value in CLASS_MAPPING.items():
                mask = (segmentation_img == value).astype(np.uint8)  # Binary mask for the class
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for contour in contours:
                    if cv2.contourArea(contour) > 50:  # Filter small contours
                        x, y, w, h = cv2.boundingRect(contour)
                        bbox = convert_to_faster_rcnn_format(base_img.shape, (x, y, w, h))
                        class_id = list(CLASS_MAPPING.keys()).index(label)

                        # Add annotation for this bounding box
                        annotation = {
                            "bbox": bbox,
                            "category_id": class_id,
                            "area": w * h,
                            "iscrowd": 0
                        }
                        image_annotations["annotations"].append(annotation)

            # Add image-specific annotations to the flight annotations
            annotations.append(image_annotations)

            # Save the corresponding image in the flight destination folder
            destination_image_path = os.path.join(flight_dest_folder, file)
            shutil.copy(image_path, destination_image_path)

    # Save annotations for the flight folder as a single JSON file
    json_path = os.path.join(flight_dest_folder, f"{flight_id}_annotations.json")
    with open(json_path, "w") as f:
        json.dump(annotations, f, indent=4)

    logging.info(f"Saved annotations and images for {flight_folder} to {flight_dest_folder}")

def main():
    locations = ["neighbourhood", "park"]
    for location in locations:
        location_path = os.path.join(source_base_path, location)
        destination_path = os.path.join(destination_base_path, location)

        for flight_id in os.listdir(location_path):
            flight_folder = os.path.join(location_path, flight_id)
            annotate_images(flight_folder, destination_path)

    logging.info("Annotation process completed for Faster R-CNN.")

if __name__ == "__main__":
    main()
