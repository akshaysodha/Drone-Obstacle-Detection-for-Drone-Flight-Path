import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.io.filesystems import FileSystems
import cv2
import numpy as np
import os

# Define class mappings
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

def process_image(element):
    """Reads an image from GCS, processes it, and generates YOLO annotations."""
    input_path, segmentation_path = element

    # Read image from GCS
    with FileSystems.open(input_path) as f:
        base_img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
    with FileSystems.open(segmentation_path) as f:
        segmentation_img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    
    if base_img is None or segmentation_img is None:
        raise RuntimeError(f"Failed to load images from {input_path}")

    # Generate YOLO annotations
    annotations = []
    for label, value in CLASS_MAPPING.items():
        mask = (segmentation_img == value).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > 50:  # Filter out small contours
                x, y, w, h = cv2.boundingRect(contour)
                x_center = (x + w / 2) / base_img.shape[1]
                y_center = (y + h / 2) / base_img.shape[0]
                width = w / base_img.shape[1]
                height = h / base_img.shape[0]
                class_id = list(CLASS_MAPPING.keys()).index(label)
                annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    
    # Return the image path and annotations
    return input_path, '\n'.join(annotations)

def save_annotations(element, output_dir):
    """Writes YOLO annotations to the output directory in GCS."""
    input_path, annotation_data = element
    output_path = os.path.join(output_dir, os.path.basename(input_path).replace('.png', '.txt').replace('.jpg', '.txt'))
    with FileSystems.create(output_path) as f:
        f.write(annotation_data.encode('utf-8'))

def main():
    # Define input and output GCS paths
    input_base_path = "gs://newbucket010/test/"
    output_base_path = "gs://newbucket010/test/annotations/"
    
    # Initialize pipeline options
    options = PipelineOptions(
        runner='DataflowRunner',  # Use 'DirectRunner' for local testing
        project='dronefinalyear',  # Your GCP project ID
        temp_location='gs://newbucket010/temp',  # Temp folder in GCS
        region='us-central1',  # Change to your preferred region
        service_account_email='775182290299-compute@developer.gserviceaccount.com'
    )

    with beam.Pipeline(options=options) as p:
        (
            p
            | "List Images" >> beam.Create([
                (f"{input_base_path}neighbourhood/image.png", f"{input_base_path}neighbourhood/segmentation.png"),
                (f"{input_base_path}park/image.png", f"{input_base_path}park/segmentation.png")
            ])  # Replace with dynamic discovery logic if needed
            | "Process Images" >> beam.Map(process_image)
            | "Save Annotations" >> beam.Map(save_annotations, output_dir=output_base_path)
        )

if __name__ == "__main__":
    main()
