import os
import csv
from google.cloud import storage

# Initialize Google Cloud Storage client
client = storage.Client()
bucket_name = "dataset_298a"  # Replace with your actual bucket name
bucket = client.get_bucket(bucket_name)

# Base path in GCS where the DDOS data is stored
base_path = "DDOS/data"  # Adjust this path as per your bucket structure

# Datasets to process: train, test, validation
datasets = ["train", "test", "validation"]

# Output CSV file to store the metadata (in GCS)
output_file = "/tmp/METADATA.csv"  # Temp location in your VM, will upload it later to GCS

# Define the columns dynamically based on sample metadata and weather CSVs
metadata_columns = []
weather_columns = []

# Sample file paths to extract columns dynamically from GCS
sample_metadata_blob = bucket.blob(f"{base_path}/train/neighbourhood/0/metadata.csv")
sample_weather_blob = bucket.blob(f"{base_path}/train/neighbourhood/0/weather.csv")

# Extract the metadata columns (if metadata exists in GCS)
if sample_metadata_blob.exists():
    sample_metadata_blob.download_to_filename("/tmp/sample_metadata.csv")
    with open("/tmp/sample_metadata.csv", mode='r') as sample_meta_file:
        metadata_columns = next(csv.DictReader(sample_meta_file)).keys()

# Extract the weather columns (if weather exists in GCS)
if sample_weather_blob.exists():
    sample_weather_blob.download_to_filename("/tmp/sample_weather.csv")
    with open("/tmp/sample_weather.csv", mode='r') as sample_weather_file:
        weather_columns = next(csv.DictReader(sample_weather_file)).keys()

# Define the basic columns and append the metadata and weather columns dynamically
columns = ['flight_id', 'dataset', 'location', 'timestamp', 'image_path', 'depth_path', 
           'segmentation_path', 'flow_path', 'surface_normals_path', 'weather_condition']

columns.extend(metadata_columns)  # Add all metadata columns
columns.extend(weather_columns)  # Add all weather columns

# Open the final metadata CSV for writing
with open(output_file, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=columns)
    writer.writeheader()

    # Loop through each dataset (train, test, validation)
    for dataset in datasets:
        dataset_path = f"{base_path}/{dataset}"
        
        # List all the objects (flights) in the base path of GCS
        blobs = bucket.list_blobs(prefix=dataset_path)

        # Store paths that need to be iterated over
        flight_paths = set()

        # Find unique flight paths
        for blob in blobs:
            path_parts = blob.name.split('/')
            if len(path_parts) >= 4:
                flight_paths.add('/'.join(path_parts[:4]))

        # Loop through each flight path
        for flight_path in flight_paths:
            path_parts = flight_path.split('/')
            location = path_parts[2]
            flight_id = path_parts[3]

            # Define the GCS paths for image, depth, segmentation, flow, surface normals
            image_path = f"gs://{bucket_name}/{flight_path}/image/0.png"
            depth_path = f"gs://{bucket_name}/{flight_path}/depth/0.png"
            segmentation_path = f"gs://{bucket_name}/{flight_path}/segmentation/0.png"
            flow_path = f"gs://{bucket_name}/{flight_path}/flow/0.png"
            surface_normals_path = f"gs://{bucket_name}/{flight_path}/surfacenormals/0.png"

            # Initialize a dictionary to hold the metadata and weather values
            metadata_values = {key: "N/A" for key in metadata_columns}
            weather_values = {key: "N/A" for key in weather_columns}
            timestamp = "N/A"
            weather_condition = "N/A"

            # Download and read metadata and weather files from GCS (if they exist)
            metadata_blob = bucket.blob(f"{flight_path}/metadata.csv")
            weather_blob = bucket.blob(f"{flight_path}/weather.csv")

            if metadata_blob.exists():
                metadata_blob.download_to_filename("/tmp/metadata.csv")
                with open("/tmp/metadata.csv", mode='r') as meta_file:
                    meta_reader = csv.DictReader(meta_file)
                    first_row = next(meta_reader)
                    metadata_values.update(first_row)  # Update metadata values

            if weather_blob.exists():
                weather_blob.download_to_filename("/tmp/weather.csv")
                with open("/tmp/weather.csv", mode='r') as weather_file:
                    weather_reader = csv.DictReader(weather_file)
                    first_row = next(weather_reader)
                    weather_values.update(first_row)  # Update weather values

            # Write the metadata for this flight
            writer.writerow({
                'flight_id': flight_id,
                'dataset': dataset,  # Add dataset to the metadata (train/test/validation)
                'location': location,
                'timestamp': metadata_values.get('timestamp', 'N/A'),  # Example timestamp from metadata
                'image_path': image_path,
                'depth_path': depth_path,
                'segmentation_path': segmentation_path,
                'flow_path': flow_path,  # Add flow path
                'surface_normals_path': surface_normals_path,  # Add surface normals path
                'weather_condition': weather_values.get('condition', 'N/A'),  # Example weather condition
                **metadata_values,  # Add all metadata fields dynamically
                **weather_values  # Add all weather fields dynamically
            })

# Upload the final metadata CSV back to GCS
output_blob = bucket.blob("DDOS/Main_Metadata/CloudMETADATA.csv")
output_blob.upload_from_filename(output_file)

print("CloudMetadata CSV has been generated and uploaded to Google Cloud Storage.")
