import os
import csv

# Base path where the DDOS data is stored locally
base_path = "E:\Final\DDOS\data"  # Change this to your actual path

# Datasets to process: train, test, validation
datasets = ["train", "test", "validation"]

# Output CSV file to store the metadata
output_file = "C:/Users/vabss/Downloads/METADATA.csv"  # Save the file locally

# Open the first metadata and weather CSV to extract the exact column names dynamically
metadata_columns = []
weather_columns = []

# Set up paths to sample metadata and weather files locally
sample_metadata_file = os.path.join(base_path, "train", "neighbourhood", "0", "metadata.csv")
sample_weather_file = os.path.join(base_path, "train", "neighbourhood", "0", "weather.csv")

# Extract the metadata columns (if metadata exists)
if os.path.exists(sample_metadata_file):
    with open(sample_metadata_file, mode='r') as sample_meta_file:
        metadata_columns = next(csv.DictReader(sample_meta_file)).keys()

# Extract the weather columns (if weather exists)
if os.path.exists(sample_weather_file):
    with open(sample_weather_file, mode='r') as sample_weather_file:
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
        dataset_path = os.path.join(base_path, dataset)
        
        # Loop through each location (e.g., neighbourhood, park)
        for location in os.listdir(dataset_path):
            location_path = os.path.join(dataset_path, location)

            if os.path.isdir(location_path):
                # Loop through each flight (e.g., 0, 1, 2, ...)
                for flight_id in os.listdir(location_path):
                    flight_path = os.path.join(location_path, flight_id)
                    
                    if os.path.isdir(flight_path):
                        # Extract the paths to the image, depth, segmentation, flow, surface normals
                        image_path = os.path.join(flight_path, "image", "0.png")
                        depth_path = os.path.join(flight_path, "depth", "0.png")
                        segmentation_path = os.path.join(flight_path, "segmentation", "0.png")
                        flow_path = os.path.join(flight_path, "flow", "0.png")
                        surface_normals_path = os.path.join(flight_path, "surfacenormals", "0.png")

                        # Initialize a dictionary to hold the metadata and weather values
                        metadata_values = {key: "N/A" for key in metadata_columns}
                        weather_values = {key: "N/A" for key in weather_columns}
                        timestamp = "N/A"
                        weather_condition = "N/A"

                        # Extract metadata from metadata.csv (if it exists)
                        metadata_file = os.path.join(flight_path, "metadata.csv")
                        if os.path.exists(metadata_file):
                            with open(metadata_file, mode='r') as meta_file:
                                meta_reader = csv.DictReader(meta_file)
                                first_row = next(meta_reader)
                                metadata_values.update(first_row)  # Update metadata values

                        # Extract weather data from weather.csv (if it exists)
                        weather_file = os.path.join(flight_path, "weather.csv")
                        if os.path.exists(weather_file):
                            with open(weather_file, mode='r') as weather_file:
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

print("Metadata CSV has been generated locally.")
