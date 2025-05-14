import pandas as pd
from google.cloud import storage

# Initialize Google Cloud Storage client
client = storage.Client()
bucket_name = "dataset_298a"  # Replace with your actual bucket name
bucket = client.bucket(bucket_name)

# Path to the CSV in GCS
gcs_input_file = "DDOS/Main_Metadata/CloudMETADATA.csv"  # GCS path to the existing metadata file
gcs_output_file = "DDOS/Main_Metadata/Main_Metadata.csv"  # GCS path to save the modified CSV

# Local temporary file paths
local_input_file = "/tmp/CloudMETADATA.csv"
local_output_file = "/tmp/Cloud_Main_Metadata.csv"

# Download the CSV file from GCS to a temporary location
blob = bucket.blob(gcs_input_file)
blob.download_to_filename(local_input_file)

# Load the CSV file
df = pd.read_csv(local_input_file)

# Check for the presence of 'timestamp.1' and remove it
if 'timestamp.1' in df.columns:
    df = df.drop(columns=['timestamp.1'])  # Drop the 'timestamp.1' column

# Save the modified CSV locally
df.to_csv(local_output_file, index=False)

# Upload the modified CSV back to GCS
output_blob = bucket.blob(gcs_output_file)
output_blob.upload_from_filename(local_output_file)

print(f"Modified CSV saved and uploaded to {gcs_output_file} in Google Cloud Storage")
