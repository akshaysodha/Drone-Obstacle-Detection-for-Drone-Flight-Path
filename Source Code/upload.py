import os

from google.cloud import storage

def upload_directory_to_gcs(local_path, bucket_name, gcs_base_path=""):
    """
    Uploads a directory (and its subdirectories) to Google Cloud Storage, preserving the folder structure.

    Args:
        local_path (str): Local directory path to upload.
        bucket_name (str): GCS bucket name.
        gcs_base_path (str): GCS folder path where the data will be uploaded. Default is the root of the bucket.
    """
    # Initialize the GCS client
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # Walk through the directory structure
    for root, _, files in os.walk(local_path):
        for file in files:
            # Local file path
            local_file_path = os.path.join(root, file)
            
            # Construct GCS blob path
            relative_path = os.path.relpath(local_file_path, local_path)
            gcs_blob_path = os.path.join(gcs_base_path, relative_path).replace("\\", "/")  # Ensure GCS-compatible paths
            
            # Upload file
            blob = bucket.blob(gcs_blob_path)
            blob.upload_from_filename(local_file_path)
            print(f"Uploaded {local_file_path} to gs://{bucket_name}/{gcs_blob_path}")

# Set your local path, bucket name, and base GCS path
local_directory = "C:\\Users\\harsh\\Test_drone_data\\data"  # Path to your local data directory
bucket_name = "finalyearbucket"  # Your GCS bucket name
gcs_base_folder = "drone_data"  # Base folder in GCS to store your data

# Call the function to upload the directory
upload_directory_to_gcs(local_directory, bucket_name, gcs_base_folder)