import pandas as pd

# Load the CSV file
df = pd.read_csv("C:/Users/vabss/Downloads/METADATA.csv")

# Check for the presence of 'timestamp.1' and remove it
if 'timestamp.1' in df.columns:
    df = df.drop(columns=['timestamp.1'])  # Drop the 'timestamp.1' column

# Save the modified CSV
df.to_csv("C:/Users/vabss/Downloads/Main_Metadata.csv", index=False)

print("Modified CSV saved as Main_Metadata.csv")
