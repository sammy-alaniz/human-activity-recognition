import os
import csv
import shutil
from pathlib import Path

# Paths
csv_file = './data/k400/annotations/filtered_activities.csv'
video_source_dir = './data/k400_targz/train'
output_base_dir = './data/filtered_activities'

# Create output base directory if it doesn't exist
os.makedirs(output_base_dir, exist_ok=True)

# Dictionary to map labels to their video files
label_to_videos = {}

# Read the filtered CSV file
with open(csv_file, 'r', newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        label = row['label']
        youtube_id = row['youtube_id']
        time_start = int(row['time_start'])
        time_end = int(row['time_end'])
        
        # Construct the expected video filename
        video_filename = f"{youtube_id}_{time_start:06d}_{time_end:06d}.mp4"
        
        # Add to the mapping dictionary
        if label not in label_to_videos:
            label_to_videos[label] = []
        
        label_to_videos[label].append(video_filename)

# Process each label
for label, video_files in label_to_videos.items():
    # Create folder for this label
    label_dir = os.path.join(output_base_dir, label.replace(' ', '_'))
    os.makedirs(label_dir, exist_ok=True)
    
    print(f"Processing {len(video_files)} videos for label '{label}'")
    
    # Move each video file to its label folder
    for video_file in video_files:
        source_path = os.path.join(video_source_dir, video_file)
        dest_path = os.path.join(label_dir, video_file)
        
        # Check if source file exists
        if os.path.exists(source_path):
            shutil.copy2(source_path, dest_path)  # Using copy2 to preserve metadata
            print(f"  Copied: {video_file}")
        else:
            print(f"  Not found: {video_file}")

print("Done! Files have been organized by activity label.")