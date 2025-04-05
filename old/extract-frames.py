import os
import cv2
import random
from pathlib import Path

def extract_random_frames(video_path, output_dir, num_frames=3):
    """Extract random frames from a video file and save them as images."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Open the video file
    video = cv2.VideoCapture(str(video_path))
    
    # Check if video opened successfully
    if not video.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames <= 0:
        print(f"Error: Could not determine frame count for {video_path}")
        return
    
    # Generate random frame indices
    if total_frames <= num_frames:
        frame_indices = list(range(total_frames))
    else:
        frame_indices = sorted(random.sample(range(total_frames), num_frames))
    
    # Extract and save the selected frames
    for i, frame_idx in enumerate(frame_indices):
        # Set video to the specific frame
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        
        # Read the frame
        success, frame = video.read()
        
        if success:
            # Generate output filename
            video_filename = os.path.basename(video_path)
            output_filename = f"{os.path.splitext(video_filename)[0]}_frame{i+1}.jpg"
            output_path = os.path.join(output_dir, output_filename)
            
            # Save the frame as an image
            cv2.imwrite(output_path, frame)
            print(f"Saved frame {frame_idx} from {video_filename}")
        else:
            print(f"Error: Could not read frame {frame_idx} from {video_path}")
    
    # Release the video
    video.release()

def process_videos(base_dir, output_base_dir):
    """Process all videos in the directory structure and extract frames."""
    # Create output base directory
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Get all label directories
    label_dirs = [d for d in Path(base_dir).iterdir() if d.is_dir()]
    
    for label_dir in label_dirs:
        label_name = label_dir.name
        print(f"Processing videos for '{label_name}'...")
        
        # Create output directory for this label
        label_output_dir = Path(output_base_dir) / label_name
        os.makedirs(label_output_dir, exist_ok=True)
        
        # Process all video files in this label directory
        video_files = list(label_dir.glob("*.mp4"))
        for video_file in video_files:
            print(f"  Extracting frames from {video_file.name}")
            extract_random_frames(video_file, label_output_dir)

if __name__ == "__main__":
    # Base directory containing the organized video files
    base_dir = './data/k400/filtered_activities'
    
    # Output directory for extracted frames
    output_base_dir = './data/k400/extracted_frames'
    
    # Process all videos
    process_videos(base_dir, output_base_dir)
    
    print("Done! All frames have been extracted.")