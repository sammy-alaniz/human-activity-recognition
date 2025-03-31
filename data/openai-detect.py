import os
import base64
import requests
import json
import time
from pathlib import Path

def encode_image_to_base64(image_path):
    """Encode an image file to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def classify_image(image_path, api_key):
    """Send an image to OpenAI's Vision API and ask which activities it shows."""
    base64_image = encode_image_to_base64(image_path)
    
    prompt = """
Look at the provided image and determine whether it shows any of the following activities. For each category, respond only with "yes" or "no" based on what's visible in the image.

Respond in this exact format:
- yoga: [yes/no]
- braiding hair: [yes/no]
- brushing teeth: [yes/no]
- carrying baby: [yes/no]
- cleaning floor: [yes/no]
- cleaning gutters: [yes/no]
- cleaning pool: [yes/no]
- cleaning shoes: [yes/no]
- cleaning toilet: [yes/no]
- cleaning windows: [yes/no]
- cooking chicken: [yes/no]
- cooking egg: [yes/no]
- cooking sausages: [yes/no]
- making a cake: [yes/no]
- making a sandwich: [yes/no]
- making bed: [yes/no]
- making jewelry: [yes/no]
- making pizza: [yes/no]
- making tea: [yes/no]
- washing dishes: [yes/no]
- washing hands/hair: [yes/no]
- watering plants: [yes/no]

Important: Only use "yes" if you're confident the specific activity is clearly shown in the image. If you're unsure or the activity is only partially visible, respond with "no".
"""
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
        "model": "gpt-4o-mini", # Updated model name from deprecated gpt-4-vision-preview
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 300
    }
    
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    
    if response.status_code == 200:
        result = response.json()
        answer_text = result["choices"][0]["message"]["content"].strip()
        
        # Parse the response to extract activity classifications
        activities = {}
        for line in answer_text.split("\n"):
            if ":" in line:
                parts = line.split(":")
                activity = parts[0].strip("- ")
                is_present = "yes" in parts[1].lower()
                activities[activity] = is_present
        
        return activities
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None

def process_frames(frames_dir, output_file, api_key):
    """Process all frames in the directory and classify them."""
    results = []
    
    # Get all image files
    image_files = []
    for root, _, files in os.walk(frames_dir):
        for file in files:
            if file.endswith('.jpg'):
                image_path = os.path.join(root, file)
                image_files.append(image_path)
    
    print(f"Found {len(image_files)} images to process")
    
    # Process each image
    for i, image_path in enumerate(image_files):
        print(f"Processing image {i+1}/{len(image_files)}: {os.path.basename(image_path)}")
        
        # Classify the image
        activities = classify_image(image_path, api_key)
        
        if activities:
            # Get the original directory label (label=activity folder name)
            relative_path = os.path.relpath(image_path, frames_dir)
            parts = relative_path.split(os.path.sep)
            source_label = parts[0] if len(parts) > 1 else "unknown"
            
            # Extract original video name and frame number
            filename = os.path.basename(image_path)
            video_parts = filename.split("_frame")
            video_name = video_parts[0] if len(video_parts) > 0 else filename
            frame_num = video_parts[1].split(".")[0] if len(video_parts) > 1 else "unknown"
            
            # Store the result
            result = {
                "image_path": image_path,
                "source_label": source_label,
                "video_name": video_name,
                "frame": f"frame{frame_num}",
                "detected_activities": activities
            }
            
            results.append(result)
            
            # Print immediate results
            detected = [act for act, is_present in activities.items() if is_present]
            print(f"  Detected activities: {detected if detected else 'None'}")
        else:
            print(f"  Failed to classify {image_path}")
        
        # Sleep to avoid rate limiting
        time.sleep(0.5)
    
    # Write results to JSON file
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate summary statistics
    activity_stats = {}
    for result in results:
        for activity, is_present in result["detected_activities"].items():
            if activity not in activity_stats:
                activity_stats[activity] = {"total": 0, "detected": 0}
            activity_stats[activity]["total"] += 1
            if is_present:
                activity_stats[activity]["detected"] += 1
    
    print("\nActivity Detection Summary:")
    for activity, stats in sorted(activity_stats.items()):
        percentage = (stats["detected"] / stats["total"]) * 100 if stats["total"] > 0 else 0
        print(f"  {activity}: {stats['detected']}/{stats['total']} ({percentage:.1f}%)")
    
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    # Directory containing the extracted frames
    frames_dir = "./data/k400/extracted_frames"
    
    # Output file for results
    output_file = "./data/classification_results.json"
    
    # Your OpenAI API key (better to use environment variable)
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        api_key = "your_openai_api_key"  # Replace with your actual API key
    
    # Process all frames
    process_frames(frames_dir, output_file, api_key)