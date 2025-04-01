import os
import base64
import requests
import json
import time
import random
from pathlib import Path

def encode_image_to_base64(image_path):
    """Encode an image file to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def classify_image_with_retry(image_path, api_key, max_retries=5, initial_backoff=1):
    """Send an image to llamafile's API with retry logic for rate limits."""
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
    
    # Updated payload structure for llamafile
    payload = {
        "model": "LLaMA_CPP",
        "messages": [
            {
                "role": "user",
                "content": f"[img-1] {prompt}"
            }
        ],
        "image_data": [
            {
                "data": base64_image,
                "id": 1
            }
        ],
        "max_tokens": 3000
    }
    
    # Implement retry with exponential backoff
    retries = 0
    backoff_time = initial_backoff
    
    while retries <= max_retries:
        response = requests.post("http://localhost:8080/v1/chat/completions", headers=headers, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            print(result)
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
        
        # Check if the error is rate limit related (429)
        elif response.status_code == 429:
            error_data = response.json()
            
            # If the API provides a specific wait time, use that
            if "error" in error_data and "message" in error_data["error"]:
                # Extract wait time from error message if available
                error_msg = error_data["error"]["message"]
                if "Please try again in" in error_msg:
                    try:
                        # Extract the wait time (e.g., "303ms")
                        wait_str = error_msg.split("Please try again in ")[1].split(".")[0]
                        if "ms" in wait_str:
                            # Convert milliseconds to seconds
                            suggested_wait = float(wait_str.replace("ms", "")) / 1000
                        elif "s" in wait_str:
                            # Convert seconds
                            suggested_wait = float(wait_str.replace("s", ""))
                        else:
                            # Default to backoff calculation
                            suggested_wait = backoff_time
                            
                        # Add small random jitter (up to 100ms)
                        wait_time = suggested_wait + (random.random() / 10)
                    except:
                        # If parsing fails, use calculated backoff
                        wait_time = backoff_time
                else:
                    wait_time = backoff_time
            else:
                wait_time = backoff_time
            
            print(f"Rate limit exceeded. Retrying in {wait_time:.2f} seconds... (Attempt {retries+1}/{max_retries})")
            time.sleep(wait_time + 1)
            
            # Increase backoff for next attempt (exponential with jitter)
            backoff_time = min(60, backoff_time * 2 * (1 + random.random() * 0.1))
            retries += 1
            continue
        
        # For other errors, retry with backoff but print the error
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            
            if retries < max_retries:
                print(f"Retrying in {backoff_time:.2f} seconds... (Attempt {retries+1}/{max_retries})")
                time.sleep(backoff_time)
                
                # Increase backoff for next attempt (exponential with jitter)
                backoff_time = min(60, backoff_time * 2 * (1 + random.random() * 0.1))
                retries += 1
                continue
            else:
                return None
    
    # If we've exhausted all retries
    print(f"Failed after {max_retries} retries")
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
        
        # Classify the image with retry
        activities = classify_image_with_retry(image_path, api_key)
        
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
        
        # Sleep a small amount between images to avoid hitting rate limits too quickly
        time.sleep(0.1)
    
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
    output_file = "./data/classification_llama_results.json"
    
    # Your OpenAI API key (better to use environment variable)
    api_key = os.environ.get("OPENAI_API_KEY", "no-key")  # For llamafile, "no-key" is fine
    
    # Process all frames
    process_frames(frames_dir, output_file, api_key)