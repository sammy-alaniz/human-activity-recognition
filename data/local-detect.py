import os
import base64
import requests
import json

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
    

    response = requests.post("http://localhost:8080/v1/chat/completions", headers=headers, json=payload)
        
    if response.status_code == 200:
        result = response.json()
        return result
    
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
        result = classify_image_with_retry(image_path, api_key)
    
        # Write results to JSON file
        with open(output_file, 'a') as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    # Directory containing the extracted frames
    frames_dir = "./data/k400/extracted_frames"
    
    # Output file for results
    output_file = "./data/classification_llama_results.json"
    
    # Your OpenAI API key (better to use environment variable)
    api_key = os.environ.get("OPENAI_API_KEY", "no-key")  # For llamafile, "no-key" is fine
    
    # Process all frames
    process_frames(frames_dir, output_file, api_key)