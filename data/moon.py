import os
import json
import signal
import sys
import moondream as md
from PIL import Image
import time

# Flag to indicate if the script should exit
keep_running = True

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    global keep_running
    print("\nCtrl+C detected. Shutting down gracefully...")
    keep_running = False

def runMoon(frames_dir, prompts):
    print('Starting runMoon')
    model = md.vl(model="/Users/samuelalaniz/dev/llms/moondream-0_5b-int4.mf")

    # Get all image files
    image_files = []
    for root, _, files in os.walk(frames_dir):
        for file in files:
            if file.endswith('.jpg'):
                image_path = os.path.join(root, file)
                image_files.append(image_path)

    print(f'Number of image encodings : {len(image_files)}')
    print(f'Number of querys : {len(prompts)*len(image_files)}')

    all_test_cases = []

    for full_path in image_files:
        # Check if we should stop due to Ctrl+C
        if not keep_running:
            break

        test_case = {}
        directory_name = os.path.basename(os.path.dirname(full_path))
        filename = os.path.basename(full_path)
        print(f'Activity : {directory_name}')
        print(f'Filename : {filename}')
        test_case['activity'] = directory_name
        test_case['filename'] = filename

        # Load and process image
        image = Image.open(full_path)
        cpu_optimal_size = (374, 374)
        resized_image = image.resize(cpu_optimal_size, Image.BILINEAR)

        start_time = time.time()
        encoded_image = model.encode_image(resized_image)
        delta_time = time.time() - start_time
        print(f"Image encoding time: {delta_time:.4f} seconds")
        test_case['encoding-time'] = delta_time

        queries = []
        for prompt in prompts:
            if not keep_running:  # Check again before querying
                break
            print(f'POMPT : {prompt}')
            start_time = time.time()
            answer = model.query(encoded_image, prompt)["answer"]
            delta_time = time.time() - start_time
            print(f'ANSWER : {answer}')
            queries.append((prompt, answer, delta_time))
        test_case['queries'] = queries
        all_test_cases.append(test_case)

        # Write individual test case to results.json
        try:
            with open('./results.json', 'a') as f:
                json_string = json.dumps(test_case)
                f.write(json_string + '\n')
        except Exception as e:
            print(f"Error writing to results.json: {e}")

    # Write all test cases to all_test_cases.json only if not interrupted
    if keep_running:
        try:
            with open('./all_test_cases.json', 'w') as f:
                json.dump(all_test_cases, f, indent=4)
        except Exception as e:
            print(f"Error writing to all_test_cases.json: {e}")
    else:
        print("Saving incomplete results to all_test_cases.json...")
        try:
            with open('./all_test_cases.json', 'w') as f:
                json.dump(all_test_cases, f, indent=4)
        except Exception as e:
            print(f"Error writing incomplete results: {e}")

    if not keep_running:
        sys.exit(0)  # Exit cleanly

if __name__ == "__main__":
    # Register the signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)

    labels = ['Describe the image',
              'Does it look like someone is cleaning?']
    runMoon("./k400/frame/train/", labels)