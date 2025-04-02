import os

import moondream as md
from PIL import Image
import time

def runMoon(frames_dir, labels_to_check):
    print('Starting runMoon')
    model = md.vl(model="/Users/samuelalaniz/dev/llms/moondream-2b-int8.mf")

    # Get all image files
    image_files = []
    for root, _, files in os.walk(frames_dir):
        for file in files:
            if file.endswith('.jpg'):
                image_path = os.path.join(root, file)
                image_files.append(image_path)

    print(f'Number of image encodings : {len(image_files)}')
    print(f'Number of querys : {len(labels_to_check)*len(image_files)}')

    for full_path in image_files:
        directory_name = os.path.basename(os.path.dirname(full_path))
        filename = os.path.basename(full_path)
        print(f'Activity : {directory_name}')
        print(f'Filename : {filename}')
        # Load and process image
        image = Image.open(full_path)
        cpu_optimal_size = (224, 224)
        resized_image = image.resize(cpu_optimal_size, Image.BILINEAR)

        start_time = time.time()
        encoded_image = model.encode_image(resized_image)
        delta_time = time.time() - start_time
        print(f"Image encoding time: {delta_time:.4f} seconds")

        for label in labels_to_check:
            prompt = f"Describe the activity, then answer -> Does it look like the activity '{label}' could be visible in the image? (yes/no)"
            print(f'POMPT : {prompt}')
            answer = model.query(encoded_image, prompt)["answer"]
            print(f'ANSWER : {answer}')


if __name__ == "__main__":
    labels = ['cooking', 'cleaning']
    runMoon("./data/k400/extracted_frames", labels)