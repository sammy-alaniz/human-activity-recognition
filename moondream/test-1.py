import moondream as md
from PIL import Image
import time
import torch

# Enable PyTorch optimizations
torch._dynamo.config.cache_size_limit = 64
torch._dynamo.config.suppress_errors = True

# Initialize with local model path
model = md.vl(model="/Users/samuelalaniz/dev/moondream-0_5b-int8.mf")

# Configure for speed (if the API exposes these parameters)
try:
    # Reduce max crops and overlap margin
    model.config.vision.max_crops = 4     # Default is 12 
    model.config.vision.overlap_margin = 2  # Default is 4
except:
    print("Unable to modify vision config params - using defaults")

# Enable model compilation if available
try:
    model.compile()
except:
    print("Model compilation not available in this API")

# Load the image and resize
image = Image.open("/Users/samuelalaniz/dev/school/human-signals/project/1-ws/human-activity-recognition/moondream/burger-1.png")
# Resize to 378×378 (the model's native crop size) or smaller for maximum speed
resized_image = image.resize((378, 378), Image.LANCZOS)

# Time the encode_image method
start_time = time.time()
encoded_image = model.encode_image(resized_image)
end_time = time.time()
encoding_time = end_time - start_time
print(f"Encoding time: {encoding_time:.4f} seconds")

# Process multiple queries with the same encoded image
questions = [
    "Is this guy eating? Respond with yes or no",
    "What is the person doing in this image?",
    "What food item is shown in the image?"
]

for question in questions:
    start_time = time.time()
    answer = model.query(encoded_image, question)["answer"]
    end_time = time.time()
    query_time = end_time - start_time
    print(f"Query: '{question}'")
    print(f"Answer: {answer}")
    print(f"Query time: {query_time:.4f} seconds\n")