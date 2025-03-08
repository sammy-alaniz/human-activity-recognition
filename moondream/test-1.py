import moondream as md
from PIL import Image
import time
import torch
import os

# CPU-specific optimizations
torch.set_num_threads(4)  # Adjust based on your CPU cores (try 2-6 for best performance)
torch.set_num_interop_threads(1)  # Lower for single model scenario

# Set environment variables for better CPU performance
os.environ["OMP_NUM_THREADS"] = "4"  # OpenMP threads - match with torch.set_num_threads
os.environ["MKL_NUM_THREADS"] = "4"  # MKL threads - match with torch.set_num_threads
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"  # Smaller memory splits

# Try to enable PyTorch optimizations
try:
    torch._dynamo.config.cache_size_limit = 64
    torch._dynamo.config.suppress_errors = True
except:
    print("PyTorch dynamic optimizations not available")

# Load model (already using int8 quantized version - good!)
model = md.vl(model="/Users/samuelalaniz/dev/moondream-0_5b-int8.mf")

# Try to reduce model parameters if API allows
try:
    # More aggressive parameter reduction for CPU
    model.config.vision.max_crops = 2      # Minimum viable value
    model.config.vision.overlap_margin = 1  # Minimum viable value
except:
    print("Using default model parameters")

# Load and prepare image (already using good settings)
image = Image.open("/Users/samuelalaniz/dev/school/human-signals/project/1-ws/human-activity-recognition/moondream/burger-1.png")
cpu_optimal_size = (224, 224)  # Good choice for CPU
resized_image = image.resize(cpu_optimal_size, Image.BILINEAR)  # Good choice of method

# Create a simple encoding cache
encoding_cache = {}

def get_encoded_image(image_path, size=(224, 224)):
    cache_key = f"{image_path}_{size[0]}x{size[1]}"
    if cache_key in encoding_cache:
        print("Using cached encoding")
        return encoding_cache[cache_key]
    
    img = Image.open(image_path).resize(size, Image.BILINEAR)
    start_time = time.time()
    encoded = model.encode_image(img)
    encoding_time = time.time() - start_time
    print(f"Encoding time: {encoding_time:.4f} seconds")
    
    encoding_cache[cache_key] = encoded
    return encoded

# Use the caching function
encoded_image = get_encoded_image("/Users/samuelalaniz/dev/school/human-signals/project/1-ws/human-activity-recognition/moondream/burger-1.png")

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

# If you want to process another image, use the cache
# encoded_image2 = get_encoded_image("another_image.png")