from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import time
import torch
import numpy as np

# Check for MPS availability
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Load model with FP16 precision
model = AutoModelForCausalLM.from_pretrained(
    "vikhyatk/moondream2",
    revision="2025-01-09",
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="mps"
)

model = torch.compile(model)

# Load and encode the image (this is the part you want to separate)
image_path = "/Users/samuelalaniz/dev/school/human-signals/project/1-ws/human-activity-recognition/moondream/burger-1.png"
image = Image.open(image_path)
cpu_optimal_size = (224, 224)  # Good choice for CPU
resized_image = image.resize(cpu_optimal_size, Image.BILINEAR)  # Good choice of method

# Encode the image
start_time = time.time()
image_embeds = model.encode_image(resized_image)
encoding_time = time.time() - start_time
print(f"Image encoding time: {encoding_time:.4f} seconds")

# Save the encoding if needed for later use
# torch.save(image_embeds, "burger_encoding.pt")

# Now you can query the encoding multiple times
start_time = time.time()
result1 = model.answer_question(image_embeds, "What is in this image?")
query_time1 = time.time() - start_time
print(f"Query 1 time: {query_time1:.4f} seconds")
print(f"Answer 1: {result1}")

start_time = time.time()
result2 = model.answer_question(image_embeds, "What color is the burger?")
query_time2 = time.time() - start_time
print(f"Query 2 time: {query_time2:.4f} seconds")
print(f"Answer 2: {result2}")

# For comparison, you can also use the combined caption function
start_time = time.time()
caption_result = model.caption(image, length="short")
caption_time = time.time() - start_time
print(f"Combined caption time: {caption_time:.4f} seconds")
print(f"Caption: {caption_result['caption']}")

# To load a saved encoding later:
# loaded_embeds = torch.load("burger_encoding.pt")
# result = model.answer_question(loaded_embeds, "What is in this image?")