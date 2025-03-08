from transformers import AutoModelForCausalLM
from PIL import Image
import time
import torch

# Load model with FP16 precision
model = AutoModelForCausalLM.from_pretrained(
    "vikhyatk/moondream2",
    revision="2025-01-09",
    trust_remote_code=True,
    torch_dtype=torch.float16
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