import moondream as md
from PIL import Image
import time

# Initialize with local model path. Can also read .mf.gz files, but we recommend decompressing
# up-front to avoid decompression overhead every time the model is initialized.
model = md.vl(model="/Users/samuelalaniz/dev/moondream-0_5b-int4.mf")

# Load and process image
image = Image.open("/Users/samuelalaniz/dev/burger-1.png")
start_time = time.time()
encoded_image = model.encode_image(image)
delta_time = time.time() - start_time
print(f"Image encoding time: {delta_time:.4f} seconds")

# Generate caption
start_time = time.time()
answer = model.query(encoded_image, "Is the human eating food? respond with yes or no.")["answer"]
delta_time = time.time() - start_time
print(f"Image query time: {delta_time:.4f} seconds")

print("Answer:", answer)