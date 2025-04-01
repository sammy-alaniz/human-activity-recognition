import moondream as md
from PIL import Image
import time

# Initialize with local model path. Can also read .mf.gz files, but we recommend decompressing
# up-front to avoid decompression overhead every time the model is initialized.
model = md.vl(model="/Users/samuelalaniz/dev/llms/moondream-2b-int8.mf")



# Load and process image
image = Image.open("/Users/samuelalaniz/dev/school/human-signals/project/1-ws/human-activity-recognition/data/k400/extracted_frames/cooking_chicken/-07Ke73N4zI_000612_000622_frame1.jpg")
cpu_optimal_size = (224, 224)
resized_image = image.resize(cpu_optimal_size, Image.BILINEAR)
rgb_image = resized_image.convert('RGB')

start_time = time.time()
encoded_image = model.encode_image(resized_image)
delta_time = time.time() - start_time
print(f"Image encoding time: {delta_time:.4f} seconds")

# Generate caption
start_time = time.time()
answer = model.query(encoded_image, "Human eating food? (yes/no)")["answer"]
delta_time = time.time() - start_time
print(f"Image query time: {delta_time:.4f} seconds")

print("Answer:", answer)