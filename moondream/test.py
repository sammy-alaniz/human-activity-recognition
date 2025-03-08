import moondream as md
from PIL import Image
import time

# Initialize with local model path. Can also read .mf.gz files, but we recommend decompressing
# up-front to avoid decompression overhead every time the model is initialized.
model = md.vl(model="/Users/samuelalaniz/dev/moondream-0_5b-int8.mf")

# Load and process image
image = Image.open("/Users/samuelalaniz/dev/school/human-signals/project/1-ws/human-activity-recognition/moondream/burger-1.png")

# Time the encode_image method
start_time = time.time()  # Record start time
encoded_image = model.encode_image(image)
end_time = time.time()    # Record end time

elapsed_time = end_time - start_time  # Calculate difference
print(f"Encoding time: {elapsed_time:.4f} seconds")

# Ask questions
start_time = time.time()  # Record start time
answer = model.query(encoded_image, "Is this guy eating? Respond with yes or no")["answer"]
end_time = time.time()    # Record end time

elapsed_time = end_time - start_time  # Calculate difference
print(f"Query time: {elapsed_time:.4f} seconds")

print("Answer:", answer)