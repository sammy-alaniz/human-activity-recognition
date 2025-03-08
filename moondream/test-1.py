import moondream as md
from PIL import Image
import time

model = md.vl(model="/Users/samuelalaniz/dev/moondream-0_5b-int8.mf")



image = Image.open("/Users/samuelalaniz/dev/school/human-signals/project/1-ws/human-activity-recognition/moondream/burger-1.png")
# Resize to smallest useful size (even smaller than GPU)
cpu_optimal_size = (224, 224)  # More aggressive reduction for CPU
resized_image = image.resize(cpu_optimal_size, Image.BILINEAR)  # BILINEAR is faster than LANCZOS

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