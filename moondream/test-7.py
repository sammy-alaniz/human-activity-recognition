import moondream as md
from PIL import Image
import time

model = md.vl(model="/Users/samuelalaniz/dev/moondream-0_5b-int8.mf")

# print(f'Max Crops Initial : {model.config.vision.max_crops}')
# print(f'Overlap Margin Initial : {model.config.vision.overlap_margin}')
# model.config.vision.max_crops = 6  # Try a smaller number
# model.config.vision.overlap_margin = 2  # Reduce overlap (default is 4)

image = Image.open("/Users/samuelalaniz/dev/school/human-signals/project/1-ws/human-activity-recognition/moondream/burger-1.png")
cpu_optimal_size = (224, 224)  # Good choice for CPU
resized_image = image.resize(cpu_optimal_size, Image.BILINEAR)  # Good choice of method

# Time the encode_image method
start_time = time.time()
encoded_image = model.encode_image(resized_image)
end_time = time.time()
encoding_time = end_time - start_time
print(f"Encoding time: {encoding_time:.4f} seconds")

# print(f'Max Crops Final : {model.config.vision.max_crops}')
# print(f'Overlap Margin Final : {model.config.vision.overlap_margin}')
