from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image

import time

import torch

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Then update your model loading
model = AutoModelForCausalLM.from_pretrained(
    "vikhyatk/moondream2",
    revision="2025-01-09",
    trust_remote_code=True,
    torch_dtype=torch.float16,  # Add this line
    device_map="mps"
)

image = Image.open("/Users/samuelalaniz/dev/school/human-signals/project/1-ws/human-activity-recognition/moondream/burger-1.png")



start_time = time.time()
print(model.caption(image, length="short")["caption"])
caption_time = time.time() - start_time
print(f"Encoding time: {caption_time:.4f} seconds")

# Captioning
print("Short caption:")
print(model.caption(image, length="short")["caption"])

print("\nNormal caption:")
for t in model.caption(image, length="normal", stream=True)["caption"]:
    # Streaming generation example, supported for caption() and detect()
    print(t, end="", flush=True)
print(model.caption(image, length="normal"))

# Visual Querying
print("\nVisual query: 'How many people are in the image?'")
print(model.query(image, "How many people are in the image?")["answer"])

# Object Detection
print("\nObject detection: 'face'")
objects = model.detect(image, "face")["objects"]
print(f"Found {len(objects)} face(s)")

# Pointing
print("\nPointing: 'person'")
points = model.point(image, "person")["points"]
print(f"Found {len(points)} person(s)")
