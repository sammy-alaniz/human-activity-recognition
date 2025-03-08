import torch
from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig
from PIL import Image

# Configure 8-bit quantization
quantization_config = BitsAndBytesConfig(load_in_8bit=True)

# Load processor
processor = AutoProcessor.from_pretrained(
    "vikhyatk/moondream2",
    trust_remote_code=True
)

# Load model with 8-bit quantization on CPU
model = AutoModelForCausalLM.from_pretrained(
    "vikhyatk/moondream2",
    trust_remote_code=True,
    quantization_config=quantization_config,
    device_map="cpu"  # Running on CPU
)

# Function to process an image and generate a response
def analyze_image(image_path, question="Describe this image in detail."):
    # Load and process the image
    image = Image.open(image_path)
    
    # Process inputs
    inputs = processor(text=question, images=image, return_tensors="pt").to("cpu")
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            temperature=0.7,
            top_p=0.9,
            num_beams=1
        )
    
    # Decode the response
    response = processor.decode(outputs[0], skip_special_tokens=True)
    
    return response

# Example usage
if __name__ == "__main__":
    # Replace with your image path
    image_path = "/Users/samuelalaniz/dev/school/human-signals/project/1-ws/human-activity-recognition/moondream/burger-1.png"
    
    # Ask a question about the image
    question = "What is shown in this image?"
    
    # Get the response
    response = analyze_image(image_path, question)
    
    print(f"Question: {question}")
    print(f"Response: {response}")