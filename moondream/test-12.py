import torch
from PIL import Image
from moondream.torch.moondream import MoondreamModel, MoondreamConfig
from moondream.torch.weights import load_weights_into_model
from moondream.torch.config import MoondreamConfig
import json

# 1. Load appropriate config for 0.5B model (if that's the model size you're using)
# Option 1: Use the built-in config file
with open("./moondream/config/config_md05.json", "r") as f:
    config_dict = json.load(f)
    config = MoondreamConfig.from_dict(config_dict)

# Option 2: Create a config with the appropriate dimensions
# config = MoondreamConfig()
# config.text.dim = 1024  # Set to 1024 instead of 2048
# config.text.ff_dim = 4096  # And appropriate other dimensions

# Initialize model with the correct config
model = MoondreamModel(config)

# Load the safetensors weights
load_weights_into_model("/Users/samuelalaniz/dev/model-int4.safetensors", model)

# Process image, etc.