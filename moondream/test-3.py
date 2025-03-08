import torch
import json
import os

import sys
sys.path.append("/Users/samuelalaniz/dev/school/human-signals/project/1-ws/moondream/moondream") 

from PIL import Image
from moondream.torch.config import MoondreamConfig
from moondream.torch.moondream import MoondreamModel
from moondream.torch.weights import load_weights_into_model
from typing import List, Dict, Union, Optional

class OptimizedMoondream:
    """Wrapper for Moondream model with optimized settings"""
    
    def __init__(
        self, 
        model_path: str,
        config_path: Optional[str] = None,
        use_smaller_model: bool = False,
        max_crops: int = 8,
        overlap_margin: int = 2,
        compile_model: bool = True,
        device: Optional[str] = None
    ):
        """
        Initialize OptimizedMoondream
        
        Args:
            model_path: Path to model weights file (.safetensors or .pt)
            config_path: Optional path to config JSON file
            use_smaller_model: If True, use the smaller MD05 config
            max_crops: Number of image crops to use (fewer = faster)
            overlap_margin: Overlap margin for crops (smaller = faster)
            compile_model: Whether to use torch.compile()
            device: Force specific device (None for auto-detect)
        """
        # Set device
        if device:
            torch.set_default_device(device)
        elif torch.cuda.is_available():
            torch.set_default_device("cuda")
            # Enable TF32 for better performance on newer GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        elif torch.backends.mps.is_available():
            torch.set_default_device("mps")
        else:
            # CPU optimizations
            torch.set_num_threads(min(os.cpu_count(), 8))
            torch.set_num_interop_threads(1)
        
        # Load configuration
        if config_path:
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
                config = MoondreamConfig.from_dict(config_dict)
        elif use_smaller_model:
            # Load smaller MD05 config from the repo
            md05_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "config", "config_md05.json"
            )
            if os.path.exists(md05_path):
                with open(md05_path, 'r') as f:
                    config_dict = json.load(f)
                    config = MoondreamConfig.from_dict(config_dict)
            else:
                print("MD05 config not found, using default config")
                config = MoondreamConfig()
        else:
            config = MoondreamConfig()
        
        # Modify config for speed
        from copy import deepcopy
        vision_config = deepcopy(config.vision)
        vision_config.max_crops = max_crops
        vision_config.overlap_margin = overlap_margin
        config = MoondreamConfig(
            text=config.text,
            vision=vision_config,
            region=config.region,
            tokenizer=config.tokenizer
        )
        
        # Create model
        self.model = MoondreamModel(config)
        load_weights_into_model(model_path, self.model)
        
        # Compile model if requested
        if compile_model:
            self.model.compile()
        
        # Create cache for encoded images
        self.image_cache = {}
        
    def encode_image(self, image: Union[str, Image.Image], cache_key: Optional[str] = None):
        """
        Encode an image and optionally cache the result
        
        Args:
            image: PIL Image or path to image file
            cache_key: Optional key for caching. If None, no caching is used
            
        Returns:
            EncodedImage object
        """
        # Check cache first
        if cache_key and cache_key in self.image_cache:
            return self.image_cache[cache_key]
        
        # Load image if path is provided
        if isinstance(image, str):
            image = Image.open(image)
            
        # Encode image
        with torch.inference_mode():
            encoded = self.model.encode_image(image)
            
        # Cache if requested
        if cache_key:
            self.image_cache[cache_key] = encoded
            
        return encoded
    
    def process_image(
        self, 
        image: Union[str, Image.Image],
        tasks: List[Dict],
        cache_key: Optional[str] = None
    ):
        """
        Process an image with multiple tasks efficiently
        
        Args:
            image: PIL Image or path to image file
            tasks: List of task dictionaries with 'type' and 'params' keys
            cache_key: Optional key for encoding cache
            
        Returns:
            List of results for each task
        """
        # Encode image once
        encoded_image = self.encode_image(image, cache_key)
        
        results = []
        
        # Process each task using the same encoded image
        for task in tasks:
            task_type = task['type']
            params = task.get('params', {})
            
            if task_type == 'caption':
                length = params.get('length', 'normal')
                result = self.model.caption(encoded_image, length=length)
            elif task_type == 'query':
                question = params.get('question', '')
                result = self.model.query(encoded_image, question)
            elif task_type == 'detect':
                object_name = params.get('object', '')
                result = self.model.detect(encoded_image, object_name)
            elif task_type == 'point':
                object_name = params.get('object', '')
                result = self.model.point(encoded_image, object_name)
            else:
                result = {'error': f'Unknown task type: {task_type}'}
                
            results.append(result)
            
        return results
    
    def clear_cache(self):
        """Clear the image encoding cache"""
        self.image_cache = {}
        
    def get_device(self):
        """Get the current device being used"""
        return self.model.device


# Example usage
if __name__ == "__main__":
    # Initialize optimized model
    moondream = OptimizedMoondream(
        model_path="/Users/samuelalaniz/dev/model.safetensors",
        use_smaller_model=True,
        max_crops=8,
        compile_model=True
    )
    
    # Process an image with multiple tasks efficiently
    results = moondream.process_image(
        "example.jpg",
        tasks=[
            {'type': 'caption', 'params': {'length': 'normal'}},
            {'type': 'query', 'params': {'question': 'What is shown in the image?'}},
            {'type': 'detect', 'params': {'object': 'person'}}
        ],
        cache_key="example_image"
    )
    
    # Print results
    for i, result in enumerate(results):
        print(f"Task {i+1} result:", result)