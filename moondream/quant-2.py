import torch
import numpy as np
from safetensors.torch import load_file, save_file
import time
import sys
from tqdm import tqdm

def quantize_tensor_to_int4(tensor):
    """Quantize a tensor to INT4 (4-bit) representation"""
    # Determine the range of values in the tensor
    min_val = tensor.min().item()
    max_val = tensor.max().item()
    
    # Calculate scaling factor for INT4 range [-8, 7]
    # INT4 range is -8 to 7 (16 values)
    scale = (max_val - min_val) / 15
    
    # Calculate zero point
    zero_point = round((-min_val / scale) - 8)
    zero_point = max(0, min(15, zero_point))  # Ensure it's in [0, 15]
    
    # Quantize
    quantized = torch.round((tensor / scale) + 8 + zero_point - 8).clamp(-8, 7).to(torch.int8)
    
    # Pack two INT4 values into one INT8
    # Even indices go to the lower 4 bits, odd indices to the upper 4 bits
    packed_shape = list(quantized.shape)
    if packed_shape[-1] % 2 == 1:
        # Pad with zeros if needed
        pad_size = list(quantized.shape)
        pad_size[-1] = 1
        quantized = torch.cat([quantized, torch.zeros(pad_size, dtype=torch.int8)], dim=-1)
        packed_shape[-1] = packed_shape[-1] + 1
    
    packed_shape[-1] = packed_shape[-1] // 2
    packed = torch.zeros(packed_shape, dtype=torch.int8)
    
    # Flatten for easier handling
    flat_quantized = quantized.reshape(-1)
    flat_packed = packed.reshape(-1)
    
    # Pack values - even indices to lower 4 bits, odd indices to upper 4 bits
    for i in range(len(flat_packed)):
        flat_packed[i] = (flat_quantized[i*2] & 0x0F) | ((flat_quantized[i*2+1] & 0x0F) << 4)
    
    # Store metadata for dequantization
    metadata = {
        "scale": scale,
        "zero_point": zero_point,
        "original_shape": tensor.shape
    }
    
    return packed, metadata

def quantize_model_to_int4(model_path, output_path):
    print(f"Starting quantization of model: {model_path}")
    print(f"Output will be saved to: {output_path}")
    
    start_time = time.time()
    
    # Load the model
    print("Loading model file...")
    model = load_file(model_path)
    
    load_time = time.time()
    print(f"Model loaded in {load_time - start_time:.2f} seconds")
    print(f"Model contains {len(model)} tensors")
    
    # Calculate total size of the model
    original_size_bytes = sum(tensor.numel() * tensor.element_size() for tensor in model.values())
    print(f"Original model size: {original_size_bytes / (1024 * 1024):.2f} MB")
    
    # Dict to store quantized tensors
    quantized_model = {}
    metadata_dict = {}
    
    # Count how many tensors will be quantized vs kept in original precision
    large_tensors = sum(1 for key, tensor in model.items() 
                      if tensor.numel() >= 1000 and not any(layer_type in key for layer_type in ["layernorm", "ln", "norm", "embedding"]))
    
    print(f"Tensors to be quantized: {large_tensors}")
    print(f"Tensors to keep in original precision: {len(model) - large_tensors}")
    
    # Quantize each tensor with progress bar
    print("Quantizing tensors...")
    quantized_count = 0
    skipped_count = 0
    
    progress_bar = tqdm(model.items(), total=len(model), desc="Quantizing")
    for key, tensor in progress_bar:
        # Skip small tensors or specifically exclude certain layers
        if tensor.numel() < 1000 or any(layer_type in key for layer_type in ["layernorm", "ln", "norm", "embedding"]):
            # Keep small tensors and layer norms in original precision
            quantized_model[key] = tensor
            skipped_count += 1
            progress_bar.set_postfix({"quantized": quantized_count, "skipped": skipped_count})
            continue
        
        # Quantize large tensors
        quantized_tensor, metadata = quantize_tensor_to_int4(tensor)
        quantized_model[key] = quantized_tensor
        metadata_dict[key] = metadata
        quantized_count += 1
        
        # Update progress bar
        progress_bar.set_postfix({"quantized": quantized_count, "skipped": skipped_count})
    
    quantize_time = time.time()
    print(f"Quantization completed in {quantize_time - load_time:.2f} seconds")
    
    # Calculate quantized size
    quantized_size_bytes = sum(tensor.numel() * tensor.element_size() for tensor in quantized_model.values())
    print(f"Quantized model size: {quantized_size_bytes / (1024 * 1024):.2f} MB")
    print(f"Size reduction: {(1 - quantized_size_bytes / original_size_bytes) * 100:.2f}%")
    
    # Save metadata separately
    print("Saving metadata...")
    torch.save(metadata_dict, output_path + ".metadata")
    
    # Save quantized model
    print("Saving quantized model...")
    save_file(quantized_model, output_path)
    
    end_time = time.time()
    print(f"Model saved. Total process time: {end_time - start_time:.2f} seconds")
    
    return quantized_model, metadata_dict


# Example usage
model_path = "/Users/samuelalaniz/dev/model.safetensors"
output_path = "/Users/samuelalaniz/dev/model-int4.safetensors"
quantized_model, metadata = quantize_model_to_int4(model_path, output_path)