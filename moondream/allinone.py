import torch
import os
import argparse
import json
from PIL import Image
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Literal, Any
from pathlib import Path
import math
import numpy as np
from tokenizers import Tokenizer
import pyvips
import time

# Configuration classes (based on the original moondream code)
@dataclass(frozen=True)
class TextConfig:
    dim: int = 2048
    ff_dim: int = 8192
    n_layers: int = 24
    vocab_size: int = 51200
    max_context: int = 2048
    n_heads: int = 32
    prefix_attn: int = 730


@dataclass(frozen=True)
class VisionConfig:
    enc_dim: int = 1152
    enc_patch_size: int = 14
    enc_n_layers: int = 27
    enc_ff_dim: int = 4304
    enc_n_heads: int = 16
    proj_out_dim: int = 2048
    crop_size: int = 378
    in_channels: int = 3
    max_crops: int = 12
    overlap_margin: int = 4
    proj_inner_dim: int = 8192


@dataclass(frozen=True)
class RegionConfig:
    dim: int = 2048
    coord_feat_dim: int = 256
    coord_out_dim: int = 1024
    size_feat_dim: int = 512
    size_out_dim: int = 2048
    inner_dim: int = 8192


@dataclass(frozen=True)
class TokenizerConfig:
    bos_id: int = 50256
    eos_id: int = 50256
    templates: Dict[str, Optional[Dict[str, List[int]]]] = field(
        default_factory=lambda: {
            "caption": {
                "short": [198, 198, 16438, 8305, 25],
                "normal": [198, 198, 24334, 1159, 25],
            },
            "query": {"prefix": [198, 198, 24361, 25], "suffix": [198, 198, 33706, 25]},
            "detect": {"prefix": [198, 198, 47504, 25], "suffix": [628]},
            "point": {"prefix": [198, 198, 12727, 25], "suffix": [628]},
        }
    )


@dataclass(frozen=True)
class MoondreamConfig:
    text: TextConfig = TextConfig()
    vision: VisionConfig = VisionConfig()
    region: RegionConfig = RegionConfig()
    tokenizer: TokenizerConfig = TokenizerConfig()

    @classmethod
    def from_dict(cls, config_dict: dict):
        text_config = TextConfig(**config_dict.get("text", {}))
        vision_config = VisionConfig(**config_dict.get("vision", {}))
        region_config = RegionConfig(**config_dict.get("region", {}))
        tokenizer_config = TokenizerConfig(**config_dict.get("tokenizer", {}))
        return cls(
            text=text_config,
            vision=vision_config,
            region=region_config,
            tokenizer=tokenizer_config,
        )

    def to_dict(self):
        return {
            "text": self.text.__dict__,
            "vision": self.vision.__dict__,
            "region": self.region.__dict__,
            "tokenizer": self.tokenizer.__dict__,
        }


# Layer implementation
@dataclass
class LinearWeights:
    weight: torch.Tensor
    bias: torch.Tensor


def linear(x: torch.Tensor, w: LinearWeights) -> torch.Tensor:
    return torch.nn.functional.linear(x, w.weight, w.bias)


@dataclass
class LayerNormWeights:
    weight: torch.Tensor
    bias: torch.Tensor


def layer_norm(x: torch.Tensor, w: LayerNormWeights) -> torch.Tensor:
    return torch.nn.functional.layer_norm(x, w.bias.shape, w.weight, w.bias)


@dataclass
class MLPWeights:
    fc1: LinearWeights
    fc2: LinearWeights
    act: Literal["gelu_approx"] = "gelu_approx"


def gelu_approx(x):
    return torch.nn.functional.gelu(x, approximate="tanh")


def mlp(x: torch.Tensor, w: MLPWeights) -> torch.Tensor:
    x = linear(x, w.fc1)
    x = gelu_approx(x)
    x = linear(x, w.fc2)
    return x


@dataclass
class AttentionWeights:
    qkv: LinearWeights
    proj: LinearWeights


def attn(x: torch.Tensor, w: AttentionWeights, n_heads: int) -> torch.Tensor:
    bsz, q_len, d_model = x.shape
    head_dim = d_model // n_heads

    q, k, v = [
        t.view(bsz, q_len, n_heads, head_dim).transpose(1, 2)
        for t in linear(x, w.qkv).chunk(3, dim=-1)
    ]
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    out = out.transpose(1, 2).reshape(bsz, q_len, d_model)
    out = linear(out, w.proj)
    return out


def precompute_freqs_cis(
    dim: int,
    end: int,
    theta: float = 10000.0,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=dtype)[: (dim // 2)] / dim))
    t = torch.arange(end, dtype=dtype).unsqueeze(1)
    freqs = t * freqs.unsqueeze(0)
    freqs = torch.exp(1j * freqs)
    return torch.stack([freqs.real, freqs.imag], dim=-1)


def apply_rotary_emb(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    position_ids: torch.Tensor,
    num_heads: int,
    rot_dim: int = 32,
    interleave: bool = False,
) -> torch.Tensor:
    assert rot_dim == freqs_cis.shape[-2] * 2
    assert num_heads == x.shape[1]

    x_rot, x_pass = x[..., :rot_dim], x[..., rot_dim:]

    if interleave:
        xq_r = x_rot.float().reshape(*x_rot.shape[:-1], -1, 2)[..., 0]
        xq_i = x_rot.float().reshape(*x_rot.shape[:-1], -1, 2)[..., 1]
    else:
        d_q = x_rot.shape[-1] // 2
        xq_r, xq_i = x_rot[..., :d_q], x_rot[..., d_q:]

    freqs_cos = freqs_cis[..., 0][position_ids, :].unsqueeze(0).unsqueeze(0)
    freqs_sin = freqs_cis[..., 1][position_ids, :].unsqueeze(0).unsqueeze(0)

    # Complex multiplication: (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xq_out = torch.stack((xq_out_r, xq_out_i), dim=-1).flatten(-2)

    return torch.cat([xq_out.to(x.dtype), x_pass], dim=-1)


# Helper functions for image processing
def select_tiling(
    height: int, width: int, crop_size: int, max_crops: int
) -> tuple[int, int]:
    """
    Determine the optimal number of tiles to cover an image with overlapping crops.
    """
    if height <= crop_size or width <= crop_size:
        return (1, 1)

    # Minimum required tiles in each dimension
    min_h = math.ceil(height / crop_size)
    min_w = math.ceil(width / crop_size)

    # If minimum required tiles exceed max_crops, return proportional distribution
    if min_h * min_w > max_crops:
        ratio = math.sqrt(max_crops / (min_h * min_w))
        return (max(1, math.floor(min_h * ratio)), max(1, math.floor(min_w * ratio)))

    # Perfect aspect-ratio tiles that satisfy max_crops
    h_tiles = math.floor(math.sqrt(max_crops * height / width))
    w_tiles = math.floor(math.sqrt(max_crops * width / height))

    # Ensure we meet minimum tile requirements
    h_tiles = max(h_tiles, min_h)
    w_tiles = max(w_tiles, min_w)

    # If we exceeded max_crops, scale down the larger dimension
    if h_tiles * w_tiles > max_crops:
        if w_tiles > h_tiles:
            w_tiles = math.floor(max_crops / h_tiles)
        else:
            h_tiles = math.floor(max_crops / w_tiles)

    return (max(1, h_tiles), max(1, w_tiles))


def overlap_crop_image(
    image: np.ndarray,
    overlap_margin: int,
    max_crops: int,
    base_size: tuple[int, int] = (378, 378),
    patch_size: int = 14,
) -> Dict:
    """
    Process an image using an overlap-and-resize cropping strategy with margin handling.
    """
    original_h, original_w = image.shape[:2]

    # Convert margin from patch units to pixels
    margin_pixels = patch_size * overlap_margin
    total_margin_pixels = margin_pixels * 2  # Both sides

    # Calculate crop parameters
    crop_patches = base_size[0] // patch_size  # patches per crop dimension
    crop_window_patches = crop_patches - (2 * overlap_margin)  # usable patches
    crop_window_size = crop_window_patches * patch_size  # usable size in pixels

    # Determine tiling
    tiling = select_tiling(
        original_h - total_margin_pixels,
        original_w - total_margin_pixels,
        crop_window_size,
        max_crops,
    )

    # Pre-allocate crops.
    n_crops = tiling[0] * tiling[1] + 1  # 1 = global crop
    crops = np.zeros(
        (n_crops, base_size[0], base_size[1], image.shape[2]), dtype=np.uint8
    )

    # Resize image to fit tiling
    target_size = (
        tiling[0] * crop_window_size + total_margin_pixels,
        tiling[1] * crop_window_size + total_margin_pixels,
    )

    # Convert to vips for resizing
    vips_image = pyvips.Image.new_from_array(image)
    scale_x = target_size[1] / image.shape[1]
    scale_y = target_size[0] / image.shape[0]
    resized = vips_image.resize(scale_x, vscale=scale_y)
    image = resized.numpy()

    # Create global crop
    scale_x = base_size[1] / vips_image.width
    scale_y = base_size[0] / vips_image.height
    global_vips = vips_image.resize(scale_x, vscale=scale_y)
    crops[0] = global_vips.numpy()

    for i in range(tiling[0]):
        for j in range(tiling[1]):
            # Calculate crop coordinates
            y0 = i * crop_window_size
            x0 = j * crop_window_size

            # Extract crop with padding if needed
            y_end = min(y0 + base_size[0], image.shape[0])
            x_end = min(x0 + base_size[1], image.shape[1])

            crop_region = image[y0:y_end, x0:x_end]
            crops[
                1 + i * tiling[1] + j, : crop_region.shape[0], : crop_region.shape[1]
            ] = crop_region

    return {"crops": crops, "tiling": tiling}


def reconstruct_from_crops(
    crops: torch.Tensor,
    tiling: tuple[int, int],
    overlap_margin: int,
    patch_size: int = 14,
) -> torch.Tensor:
    """
    Reconstruct the original image from overlapping crops into a single seamless image.
    """
    tiling_h, tiling_w = tiling
    crop_height, crop_width = crops[0].shape[:2]
    margin_pixels = overlap_margin * patch_size

    # Calculate output size (only adding margins once)
    output_h = (crop_height - 2 * margin_pixels) * tiling_h + 2 * margin_pixels
    output_w = (crop_width - 2 * margin_pixels) * tiling_w + 2 * margin_pixels

    reconstructed = torch.zeros(
        (output_h, output_w, crops[0].shape[2]),
        device=crops[0].device,
        dtype=crops[0].dtype,
    )

    for i, crop in enumerate(crops):
        tile_y = i // tiling_w
        tile_x = i % tiling_w

        # For each tile, determine which part to keep
        # Keep left margin only for first column
        x_start = 0 if tile_x == 0 else margin_pixels
        # Keep right margin only for last column
        x_end = crop_width if tile_x == tiling_w - 1 else crop_width - margin_pixels
        # Keep top margin only for first row
        y_start = 0 if tile_y == 0 else margin_pixels
        # Keep bottom margin only for last row
        y_end = crop_height if tile_y == tiling_h - 1 else crop_height - margin_pixels

        # Calculate where this piece belongs in the output
        out_x = tile_x * (crop_width - 2 * margin_pixels)
        out_y = tile_y * (crop_height - 2 * margin_pixels)

        # Place the piece
        reconstructed[
            out_y + y_start : out_y + y_end, out_x + x_start : out_x + x_end
        ] = crop[y_start:y_end, x_start:x_end]

    return reconstructed


def create_patches(x, patch_size):
    # Original shape: [B, C, H, W]
    B, C, H, W = x.shape
    P1 = P2 = patch_size

    # Step 1: Split H and W dimensions into patches
    # [B, C, H/P1, P1, W/P2, P2]
    x = x.reshape(B, C, H // P1, P1, W // P2, P2)

    # Step 2: Rearrange dimensions to match target shape
    # [B, H/P1, W/P2, C, P1, P2]
    x = x.permute(0, 2, 4, 1, 3, 5)

    # Step 3: Combine dimensions to get final shape
    # [B, (H/P1)*(W/P2), C*P1*P2]
    x = x.reshape(B, (H // P1) * (W // P2), C * P1 * P2)

    return x


# Int4 quantization functions
def quantize_to_int4(tensor):
    """
    Quantize a tensor to int4 format (0-15 range).
    """
    # Store original shape
    original_shape = tensor.shape
    
    # Flatten tensor
    tensor_flat = tensor.reshape(-1)
    
    # Find min and max values
    min_val = tensor_flat.min().item()
    max_val = tensor_flat.max().item()
    
    # Calculate scale for 4-bit range (0-15)
    scale = (max_val - min_val) / 15
    
    # Quantize to int4 range (0-15)
    quantized = torch.round((tensor_flat - min_val) / scale).clamp(0, 15).to(torch.uint8)
    
    return {
        "quantized": quantized,
        "scale": scale,
        "min_val": min_val,
        "shape": original_shape,
    }


def dequantize_from_int4(q_data, device="cpu"):
    """
    Dequantize a tensor from int4 format back to floating point.
    """
    # Extract quantization parameters
    quantized = q_data["quantized"].to(device)
    scale = q_data["scale"]
    min_val = q_data["min_val"]
    original_shape = q_data["shape"]
    
    # Dequantize
    dequantized = (quantized.float() * scale + min_val).to(torch.float16)
    
    # Reshape to original dimensions
    return dequantized.reshape(original_shape)


class KVCache:
    def __init__(self, n_heads, max_context, dim, device, dtype):
        self.k_cache = torch.zeros(
            1, n_heads, max_context, dim // n_heads, 
            device=device, dtype=dtype
        )
        self.v_cache = torch.zeros(
            1, n_heads, max_context, dim // n_heads,
            device=device, dtype=dtype
        )

    def update(self, pos_ids, k, v):
        kout, vout = self.k_cache, self.v_cache
        kout[:, :, pos_ids, :] = k
        vout[:, :, pos_ids, :] = v
        return kout, vout


@dataclass(frozen=True)
class EncodedImage:
    pos: int
    caches: List[Tuple[torch.Tensor, torch.Tensor]]


class Int4QuantizedMoondream:
    def __init__(self, model_path, config_path=None, device=None):
        """
        Initialize the int4 quantized Moondream model.
        
        Args:
            model_path: Path to the original model weights or already quantized weights
            config_path: Path to config file (optional)
            device: Device to run on ("mps" for Apple Silicon, "cpu", or "cuda")
        """
        # Determine device
        if device is None:
            if torch.backends.mps.is_available():
                self.device = "mps"
                print("Using MPS for Apple Silicon")
            elif torch.cuda.is_available():
                self.device = "cuda"
                print("Using CUDA")
            else:
                self.device = "cpu"
                print("Using CPU")
        else:
            self.device = device
            
        # Load config
        if config_path:
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
                self.config = MoondreamConfig.from_dict(config_dict)
        else:
            self.config = MoondreamConfig()
            
        # Load tokenizer
        self.tokenizer = Tokenizer.from_pretrained("vikhyatk/moondream2")
        
        # Check if model is already quantized
        if model_path.endswith(".int4.pt"):
            self.quantized_weights = torch.load(model_path, map_location="cpu")
            print(f"Loaded quantized model from {model_path}")
        else:
            # Quantize weights if needed
            self.quantized_weights = self._quantize_model(model_path)
            quantized_path = f"{os.path.splitext(model_path)[0]}.int4.pt"
            torch.save(self.quantized_weights, quantized_path)
            print(f"Model quantized and saved to {quantized_path}")
        
        # Initialize model components
        self._setup_model()
        
    def _quantize_model(self, model_path):
        """
        Quantize model weights to int4.
        """
        print(f"Quantizing model from {model_path}...")
        
        # Load model weights
        if model_path.endswith(".safetensors"):
            from safetensors.torch import load_file
            weights = load_file(model_path)
        else:
            weights = torch.load(model_path, map_location="cpu")
        
        # Quantize weights
        quantized_weights = {}
        for name, tensor in weights.items():
            if name.endswith(".weight") and tensor.ndim > 1:  # Only quantize weight matrices
                quantized_weights[name] = quantize_to_int4(tensor)
            else:
                quantized_weights[name] = tensor
                
        return quantized_weights
    
    def _get_tensor(self, name):
        """Get a tensor from the quantized weights, dequantizing if needed."""
        if name in self.quantized_weights:
            if isinstance(self.quantized_weights[name], dict) and "quantized" in self.quantized_weights[name]:
                # This is a quantized tensor
                return dequantize_from_int4(self.quantized_weights[name], self.device)
            else:
                # This is a non-quantized tensor
                return self.quantized_weights[name].to(self.device)
        else:
            print(f"Warning: Tensor {name} not found in quantized weights")
            return None
    
    def _setup_model(self):
        """
        Initialize model structure with tensors from quantized weights.
        """
        # Set up vision encoder
        self.vision = self._setup_vision_encoder()
        
        # Set up text model
        self.text = self._setup_text_model()
        
        # Set up region model
        self.region = self._setup_region_model()
        
        # Set up attention mask
        attn_mask = torch.tril(
            torch.ones(
                1, 1, self.config.text.max_context, self.config.text.max_context, 
                dtype=torch.bool, device=self.device
            )
        )
        patch_w = self.config.vision.crop_size // self.config.vision.enc_patch_size
        prefix_attn_len = 1 + patch_w**2
        attn_mask[..., :prefix_attn_len, :prefix_attn_len] = 1
        self.attn_mask = attn_mask
        
        # Setup caches for KV attention to improve generation speed
        self._setup_kv_caches()
    
    def _setup_vision_encoder(self):
        """Set up the vision encoder component."""
        vision = torch.nn.ModuleDict({
            "blocks": torch.nn.ModuleList([
                torch.nn.ModuleDict({
                    "ln1": torch.nn.LayerNorm(self.config.vision.enc_dim),
                    "attn": torch.nn.ModuleDict({
                        "qkv": torch.nn.Linear(
                            self.config.vision.enc_dim, 3 * self.config.vision.enc_dim
                        ),
                        "proj": torch.nn.Linear(
                            self.config.vision.enc_dim, self.config.vision.enc_dim
                        ),
                    }),
                    "ln2": torch.nn.LayerNorm(self.config.vision.enc_dim),
                    "mlp": torch.nn.ModuleDict({
                        "fc1": torch.nn.Linear(
                            self.config.vision.enc_dim, self.config.vision.enc_ff_dim
                        ),
                        "fc2": torch.nn.Linear(
                            self.config.vision.enc_ff_dim, self.config.vision.enc_dim
                        ),
                    }),
                })
                for _ in range(self.config.vision.enc_n_layers)
            ]),
            "post_ln": torch.nn.LayerNorm(self.config.vision.enc_dim),
            "patch_emb": torch.nn.Linear(
                self.config.vision.enc_patch_size * self.config.vision.enc_patch_size * self.config.vision.in_channels,
                self.config.vision.enc_dim
            ),
            "proj_mlp": torch.nn.ModuleDict({
                "fc1": torch.nn.Linear(
                    self.config.vision.enc_dim * 2, self.config.vision.proj_inner_dim
                ),
                "fc2": torch.nn.Linear(
                    self.config.vision.proj_inner_dim, self.config.vision.proj_out_dim
                ),
            }),
        })
        
        # Load quantized weights for vision encoder
        for i in range(self.config.vision.enc_n_layers):
            prefix = f"vision_encoder.encoder.model.visual.blocks.{i}"
            
            # Attention weights
            vision["blocks"][i]["attn"]["qkv"].weight.data = self._get_tensor(f"{prefix}.attn.qkv.weight")
            vision["blocks"][i]["attn"]["qkv"].bias.data = self._get_tensor(f"{prefix}.attn.qkv.bias")
            vision["blocks"][i]["attn"]["proj"].weight.data = self._get_tensor(f"{prefix}.attn.proj.weight")
            vision["blocks"][i]["attn"]["proj"].bias.data = self._get_tensor(f"{prefix}.attn.proj.bias")
            
            # LayerNorm weights
            vision["blocks"][i]["ln1"].weight.data = self._get_tensor(f"{prefix}.norm1.weight")
            vision["blocks"][i]["ln1"].bias.data = self._get_tensor(f"{prefix}.norm1.bias")
            vision["blocks"][i]["ln2"].weight.data = self._get_tensor(f"{prefix}.norm2.weight")
            vision["blocks"][i]["ln2"].bias.data = self._get_tensor(f"{prefix}.norm2.bias")
            
            # MLP weights
            vision["blocks"][i]["mlp"]["fc1"].weight.data = self._get_tensor(f"{prefix}.mlp.fc1.weight")
            vision["blocks"][i]["mlp"]["fc1"].bias.data = self._get_tensor(f"{prefix}.mlp.fc1.bias")
            vision["blocks"][i]["mlp"]["fc2"].weight.data = self._get_tensor(f"{prefix}.mlp.fc2.weight")
            vision["blocks"][i]["mlp"]["fc2"].bias.data = self._get_tensor(f"{prefix}.mlp.fc2.bias")
        
        # Patch embedding and LayerNorm weights
        vision["patch_emb"].weight.data = self._get_tensor("vision_encoder.encoder.model.visual.patch_embed.linear.weight")
        vision["patch_emb"].bias.data = self._get_tensor("vision_encoder.encoder.model.visual.patch_embed.linear.bias")
        vision["post_ln"].weight.data = self._get_tensor("vision_encoder.encoder.model.visual.norm.weight")
        vision["post_ln"].bias.data = self._get_tensor("vision_encoder.encoder.model.visual.norm.bias")
        
        # Projection MLP weights
        vision["proj_mlp"]["fc1"].weight.data = self._get_tensor("vision_encoder.projection.mlp.fc1.weight")
        vision["proj_mlp"]["fc1"].bias.data = self._get_tensor("vision_encoder.projection.mlp.fc1.bias")
        vision["proj_mlp"]["fc2"].weight.data = self._get_tensor("vision_encoder.projection.mlp.fc2.weight")
        vision["proj_mlp"]["fc2"].bias.data = self._get_tensor("vision_encoder.projection.mlp.fc2.bias")
        
        # Position embedding
        grid_size = self.config.vision.crop_size // self.config.vision.enc_patch_size
        num_patches = grid_size * grid_size
        vision.pos_emb = torch.nn.Parameter(
            self._get_tensor("vision_encoder.encoder.model.visual.pos_embed")
        )
        
        return vision.to(self.device)
    
    def _setup_text_model(self):
        """Set up the text model component."""
        # Create text model structure
        text = torch.nn.ModuleDict({
            "blocks": torch.nn.ModuleList([
                torch.nn.ModuleDict({
                    "ln": torch.nn.LayerNorm(self.config.text.dim),
                    "attn": torch.nn.ModuleDict({
                        "qkv": torch.nn.Linear(
                            self.config.text.dim, 3 * self.config.text.dim
                        ),
                        "proj": torch.nn.Linear(
                            self.config.text.dim, self.config.text.dim
                        ),
                    }),
                    "mlp": torch.nn.ModuleDict({
                        "fc1": torch.nn.Linear(
                            self.config.text.dim, self.config.text.ff_dim
                        ),
                        "fc2": torch.nn.Linear(
                            self.config.text.ff_dim, self.config.text.dim
                        ),
                    }),
                })
                for _ in range(self.config.text.n_layers)
            ]),
            "post_ln": torch.nn.LayerNorm(self.config.text.dim),
            "lm_head": torch.nn.Linear(self.config.text.dim, self.config.text.vocab_size),
        })
        
        # Word token embeddings
        text.wte = torch.nn.Parameter(self._get_tensor("text_model.transformer.embd.wte.weight"))
        
        # RoPE embeddings
        text.freqs_cis = precompute_freqs_cis(
            self.config.text.dim // (2 * self.config.text.n_heads), 
            self.config.text.max_context
        ).to(self.device)
        
        # Load text model weights
        for i in range(self.config.text.n_layers):
            prefix = f"text_model.transformer.h.{i}"
            
            # Attention weights
            text["blocks"][i]["attn"]["qkv"].weight.data = self._get_tensor(f"{prefix}.mixer.Wqkv.weight")
            text["blocks"][i]["attn"]["qkv"].bias.data = self._get_tensor(f"{prefix}.mixer.Wqkv.bias")
            text["blocks"][i]["attn"]["proj"].weight.data = self._get_tensor(f"{prefix}.mixer.out_proj.weight")
            text["blocks"][i]["attn"]["proj"].bias.data = self._get_tensor(f"{prefix}.mixer.out_proj.bias")
            
            # LayerNorm weights
            text["blocks"][i]["ln"].weight.data = self._get_tensor(f"{prefix}.ln.weight")
            text["blocks"][i]["ln"].bias.data = self._get_tensor(f"{prefix}.ln.bias")
            
            # MLP weights
            text["blocks"][i]["mlp"]["fc1"].weight.data = self._get_tensor(f"{prefix}.mlp.fc1.weight")
            text["blocks"][i]["mlp"]["fc1"].bias.data = self._get_tensor(f"{prefix}.mlp.fc1.bias")
            text["blocks"][i]["mlp"]["fc2"].weight.data = self._get_tensor(f"{prefix}.mlp.fc2.weight")
            text["blocks"][i]["mlp"]["fc2"].bias.data = self._get_tensor(f"{prefix}.mlp.fc2.bias")
        
        # Post LayerNorm weights
        text["post_ln"].weight.data = self._get_tensor("text_model.lm_head.ln.weight")
        text["post_ln"].bias.data = self._get_tensor("text_model.lm_head.ln.bias")
        
        # Language model head weights
        text["lm_head"].weight.data = self._get_tensor("text_model.lm_head.linear.weight")
        text["lm_head"].bias.data = self._get_tensor("text_model.lm_head.linear.bias")
        
        return text.to(self.device)
    
    def _setup_region_model(self):
        """Set up the region model component."""
        # Create region model structure
        region = torch.nn.ModuleDict({
            "coord_encoder": torch.nn.Linear(
                self.config.region.coord_feat_dim, self.config.region.dim
            ),
            "coord_decoder": torch.nn.ModuleDict({
                "fc1": torch.nn.Linear(
                    self.config.region.dim, self.config.region.inner_dim
                ),
                "fc2": torch.nn.Linear(
                    self.config.region.inner_dim, self.config.region.coord_out_dim
                ),
            }),
            "size_encoder": torch.nn.Linear(
                self.config.region.size_feat_dim, self.config.region.dim
            ),
            "size_decoder": torch.nn.ModuleDict({
                "fc1": torch.nn.Linear(
                    self.config.region.dim, self.config.region.inner_dim
                ),
                "fc2": torch.nn.Linear(
                    self.config.region.inner_dim, self.config.region.size_out_dim
                ),
            }),
        })
        
        # Load region model weights
        region["coord_encoder"].weight.data = self._get_tensor("region_model.coordinate_encoder.weight")
        region["coord_encoder"].bias.data = self._get_tensor("region_model.coordinate_encoder.bias")
        
        region["coord_decoder"]["fc1"].weight.data = self._get_tensor("region_model.coordinate_decoder.fc1.weight")
        region["coord_decoder"]["fc1"].bias.data = self._get_tensor("region_model.coordinate_decoder.fc1.bias")
        region["coord_decoder"]["fc2"].weight.data = self._get_tensor("region_model.coordinate_decoder.fc2.weight")
        region["coord_decoder"]["fc2"].bias.data = self._get_tensor("region_model.coordinate_decoder.fc2.bias")
        
        region["size_encoder"].weight.data = self._get_tensor("region_model.size_encoder.weight")
        region["size_encoder"].bias.data = self._get_tensor("region_model.size_encoder.bias")
        
        region["size_decoder"]["fc1"].weight.data = self._get_tensor("region_model.size_decoder.fc1.weight")
        region["size_decoder"]["fc1"].bias.data = self._get_tensor("region_model.size_decoder.fc1.bias")
        region["size_decoder"]["fc2"].weight.data = self._get_tensor("region_model.size_decoder.fc2.weight")
        region["size_decoder"]["fc2"].bias.data = self._get_tensor("region_model.size_decoder.fc2.bias")
        
        # Feature embeddings
        region.coord_features = torch.nn.Parameter(
            self._get_tensor("region_model.coordinate_features.weight").T
        )
        region.size_features = torch.nn.Parameter(
            self._get_tensor("region_model.size_features.weight").T
        )
        
        return region.to(self.device)
    
    def _setup_kv_caches(self):
        """Set up key-value caches for efficient text generation."""
        for i, block in enumerate(self.text["blocks"]):
            block.kv_cache = KVCache(
                self.config.text.n_heads,
                self.config.text.max_context,
                self.config.text.dim,
                device=self.device,
                dtype=torch.float16
            )
    
    def _run_vision_encoder(self, image):
        """Process image through vision encoder."""
        all_crops, tiling = self._prepare_crops(image)
        
        # Process each crop through vision encoder
        outputs = []
        for crop in all_crops:
            # Create patches from the crop
            x = create_patches(crop, self.config.vision.enc_patch_size)
            
            # Pass through patch embedding
            x = torch.nn.functional.linear(x, self.vision["patch_emb"].weight, self.vision["patch_emb"].bias)
            
            # Add position embeddings
            x = x + self.vision.pos_emb
            
            # Pass through transformer blocks
            for block in self.vision["blocks"]:
                # Attention
                x_ln1 = torch.nn.functional.layer_norm(x, (self.config.vision.enc_dim,), block["ln1"].weight, block["ln1"].bias)
                x_attn = attn(
                    x_ln1, 
                    AttentionWeights(
                        LinearWeights(block["attn"]["qkv"].weight, block["attn"]["qkv"].bias),
                        LinearWeights(block["attn"]["proj"].weight, block["attn"]["proj"].bias)
                    ), 
                    self.config.vision.enc_n_heads
                )
                
                # MLP
                x_ln2 = torch.nn.functional.layer_norm(x, (self.config.vision.enc_dim,), block["ln2"].weight, block["ln2"].bias)
                x_mlp = mlp(
                    x_ln2,
                    MLPWeights(
                        LinearWeights(block["mlp"]["fc1"].weight, block["mlp"]["fc1"].bias),
                        LinearWeights(block["mlp"]["fc2"].weight, block["mlp"]["fc2"].bias)
                    )
                )
                
                # Residual connections
                x = x + x_attn + x_mlp
            
            # Final layer norm
            x = torch.nn.functional.layer_norm(x, (self.config.vision.enc_dim,), self.vision["post_ln"].weight, self.vision["post_ln"].bias)
            outputs.append(x)
        
        # Extract global and local features
        global_features = outputs[0]
        local_features = torch.stack(outputs[1:])
        
        # Reshape local features
        local_features = local_features.view(
            -1,
            self.config.vision.enc_n_layers,
            self.config.vision.enc_n_layers,
            self.config.vision.enc_dim,
        )
        
        # Reconstruct features
        reconstructed = reconstruct_from_crops(
            local_features,
            tiling,
            patch_size=1,
            overlap_margin=self.config.vision.overlap_margin,
        )
        
        # Project features to text dimension
        final_features = torch.cat([global_features, reconstructed.view(729, self.config.vision.enc_dim)], dim=-1)
        projected_features = mlp(
            final_features,
            MLPWeights(
                LinearWeights(self.vision["proj_mlp"]["fc1"].weight, self.vision["proj_mlp"]["fc1"].bias),
                LinearWeights(self.vision["proj_mlp"]["fc2"].weight, self.vision["proj_mlp"]["fc2"].bias)
            )
        )
        
        return projected_features
    
    def _prepare_crops(self, image):
        """Prepare image crops for vision processing."""
        if isinstance(image, Image.Image):
            np_image = np.array(image.convert("RGB"))
        else:
            np_image = np.array(image)
            
        # Create overlapping crops
        overlap_crops = overlap_crop_image(
            np_image, 
            max_crops=self.config.vision.max_crops, 
            overlap_margin=self.config.vision.overlap_margin,
            base_size=(self.config.vision.crop_size, self.config.vision.crop_size),
            patch_size=self.config.vision.enc_patch_size
        )
        
        all_crops = overlap_crops["crops"]
        all_crops = np.transpose(all_crops, (0, 3, 1, 2))
        all_crops = torch.from_numpy(all_crops).to(device=self.device, dtype=torch.float16)
        all_crops = all_crops / 255.0  # Normalize to [0, 1]
        all_crops = (all_crops - 0.5) / 0.5  # Normalize to [-1, 1]
        
        return all_crops, overlap_crops["tiling"]
    
    def text_encoder(self, input_ids):
        """Encode token IDs to embeddings."""
        return torch.nn.functional.embedding(input_ids, self.text.wte)
    
    def encode_image(self, image):
        """Encode image and run through initial text prefill."""
        with torch.inference_mode():
            # Process image through vision encoder
            img_emb = self._run_vision_encoder(image)
            
            # Get BOS token embedding
            bos_emb = self.text_encoder(
                torch.tensor([[self.config.tokenizer.bos_id]], device=self.device)
            )
            
            # Combine embeddings
            inputs_embeds = torch.cat([bos_emb, img_emb[None]], dim=1)
            
            # Prefill KV cache
            mask = self.attn_mask[:, :, 0 : inputs_embeds.size(1), :]
            pos_ids = torch.arange(inputs_embeds.size(1), device=self.device, dtype=torch.long)
            hidden = self._prefill(inputs_embeds, mask, pos_ids)
            
            # Store KV cache state
            caches = [
                (
                    block.kv_cache.k_cache[:, :, : inputs_embeds.size(1), :].clone(),
                    block.kv_cache.v_cache[:, :, : inputs_embeds.size(1), :].clone(),
                )
                for block in self.text["blocks"]
            ]
            
            return EncodedImage(
                pos=inputs_embeds.size(1),
                caches=caches
            )
    
    def load_encoded_image(self, encoded_image):
        """Load pre-encoded image into KV caches."""
        for block, (k, v) in zip(self.text["blocks"], encoded_image.caches):
            block.kv_cache.k_cache[:, :, : k.size(2), :] = k
            block.kv_cache.v_cache[:, :, : v.size(2), :] = v
    
    def _prefill(self, x, attn_mask, pos_ids):
        """Process input through text model with attention mask."""
        for block in self.text["blocks"]:
            # Layer normalization
            ln_out = torch.nn.functional.layer_norm(
                x, (self.config.text.dim,), block["ln"].weight, block["ln"].bias
            )
            
            # Self-attention
            bsz, q_len, _ = ln_out.shape
            head_dim = self.config.text.dim // self.config.text.n_heads
            
            # QKV projection
            qkv = torch.nn.functional.linear(ln_out, block["attn"]["qkv"].weight, block["attn"]["qkv"].bias)
            q, k, v = [
                t.view(bsz, q_len, self.config.text.n_heads, head_dim).transpose(1, 2)
                for t in qkv.chunk(3, dim=-1)
            ]
            
            # Apply rotary embeddings
            q = apply_rotary_emb(q, self.text.freqs_cis, pos_ids, self.config.text.n_heads)
            k = apply_rotary_emb(k, self.text.freqs_cis, pos_ids, self.config.text.n_heads)
            
            # Update KV cache
            k, v = block.kv_cache.update(pos_ids, k, v)
            
            # Attention
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask
            ).to(x.dtype)
            
            # Reshape attention output
            attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, self.config.text.dim)
            
            # Projection
            attn_output = torch.nn.functional.linear(
                attn_output, block["attn"]["proj"].weight, block["attn"]["proj"].bias
            )
            
            # First residual connection
            x = x + attn_output
            
            # MLP
            mlp_output = mlp(
                ln_out,
                MLPWeights(
                    LinearWeights(block["mlp"]["fc1"].weight, block["mlp"]["fc1"].bias),
                    LinearWeights(block["mlp"]["fc2"].weight, block["mlp"]["fc2"].bias)
                )
            )
            
            # Second residual connection
            x = x + mlp_output
            
        return x
    
    def lm_head(self, hidden):
        """Apply layer norm and project to vocab size."""
        hidden = torch.nn.functional.layer_norm(
            hidden, (self.config.text.dim,), self.text["post_ln"].weight, self.text["post_ln"].bias
        )
        return torch.nn.functional.linear(hidden, self.text["lm_head"].weight, self.text["lm_head"].bias)
    
    def _decode_one_token(self, token_embedding, masked_pos, pos_id):
        """Decode a single token using cached key-values."""
        with torch.inference_mode():
            # Initialize mask and position ID
            mask = torch.zeros(1, 1, self.config.text.max_context, device=self.device, dtype=torch.bool)
            mask[:, :, :masked_pos] = 1
            pos_id_tensor = torch.tensor([pos_id], device=self.device, dtype=torch.long)
            
            # Forward through transformers blocks
            hidden = token_embedding
            
            for block in self.text["blocks"]:
                # Layer normalization
                ln_out = torch.nn.functional.layer_norm(
                    hidden, (self.config.text.dim,), block["ln"].weight, block["ln"].bias
                )
                
                # Self-attention
                bsz, q_len, _ = ln_out.shape
                head_dim = self.config.text.dim // self.config.text.n_heads
                
                # QKV projection
                qkv = torch.nn.functional.linear(ln_out, block["attn"]["qkv"].weight, block["attn"]["qkv"].bias)
                q, k, v = [
                    t.view(bsz, q_len, self.config.text.n_heads, head_dim).transpose(1, 2)
                    for t in qkv.chunk(3, dim=-1)
                ]
                
                # Apply rotary embeddings
                q = apply_rotary_emb(q, self.text.freqs_cis, pos_id_tensor, self.config.text.n_heads)
                k = apply_rotary_emb(k, self.text.freqs_cis, pos_id_tensor, self.config.text.n_heads)
                
                # Update KV cache 
                k, v = block.kv_cache.update(pos_id_tensor, k, v)
                
                # Attention
                attn_output = torch.nn.functional.scaled_dot_product_attention(
                    q, k[:, :, :masked_pos+1], v[:, :, :masked_pos+1]
                ).to(hidden.dtype)
                
                # Reshape attention output
                attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, self.config.text.dim)
                
                # Projection
                attn_output = torch.nn.functional.linear(
                    attn_output, block["attn"]["proj"].weight, block["attn"]["proj"].bias
                )
                
                # First residual connection
                hidden = hidden + attn_output
                
                # MLP
                mlp_output = mlp(
                    ln_out,
                    MLPWeights(
                        LinearWeights(block["mlp"]["fc1"].weight, block["mlp"]["fc1"].bias),
                        LinearWeights(block["mlp"]["fc2"].weight, block["mlp"]["fc2"].bias)
                    )
                )
                
                # Second residual connection
                hidden = hidden + mlp_output
            
            # Apply language model head to get logits
            logits = self.lm_head(hidden[:, -1:])
            
            # Get next token
            next_token = torch.argmax(logits, dim=-1)
            
            return next_token
    
    def _prefill_prompt(self, prompt_tokens, pos):
        """Run the prompt through the model and get the initial token."""
        with torch.inference_mode():
            # Get embeddings
            prompt_emb = self.text_encoder(prompt_tokens)
            
            # Run through model
            mask = self.attn_mask[:, :, pos : pos + prompt_emb.size(1), :]
            pos_ids = torch.arange(pos, pos + prompt_emb.size(1), device=self.device, dtype=torch.long)
            hidden = self._prefill(prompt_emb, mask, pos_ids)
            
            # Get next token
            logits = self.lm_head(hidden)
            next_token = torch.argmax(logits[:, -1:], dim=-1)
            
            return next_token, pos + prompt_emb.size(1)
    
    def _generate_text(self, prompt_tokens, image_pos, max_tokens=512):
        """Generate text based on prompt and image."""
        # Prefill prompt
        next_token, pos = self._prefill_prompt(prompt_tokens, image_pos)
        
        # Generate tokens
        generated_tokens = 0
        output_ids = []
        
        while next_token.item() != self.config.tokenizer.eos_id and generated_tokens < max_tokens:
            output_ids.append(next_token.item())
            yield self.tokenizer.decode([next_token.item()])
            
            # Encode the next token
            token_emb = self.text_encoder(next_token)
            
            # Get next token
            next_token = self._decode_one_token(token_emb, pos, pos)
            pos += 1
            generated_tokens += 1
        
        if next_token.item() == self.config.tokenizer.eos_id:
            output_ids.append(next_token.item())
            yield self.tokenizer.decode([next_token.item()])
    
    def query(self, image, question, stream=False, max_tokens=512):
        """Ask a question about an image."""
        if self.config.tokenizer.templates["query"] is None:
            raise NotImplementedError("Model does not support querying.")
        
        # Encode image
        encoded_image = self.encode_image(image)
        
        # Load encoded image into KV cache
        self.load_encoded_image(encoded_image)
        
        # Create prompt tokens
        prompt_tokens = torch.tensor(
            [
                self.config.tokenizer.templates["query"]["prefix"]
                + self.tokenizer.encode(question).ids
                + self.config.tokenizer.templates["query"]["suffix"]
            ],
            device=self.device,
        )
        
        # Generate answer
        def generator():
            for token in self._generate_text(prompt_tokens, encoded_image.pos, max_tokens):
                yield token
        
        if stream:
            return {"answer": generator()}
        else:
            return {"answer": "".join(list(generator()))}
    
    def caption(self, image, length="normal", stream=False, max_tokens=512):
        """Generate a caption for an image."""
        if self.config.tokenizer.templates["caption"] is None:
            raise NotImplementedError("Model does not support captioning.")
        
        if length not in self.config.tokenizer.templates["caption"]:
            raise ValueError(f"Model does not support caption length '{length}'.")
        
        # Encode image
        encoded_image = self.encode_image(image)
        
        # Load encoded image into KV cache
        self.load_encoded_image(encoded_image)
        
        # Create prompt tokens
        prompt_tokens = torch.tensor(
            [self.config.tokenizer.templates["caption"][length]],
            device=self.device,
        )
        
        # Generate caption
        def generator():
            for token in self._generate_text(prompt_tokens, encoded_image.pos, max_tokens):
                yield token
        
        if stream:
            return {"caption": generator()}
        else:
            return {"caption": "".join(list(generator()))}


def main():
    parser = argparse.ArgumentParser(description="Run Moondream model in int4 quantization")
    parser.add_argument("--model", "-m", type=str, required=True, help="Path to model weights")
    parser.add_argument("--config", "-c", type=str, default=None, help="Path to config file")
    parser.add_argument("--image", "-i", type=str, required=True, help="Path to image file")
    parser.add_argument("--question", "-q", type=str, default=None, help="Question to ask about the image")
    parser.add_argument("--caption", "-p", action="store_true", help="Generate captions for the image")
    parser.add_argument("--device", "-d", type=str, default=None, help="Device to run on (mps, cuda, cpu)")
    parser.add_argument("--stream", "-s", action="store_true", help="Stream output tokens")
    args = parser.parse_args()
    
    print(f"Loading model from {args.model}...")
    model = Int4QuantizedMoondream(
        model_path=args.model,
        config_path=args.config,
        device=args.device
    )
    
    print(f"Loading image from {args.image}...")
    image = Image.open(args.image)
    
    if args.caption:
        print("\nGenerating caption:")
        for length in ["short", "normal"]:
            print(f"\n{length.capitalize()} caption:")
            caption_result = model.caption(image, length=length, stream=args.stream)
            if args.stream:
                for token in caption_result["caption"]:
                    print(token, end="", flush=True)
                print()
            else:
                print(caption_result["caption"])
    
    if args.question:
        print(f"\nQuestion: {args.question}")
        query_result = model.query(image, args.question, stream=args.stream)
        if args.stream:
            for token in query_result["answer"]:
                print(token, end="", flush=True)
            print()
        else:
            print(f"Answer: {query_result['answer']}")


if __name__ == "__main__":
    main()