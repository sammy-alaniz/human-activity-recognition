import argparse
import struct
import gzip
import json
import os
import numpy as np
import math
import onnxruntime as ort
from io import BytesIO
from PIL import Image
from tokenizers import Tokenizer
from typing import BinaryIO, Tuple, Iterator, Union, Dict, Any, List, Generator

# --- .mf File Loading Logic (Adapted from moondream/clients/python/moondream/moonfile.py) ---

MOON_MAGIC = b"MOON"
MOON_VERSION = 1

def unpack(input_path: str) -> Iterator[Tuple[str, bytes]]:
    """Unpack a .mf or .mf.gz file yielding (filename, content) pairs."""
    def _get_file_handle() -> Union[BinaryIO, gzip.GzipFile]:
        if input_path.endswith(".gz"):
            return gzip.open(input_path, "rb")
        return open(input_path, "rb")

    def _validate_header(f: Union[BinaryIO, gzip.GzipFile]) -> None:
        magic = f.read(4)
        if magic != MOON_MAGIC:
            raise ValueError(f"Invalid magic bytes: {magic}")
        version = struct.unpack("!B", f.read(1))[0]
        if version != MOON_VERSION:
            raise ValueError(f"Unsupported version: {version}")

    with _get_file_handle() as f:
        _validate_header(f)
        while True:
            filename_len_bytes = f.read(4)
            if not filename_len_bytes:
                break
            filename_len = struct.unpack("!I", filename_len_bytes)[0]
            filename = f.read(filename_len).decode("utf-8")
            content_len = struct.unpack("!Q", f.read(8))[0]
            content = f.read(content_len)
            yield filename, content

# --- Image Preprocessing Logic (Adapted from moondream/clients/python/moondream/preprocess.py) ---

def im_resize(
    image: Image.Image,
    size: Tuple[int, int],
    resample: int = Image.Resampling.BICUBIC,
) -> Image.Image:
    return image.resize(size, resample=resample)

def adaptive_avg_pool2d(x, output_size):
    """ NumPy implementation of adaptive_avg_pool2d for (H, W, C) inputs """
    H, W, C = x.shape
    if isinstance(output_size, int):
        out_H, out_W = output_size, output_size
    else:
        out_H, out_W = output_size

    stride_h = H // out_H
    stride_w = W // out_W
    kernel_h = H - (out_H - 1) * stride_h
    kernel_w = W - (out_W - 1) * stride_w

    output = np.zeros((out_H, out_W, C), dtype=x.dtype)
    for i in range(out_H):
        for j in range(out_W):
            h_start = i * stride_h
            h_end = h_start + kernel_h
            w_start = j * stride_w
            w_end = w_start + kernel_w
            output[i, j, :] = np.mean(x[h_start:h_end, w_start:w_end, :], axis=(0, 1))
    return output

def normalize(
    image: np.ndarray,
    mean: List[float] = [0.5, 0.5, 0.5],
    std: List[float] = [0.5, 0.5, 0.5],
) -> np.ndarray:
    return (image - np.array(mean)) / np.array(std)

def create_patches(
    image: Image.Image, image_patch_size=378
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """ Create patches for vision encoder """
    image = image.convert("RGB")
    patches = [im_resize(image, (image_patch_size, image_patch_size))]
    res_templates = [(1, 2), (2, 1), (2, 2)]
    im_width, im_height = image.size
    max_dim = max(im_width, im_height)

    if max_dim < image_patch_size * 1.4:
        selected_template = (1, 1)
    else:
        aspect_ratio = im_width / im_height
        selected_template = min(
            res_templates, key=lambda size: abs((size[1] / size[0]) - aspect_ratio)
        )
        patch_width = im_width // selected_template[1]
        patch_height = im_height // selected_template[0]
        for row in range(selected_template[0]):
            for col in range(selected_template[1]):
                x_min, y_min = col * patch_width, row * patch_height
                x_max, y_max = x_min + patch_width, y_min + patch_height
                patches.append(
                    im_resize(
                        image.crop((x_min, y_min, x_max, y_max)),
                        (image_patch_size, image_patch_size),
                    )
                )

    normalized_patches = np.stack(
        [
            normalize((np.array(p) / 255.0)).transpose(2, 0, 1)
            for p in patches
        ],
        dtype=np.float16, # Using float16 as seen in ONNX model usage
    )
    return normalized_patches, selected_template

# --- ONNX Model Execution Core Logic (Adapted from moondream/clients/python/moondream/onnx_vl.py) ---


class OnnxMoondream:
    def __init__(self, model_path: str):
        """
        Initialise the Moondream ONNX model and wire it up to Apple’s Core-ML
        execution provider when available (≈ GPU / Neural-Engine on Apple Silicon).
    
        Parameters
        ----------
        model_path : str
            Path to the `.mf` / `.mf.gz` bundle that contains the ONNX graphs,
            weights, tokenizer, config, etc.
        """
        import onnxruntime as ort
        from io import BytesIO
    
        print("Loading ONNX Moondream model components…")
        ort.set_default_logger_severity(0)       # silence most runtime chatter
    
        # ----------------------------------------------------------
        #  Select the best execution providers for this machine
        # ----------------------------------------------------------
        def _select_providers() -> list:
            """
            Prefer CoreMLExecutionProvider on Apple Silicon; always keep the CPU
            provider as a fallback so every op has somewhere to run.
            """
            available = ort.get_available_providers()
            print(f'AVAIL {available}')
            providers = []
    
            if "CoreMLExecutionProvider" in available:
                print("Using CoreMLExecutionProvider (Apple GPU / ANE).")
                coreml_options = {
                    # "coreml.use_cpu_only": False,      # push work to GPU / ANE
                    # "coreml.enable_on_subgraph": True, # let Core ML take partial graphs
                }
                providers.append(("CoreMLExecutionProvider", coreml_options))
    
            # Always leave the CPU EP last as a safe harbour.
            providers.append("CPUExecutionProvider")
            return providers
    
        # providers = _select_providers()
        providers = ['CoreMLExecutionProvider']
    
        # Fine-tune optimisation, keep default otherwise
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
    
        # ----------------------------------------------------------
        #  Sanity-check the model path
        # ----------------------------------------------------------
        if not os.path.isfile(model_path):
            raise ValueError(
                f"Model path is invalid or file does not exist: {model_path}"
            )
    
        # ----------------------------------------------------------
        #  Unpack the .mf bundle and load each component
        # ----------------------------------------------------------
        components: Dict[str, Any] = {}
        file_handlers = {
            "onnx": lambda c: ort.InferenceSession(
                c, providers=providers, sess_options=sess_options
            ),
            "json": lambda c: json.loads(c),
            "npy":  lambda c: np.load(BytesIO(c)),
        }
    
        required_components = [
            "vision_encoder",
            "vision_projection",
            "text_encoder",
            "text_decoder",
            "tokenizer",
            "initial_kv_cache",
            "config",
        ]
    
        found_components = set()
        tokenizer_loaded = False
    
        for filename, contents in unpack(model_path):
            key, ext = os.path.splitext(os.path.basename(filename))
            ext = ext.lstrip(".")
    
            # --- special-case the tokenizer because the API is different ---
            if key == "tokenizer":
                print("  Loading tokenizer…")
                self.tokenizer: Tokenizer = Tokenizer.from_buffer(contents)
                found_components.add("tokenizer")
                tokenizer_loaded = True
            # everything else: use the generic handlers above
            elif ext in file_handlers:
                print(f"  Loading {key}.{ext}…")
                components[key] = file_handlers[ext](contents)
                found_components.add(key)
            else:
                print(f"  Skipping unknown file type: {filename}")
    
        # tokenizer might have ended up in `components` if not caught above
        if not tokenizer_loaded and "tokenizer" in components:
            self.tokenizer = components["tokenizer"]
            found_components.add("tokenizer")
    
        # ----------------------------------------------------------
        #  Make sure we have every part we need
        # ----------------------------------------------------------
        for key in required_components:
            if key not in found_components:
                raise ValueError(f"Missing required component ‘{key}’ in model file.")
            if key == "tokenizer" and tokenizer_loaded:
                continue  # already stored in self.tokenizer
            setattr(self, key, components.get(key))
    
        # ----------------------------------------------------------
        #  Pull a few frequently-used config bits into attributes
        # ----------------------------------------------------------
        self.special_tokens = self.config["special_tokens"]
    
        if (
            "tokenizer" in self.config
            and "templates" in self.config["tokenizer"]
        ):
            self.templates = self.config["tokenizer"]["templates"]
        elif "templates" in self.config:
            self.templates = self.config["templates"]
        else:
            raise ValueError("Could not find ‘templates’ in the model config.")
    
        self.eos_token_id = self.special_tokens["eos"]
    
        print("ONNX Moondream model loaded successfully.")
    
    

    def encode_image(self, image: Image.Image) -> Tuple[int, np.ndarray]:
        """ Encodes an image into embeddings and KV cache state. """
        print("Encoding image...")
        image_patches, template = create_patches(image)

        # Run vision encoder
        (patch_emb,) = self.vision_encoder.run(None, {"input": image_patches})

        # Reassemble patches (simplified version from onnx_vl.py logic)
        global_patch = patch_emb[0]
        if patch_emb.shape[0] > 1: # If we have more than the global patch
            seq_len = patch_emb.shape[-2] # Sequence length (e.g., 729 for 27x27 grid)
            w = int(math.sqrt(seq_len))
            patch_emb_local = patch_emb[1:] # Local patches

            # Reshape local patches and potentially use adaptive_avg_pool2d
            # This part is complex to replicate exactly without PyTorch,
            # using a simplified approach for demonstration.
            # We'll average the local patch embeddings as a placeholder.
            # A more accurate implementation would involve the spatial rearrangement
            # and pooling seen in the original PyTorch code.
            patch_emb_avg = np.mean(patch_emb_local, axis=0)
            patch_emb_combined = np.concatenate([global_patch, patch_emb_avg], axis=-1)
        else: # Only global patch
             patch_emb_combined = np.concatenate([global_patch, global_patch], axis=-1)


        # Run vision projection
        patch_emb_combined = np.expand_dims(patch_emb_combined, axis=0) # Add batch dimension
        (input_embeds,) = self.vision_projection.run(None, {"input": patch_emb_combined})

        # Run image embeddings through text decoder to get initial KV cache state
        kv_cache = self.initial_kv_cache.copy() # Use the precomputed initial cache
        (kv_cache_update,) = self.text_decoder.run(
            ["new_kv_cache"], {"input_embeds": input_embeds, "kv_cache": kv_cache}
        )
        kv_cache = np.concatenate([kv_cache, kv_cache_update], axis=-2) # Axis based on typical cache shapes (..., seq_len, dim)
        pos = input_embeds.shape[-2] + self.initial_kv_cache.shape[-2]
        print("Image encoding complete.")
        return pos, kv_cache


    def generate_answer(self, image_encoding: Tuple[int, np.ndarray], question: str, max_tokens: int = 50) -> str:
            """ Generates an answer to the question based on the encoded image. """
            print("Generating answer...")
            initial_pos, initial_kv_cache = image_encoding
    
            # Prepare prompt tokens
            if "query" not in self.templates:
                raise ValueError("Model config does not support querying.")
            # Ensure templates are accessed correctly
            query_template = self.templates.get("query", {})
            prefix = query_template.get("prefix", [])
            suffix = query_template.get("suffix", [])
            if not prefix or not suffix:
                 print("Warning: Query prefix or suffix not found in model config templates.")
                 # Add default fallbacks if necessary, or raise a more specific error
                 # For now, we proceed, but this might indicate a config loading issue
    
            prompt_toks = prefix + self.tokenizer.encode(question).ids + suffix
            prompt_toks = np.array([prompt_toks], dtype=np.int64)
    
            # Encode prompt
            (prompt_embeds,) = self.text_encoder.run(None, {"input_ids": prompt_toks})
    
            # --- Autoregressive Generation Loop ---
            generated_tokens = []
            pos = initial_pos
            kv_cache = initial_kv_cache.copy() # Use a copy for generation
            # Expand kv_cache seq_len dimension if needed (e.g., to 2048) - common practice
            target_seq_len = 2048 # Or get from config if available
            current_seq_len = kv_cache.shape[-2]
            if current_seq_len < target_seq_len:
                 pad_width = [(0, 0)] * (kv_cache.ndim - 2) + [(0, target_seq_len - current_seq_len), (0, 0)]
                 kv_cache = np.pad(kv_cache, pad_width, mode='constant', constant_values=0)
    
            current_embeds = prompt_embeds
            input_len = current_embeds.shape[1]
    
            for _ in range(max_tokens):
                # Get logits and update KV cache
                # Note: Slicing the kv_cache up to the current position 'pos' is crucial
                kv_cache_input = kv_cache[..., :pos, :]
                logits, kv_cache_update = self.text_decoder.run(
                    ["logits", "new_kv_cache"],
                    {"input_embeds": current_embeds, "kv_cache": kv_cache_input},
                )
    
                # Update the actual kv_cache in place
                # Ensure kv_cache_update fits into the allocated space
                update_len = kv_cache_update.shape[-2]
                if pos + update_len > kv_cache.shape[-2]:
                     print(f"Warning: KV cache update exceeds allocated size. Current pos: {pos}, update_len: {update_len}, cache size: {kv_cache.shape[-2]}")
                     # Handle potential overflow, e.g., by stopping or resizing cache (complex)
                     break
                kv_cache[..., pos : pos + update_len, :] = kv_cache_update
                pos += update_len # Use actual update length
    
                # --- MODIFIED LINE ---
                # Sample next token (greedy decoding)
                # Assuming logits output shape is (batch_size, vocab_size) for single token prediction
                if logits.ndim == 2 and logits.shape[0] == 1:
                     next_token_id = np.argmax(logits[0, :]) # Simpler indexing for (1, vocab_size)
                elif logits.ndim == 3 and logits.shape[1] == 1: # Shape might be (batch, 1, vocab)
                     next_token_id = np.argmax(logits[0, 0, :])
                elif logits.ndim == 3 and logits.shape[1] > 1: # Original handling if sequence length > 1
                     next_token_id = np.argmax(logits[0, -1, :]) # Get last token prediction from first batch item
                else:
                     # Fallback or error if shape is unexpected
                     print(f"Warning: Unexpected logits shape: {logits.shape}. Attempting argmax on last dimension.")
                     try:
                         next_token_id = np.argmax(logits.flatten()) # Best guess fallback
                     except Exception as e:
                          print(f"Error determining next token from logits shape {logits.shape}: {e}")
                          break # Stop generation if we can't determine the next token
    
                # --- END MODIFIED BLOCK ---
    
                if next_token_id == self.eos_token_id:
                    print("EOS token encountered.")
                    break
    
                generated_tokens.append(int(next_token_id)) # Ensure it's an integer
    
                # Prepare input for next iteration
                next_token_arr = np.array([[next_token_id]], dtype=np.int64)
                (current_embeds,) = self.text_encoder.run(None, {"input_ids": next_token_arr})
                input_len = 1 # Next input is just one token
    
    
            answer = self.tokenizer.decode(generated_tokens)
            print("Answer generation complete.")
            return answer

# --- Main Execution Logic ---

def main():
    parser = argparse.ArgumentParser(description="Load Moondream ONNX model, parse image, and query from scratch.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the .mf or .mf.gz model file.")
    parser.add_argument("--image-path", type=str, required=True, help="Path to the image file.")
    parser.add_argument("--query", type=str, required=True, help="The question to ask about the image.")
    parser.add_argument("--max-tokens", type=int, default=50, help="Maximum number of tokens to generate.")
    args = parser.parse_args()

    try:
        # 1. Load Model Components from .mf file
        onnx_model = OnnxMoondream(args.model_path)

        # 2. Load Image
        print(f"Loading image from: {args.image_path}")
        try:
            image = Image.open(args.image_path)
        except FileNotFoundError:
            print(f"Error: Image file not found at {args.image_path}")
            return
        except Exception as e:
            print(f"Error loading image: {e}")
            return

        # 3. Encode Image (Parse)
        image_encoding = onnx_model.encode_image(image)

        # 4. Generate Answer (Query)
        answer = onnx_model.generate_answer(image_encoding, args.query, args.max_tokens)

        # 5. Print Result
        print("\n" + "="*20 + " Query Result " + "="*20)
        print(f"Query: {args.query}")
        print(f"Answer: {answer}")
        print("="*54)

    except ValueError as ve:
        print(f"\nError: {ve}")
    except FileNotFoundError as fnf:
         print(f"\nError: {fnf}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()