import requests
import json
import argparse
import sys
import base64
import os

# Default parameters mirroring the JavaScript 'params' signal object
DEFAULT_PARAMS = {
    "n_predict": 400,
    "temperature": 0.7,
    "repeat_last_n": 256,
    "repeat_penalty": 1.18,
    "top_k": 40,
    "top_p": 0.95,
    "min_p": 0.05,
    "tfs_z": 1.0,
    "typical_p": 1.0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "mirostat": 0,
    "mirostat_tau": 5.0,
    "mirostat_eta": 0.1,
    "grammar": "",
    "n_probs": 0,
    "min_keep": 0,
    "cache_prompt": True,
    "api_key": "",
    # --- Parameters added dynamically in JS ---
    # "prompt": "...", # Added from input or session
    # "stream": True,   # Essential for streaming response
    # "stop": [],       # Stop sequences (e.g., ["</s>", "User:"])
    # "image_data": [], # For multimodal models
    # "slot_id": -1,    # For multi-user context management
}

def make_request(url: str, payload: dict):
    """
    Sends a request to the llama.cpp server and streams the response.

    Args:
        url: The full URL of the server's completion endpoint.
        payload: A dictionary containing the prompt and generation parameters.
    """
    headers = {"Content-Type": "application/json"}
    # Note: The JS code includes api_key within the 'params' object which
    # becomes part of the JSON body. If your server expects an Authorization
    # header instead, adjust accordingly (e.g., add
    # headers["Authorization"] = f"Bearer {payload.get('api_key', '')}")
    # and remove api_key from the payload if necessary.

    print("--- Sending Request ---")
    print(f"URL: {url}")
    # Avoid printing potentially large image data
    print_payload = {k: v for k, v in payload.items() if k != 'image_data'}
    print(f"Payload: {json.dumps(print_payload, indent=2)}")
    print("--- Streaming Response ---")

    try:
        response = requests.post(url, json=payload, stream=True, headers=headers)
        print(response)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

        full_response = ""
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                if decoded_line.startswith("data: "):
                    json_content = decoded_line[len("data: "):]
                    try:
                        chunk = json.loads(json_content)
                        content = chunk.get("content", "")
                        full_response += content
                        print(content, end='', flush=True) # Print content as it arrives

                        # Check if generation stopped
                        if chunk.get("stop"):
                            print("\n--- Generation Stopped ---")
                            timings = chunk.get("timings")
                            if timings:
                                print("--- Timings ---")
                                print(json.dumps(timings, indent=2))
                            break # Exit loop after stop signal

                    except json.JSONDecodeError:
                        print(f"\n[Warning: Could not decode JSON chunk: {json_content}]", file=sys.stderr)
                    except Exception as e:
                         print(f"\n[Error processing chunk: {e}]", file=sys.stderr)
                         print(f"Problematic chunk content: {json_content}", file=sys.stderr)


        print("\n--- Request Finished ---")
        # print("\nFull Response:\n", full_response) # Optionally print the full concatenated response

    except requests.exceptions.RequestException as e:
        print(f"\n--- Request Error ---", file=sys.stderr)
        print(f"Error connecting to or communicating with the server: {e}", file=sys.stderr)
    except Exception as e:
        print(f"\n--- An Unexpected Error Occurred ---", file=sys.stderr)
        print(f"{e}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Send requests to a local llama.cpp server.")

    parser.add_argument("prompt", help="The input prompt for the LLM.")
    parser.add_argument("--url", default="http://localhost:8080/completion",
                        help="URL of the llama.cpp server /completion endpoint (default: http://localhost:8080/completion)")
    parser.add_argument("--image", help="Path to an image file for multimodal input.")
    parser.add_argument("--image-id", type=int, default=10, help="ID to associate with the image (default: 10)")
    parser.add_argument("--stop", action='append', default=[],
                        help="Specify stop sequences. Can be used multiple times (e.g., --stop 'User:' --stop '</s>').")

    # Add arguments for parameters, using defaults from DEFAULT_PARAMS
    for key, value in DEFAULT_PARAMS.items():
        arg_name = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
             parser.add_argument(arg_name, type=lambda x: (str(x).lower() == 'true'), default=value,
                                help=f"(default: {value})")
        elif isinstance(value, int):
            parser.add_argument(arg_name, type=int, default=value,
                                help=f"(default: {value})")
        elif isinstance(value, float):
             parser.add_argument(arg_name, type=float, default=value,
                                help=f"(default: {value})")
        else: # Primarily strings (like grammar, api_key)
             parser.add_argument(arg_name, type=str, default=value,
                                help=f"(default: '{value}')")

    args = parser.parse_args()

    # --- Build Payload ---
    payload = {}

    # Add parameters from command line args, overriding defaults
    for key in DEFAULT_PARAMS.keys():
        payload[key] = getattr(args, key.replace('-', '_')) # Convert hyphens back to underscores for attribute access

    # Add core request data
    payload["prompt"] = args.prompt
    payload["stream"] = True # Force streaming as we process line by line
    payload["stop"] = args.stop # Add stop sequences specified by user

    # Handle image data if provided
    if args.image:
        if not os.path.exists(args.image):
            print(f"Error: Image file not found at {args.image}", file=sys.stderr)
            sys.exit(1)
        try:
            with open(args.image, "rb") as f:
                encoded_image = base64.b64encode(f.read()).decode('utf-8')
            # The JS code strips the prefix `data:image/...;base64,`, so we provide just the base64 data
            payload["image_data"] = [{"data": encoded_image, "id": args.image_id}]
            print(f"--- Added image: {args.image} with ID: {args.image_id} ---")
            # Often, multimodal models require a specific prompt format including the image marker
            # Example: "USER: [img-10]Describe this image.\nASSISTANT:"
            # Ensure your --prompt includes the appropriate marker (e.g., [img-{args.image_id}])
            if f"[img-{args.image_id}]" not in payload["prompt"]:
                 print(f"[Warning: Prompt does not contain image marker '[img-{args.image_id}]'. Make sure it's included for the model to see the image.]", file=sys.stderr)

        except Exception as e:
             print(f"Error reading or encoding image file {args.image}: {e}", file=sys.stderr)
             sys.exit(1)
    else:
         payload["image_data"] = [] # Ensure it's an empty list if no image

    # Use the function to make the request
    make_request(args.url, payload)

if __name__ == "__main__":
    main()