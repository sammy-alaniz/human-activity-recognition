#!/usr/bin/env python3

import requests
import json

# --- Configuration ---
# Adjust the API endpoint URL if your server runs elsewhere
API_URL = "http://localhost:8080/completions/"

# Define the payload (parameters for the completion)
# Modify the prompt, stop sequences, and other parameters as needed
payload = {
  "prompt": "This is a conversation between User and Llama, a friendly chatbot.\n\nUser: Write a short poem about Fort Worth, Texas.\nLlama:",
  "stream": True,         # <<< Important: Set to True for streaming
  "n_predict": 400,       # Max tokens to generate
  "temperature": 0.7,
  "stop": [               # Sequences that stop generation
    "</s>",
    "Llama:",
    "User:"
  ],
  "repeat_penalty": 1.18,
  "repeat_last_n": 256,
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
  "grammar": "",          # Set grammar if needed (e.g., for JSON output)
  "n_probs": 0,
  "min_keep": 0,
  "image_data": [],       # Add image data if using a multimodal model
  "cache_prompt": True,
  "slot_id": -1,          # Use -1 to automatically assign a slot
  "api_key": ""           # Add API key if required by server config
}
# --- End Configuration ---

def run_completion():
    """Sends the request and processes the streaming response."""
    print(f"Sending request to {API_URL}...")
    print("-" * 20)
    print("Llama's response:")

    full_response_content = ""

    try:
        # Make the streaming POST request
        response = requests.post(
            API_URL,
            headers={"Content-Type": "application/json"}, # Optional: requests sets this with json=
            json=payload,           # Pass the payload as JSON
            stream=True             # <<< Enable response streaming
        )

        # Check for successful response (e.g., 200 OK)
        response.raise_for_status()

        # Process the stream line by line (Server-Sent Events)
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                # SSE lines start with "data: "
                if decoded_line.startswith('data: '):
                    json_data_string = decoded_line[len('data: '):].strip()
                    if json_data_string: # Ensure it's not just "data: "
                        try:
                            data = json.loads(json_data_string)
                            content = data.get('content', '')
                            full_response_content += content
                            # Print the content chunk immediately
                            print(content, end='', flush=True)

                            # Optional: Check if the server signalled stop
                            if data.get('stop'):
                                print("\n[Server signalled stop]")
                                break
                        except json.JSONDecodeError:
                            print(f"\n[Error decoding JSON chunk: {json_data_string}]")

        print("\n" + "-" * 20)
        print("[End of Stream]")

    except requests.exceptions.RequestException as e:
        print(f"\nError connecting to server or during request: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

    # Optional: Return the fully assembled response content
    # return full_response_content

if __name__ == "__main__":
    run_completion()
    # To use the full response later:
    # final_text = run_completion()
    # if final_text:
    #     print("\n--- Full Assembled Response ---")
    #     print(final_text)