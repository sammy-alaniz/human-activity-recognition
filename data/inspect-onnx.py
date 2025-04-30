import onnx
import os
import sys

def inspect_onnx_operations(directory):
    """
    Loads all ONNX files in a directory and compiles a unique list of operations.

    Args:
        directory (str): The path to the directory containing the .onnx files.

    Returns:
        list: A sorted list of unique operation types found, or None if errors occur.
    """
    if not os.path.isdir(directory):
        print(f"Error: Directory not found: {directory}", file=sys.stderr)
        return None

    onnx_files = [f for f in os.listdir(directory) if f.endswith(".onnx")]

    if not onnx_files:
        print(f"Error: No .onnx files found in directory: {directory}", file=sys.stderr)
        return None

    print(f"Found {len(onnx_files)} ONNX files to inspect:")
    for f in onnx_files:
        print(f"  - {f}")
    print("-" * 30)

    all_ops = set()
    total_nodes_processed = 0

    for filename in onnx_files:
        file_path = os.path.join(directory, filename)
        print(f"Processing: {filename}...")
        try:
            # Load the ONNX model
            model = onnx.load(file_path)
            print(f"  Successfully loaded.")

            # Access the graph
            graph = model.graph
            file_ops = set()
            nodes_in_file = len(graph.node)
            total_nodes_processed += nodes_in_file

            # Iterate through all nodes in the graph
            for node in graph.node:
                all_ops.add(node.op_type)
                file_ops.add(node.op_type)

            print(f"  Found {len(file_ops)} unique ops in this file ({nodes_in_file} nodes total).")

        except FileNotFoundError:
            print(f"  Error: File not found at {file_path}", file=sys.stderr)
        except onnx.onnx_cpp2py_export.checker.ValidationError as e:
             print(f"  Error: Model validation failed for {filename}: {e}", file=sys.stderr)
        except Exception as e:
            print(f"  Error processing {filename}: {e}", file=sys.stderr)
        print("-" * 30)


    if not all_ops:
        print("No operations could be extracted.", file=sys.stderr)
        return None

    # Convert set to a sorted list for readability
    sorted_ops = sorted(list(all_ops))
    return sorted_ops

# --- Main execution ---
if __name__ == "__main__":
    # ** IMPORTANT: Set this to the correct directory where your .onnx files are **
    onnx_directory = 'k400/weights/ext/' # Use the path from your extraction log

    unique_operations = inspect_onnx_operations(onnx_directory)

    if unique_operations:
        print("\n==============================================")
        print("Unique Operations Found Across All Files:")
        print("==============================================")
        for op in unique_operations:
            print(op)
        print(f"\nTotal unique operations found: {len(unique_operations)}")