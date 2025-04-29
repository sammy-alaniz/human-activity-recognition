import argparse
import struct
import gzip
import os
from typing import BinaryIO, Iterator, Tuple, Union

# --- .mf File Loading Logic (Adapted from moondream/clients/python/moondream/moonfile.py) ---

MOON_MAGIC = b"MOON"
MOON_VERSION = 1

def unpack(input_path: str) -> Iterator[Tuple[str, bytes]]:
    """
    Unpack a .mf or .mf.gz file yielding (filename, content) pairs.
    The filename can include directory paths relative to the archive root.
    """
    def _get_file_handle() -> Union[BinaryIO, gzip.GzipFile]:
        if not os.path.isfile(input_path):
             raise FileNotFoundError(f"Input file not found: {input_path}")
        try:
            if input_path.endswith(".gz"):
                return gzip.open(input_path, "rb")
            return open(input_path, "rb")
        except Exception as e:
            raise IOError(f"Error opening file {input_path}: {e}") from e

    def _validate_header(f: Union[BinaryIO, gzip.GzipFile]) -> None:
        try:
            magic = f.read(4)
            if magic != MOON_MAGIC:
                raise ValueError(f"Invalid magic bytes: {magic!r}. Expected {MOON_MAGIC!r}.")
            version_byte = f.read(1)
            if not version_byte:
                 raise ValueError("File ended unexpectedly after magic bytes.")
            version = struct.unpack("!B", version_byte)[0]
            if version != MOON_VERSION:
                raise ValueError(f"Unsupported version: {version}. Expected {MOON_VERSION}.")
        except struct.error as e:
             raise ValueError(f"Error reading header structure: {e}") from e
        except Exception as e:
            # Catch potential read errors
            raise IOError(f"Error reading header from file: {e}") from e

    print(f"Opening archive: {input_path}")
    try:
        with _get_file_handle() as f:
            _validate_header(f)
            print("Archive header validated.")
            file_index = 0
            while True:
                # Read filename length (4 bytes, unsigned int, big-endian)
                filename_len_bytes = f.read(4)
                if not filename_len_bytes:
                    print("End of archive reached.")
                    break # End of archive
                if len(filename_len_bytes) < 4:
                    raise ValueError(f"Incomplete data reading filename length at file index {file_index}.")
                filename_len = struct.unpack("!I", filename_len_bytes)[0]

                # Read filename (UTF-8 encoded)
                if filename_len == 0:
                     raise ValueError(f"Filename length is zero at file index {file_index}.")
                filename_bytes = f.read(filename_len)
                if len(filename_bytes) < filename_len:
                     raise ValueError(f"Incomplete data reading filename (expected {filename_len} bytes) at file index {file_index}.")
                try:
                    filename = filename_bytes.decode("utf-8")
                except UnicodeDecodeError as e:
                    raise ValueError(f"Error decoding filename at index {file_index}: {e}") from e

                # Read content length (8 bytes, unsigned long long, big-endian)
                content_len_bytes = f.read(8)
                if not content_len_bytes or len(content_len_bytes) < 8:
                     raise ValueError(f"Incomplete data reading content length for file '{filename}' (index {file_index}).")
                content_len = struct.unpack("!Q", content_len_bytes)[0]

                # Read content
                print(f"  Reading file {file_index}: '{filename}' ({content_len} bytes)...")
                # Read in chunks to handle potentially large files and provide feedback
                content = bytearray()
                bytes_read = 0
                chunk_size = 1024 * 1024 # 1MB chunks
                while bytes_read < content_len:
                    read_size = min(chunk_size, content_len - bytes_read)
                    chunk = f.read(read_size)
                    if not chunk:
                         raise ValueError(f"Incomplete data reading content for file '{filename}'. Expected {content_len} bytes, got {bytes_read}.")
                    content.extend(chunk)
                    bytes_read += len(chunk)

                yield filename, bytes(content) # Yield filename and the complete content as bytes
                file_index += 1

    except (ValueError, IOError, FileNotFoundError) as e:
         print(f"\nError during unpacking: {e}")
         # Re-raise to indicate failure in the calling function
         raise
    except Exception as e:
         print(f"\nAn unexpected error occurred during unpacking: {e}")
         import traceback
         traceback.print_exc()
         raise

# --- Main Extraction Logic ---

def main():
    parser = argparse.ArgumentParser(description="Extract all files from a .mf or .mf.gz archive.")
    parser.add_argument("input_file", type=str, help="Path to the input .mf or .mf.gz file.")
    parser.add_argument("output_dir", type=str, help="Directory where extracted files will be saved.")
    args = parser.parse_args()

    print(f"Input archive: {args.input_file}")
    print(f"Output directory: {args.output_dir}")

    try:
        # Create the output directory if it doesn't exist
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Ensured output directory exists: {args.output_dir}")

        extracted_count = 0
        # Unpack the archive and save each file
        for filename, content in unpack(args.input_file):
            # Construct the full path for the output file
            output_path = os.path.join(args.output_dir, filename)

            # Create necessary subdirectories within the output directory
            output_subdir = os.path.dirname(output_path)
            if output_subdir: # Only create if filename includes a path
                os.makedirs(output_subdir, exist_ok=True)

            # Write the content to the file
            try:
                with open(output_path, "wb") as outfile:
                    outfile.write(content)
                print(f"  Extracted: '{filename}' -> '{output_path}'")
                extracted_count += 1
            except IOError as e:
                 print(f"  Error writing file '{output_path}': {e}")
            except Exception as e:
                 print(f"  An unexpected error occurred writing file '{output_path}': {e}")


        print(f"\nExtraction complete. {extracted_count} file(s) extracted to '{args.output_dir}'.")

    except FileNotFoundError as e:
        print(f"\nError: {e}")
    except (ValueError, IOError) as e:
         # Errors from unpack or file writing errors not caught inside the loop
        print(f"\nExtraction failed due to error: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()