import os
import argparse
import sys
from math import ceil, sqrt

# Attempt to import Pillow (PIL)
try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("Error: Pillow library not found.")
    print("Please install it using: pip install Pillow")
    sys.exit(1) # Exit if Pillow is not available

def find_jpg_files(directories, max_files):
    """
    Recursively walks through specified directories to find JPG files.

    Args:
        directories (list): A list of directory paths to search within.
        max_files (int): The maximum number of JPG file paths to collect.

    Returns:
        list: A list containing the full paths of the found JPG files,
              up to max_files. Returns an empty list if no JPGs are found
              or if the directories don't exist.
    """
    jpg_files = []
    found_count = 0

    if not isinstance(directories, list):
        print("Error: Input 'directories' must be a list of paths.")
        return []

    print(f"Searching for up to {max_files} JPG/JPEG files...")
    for directory in directories:
        if not os.path.isdir(directory):
            print(f"Warning: '{directory}' is not a valid directory. Skipping.")
            continue

        print(f"Scanning directory: {directory}")
        for root, _, files in os.walk(directory):
            if found_count >= max_files:
                break

            # Sort files to potentially get a more consistent order (optional)
            files.sort()

            for filename in files:
                if filename.lower().endswith(('.jpg', '.jpeg')):
                    full_path = os.path.join(root, filename)
                    jpg_files.append(full_path)
                    found_count += 1
                    if found_count % 50 == 0: # Print progress indicator
                         print(f"  Found {found_count} files...")
                    if found_count >= max_files:
                        print(f"Reached target of {max_files} files.")
                        return jpg_files # Return immediately

            if found_count >= max_files:
                break # Stop searching this directory

        if found_count >= max_files:
            break # Stop searching remaining directories

    if not jpg_files:
        print("No JPG files found.")
    elif found_count < max_files:
        print(f"Warning: Found only {found_count} JPG files, less than the target {max_files}.")

    return jpg_files

def create_photo_grid(image_paths, grid_dims, tile_size, output_filename, background_color='gray'):
    """
    Creates a grid image from a list of image paths.

    Args:
        image_paths (list): List of paths to the images.
        grid_dims (tuple): A tuple (columns, rows) for the grid layout.
        tile_size (tuple): A tuple (width, height) for each tile in pixels.
        output_filename (str): Path to save the final grid image.
        background_color (str): Background color for the grid canvas.
    """
    grid_cols, grid_rows = grid_dims
    tile_w, tile_h = tile_size
    num_expected_files = grid_cols * grid_rows
    num_found_files = len(image_paths)

    if num_found_files == 0:
        print("Error: No image paths provided to create the grid.")
        return

    print(f"\nCreating a {grid_cols}x{grid_rows} grid...")
    print(f"Tile size: {tile_w}x{tile_h} pixels")
    print(f"Output file: {output_filename}")

    # Calculate final image dimensions
    canvas_width = grid_cols * tile_w
    canvas_height = grid_rows * tile_h

    # Create the blank canvas
    canvas = Image.new('RGB', (canvas_width, canvas_height), color=background_color)
    draw = ImageDraw.Draw(canvas)
    try:
        # Use a default font or try to load a small truetype font if available
        font = ImageFont.load_default()
        # font = ImageFont.truetype("arial.ttf", 10) # Example: Requires arial.ttf
    except IOError:
        print("Warning: Default font not found. Error text on tiles might not render.")
        font = None # Set font to None if loading fails

    processed_count = 0
    # Iterate through the grid positions and paste images
    for i in range(num_expected_files):
        row = i // grid_cols
        col = i % grid_cols
        paste_x = col * tile_w
        paste_y = row * tile_h

        if i < num_found_files:
            img_path = image_paths[i]
            try:
                # Open the image
                img = Image.open(img_path)
                # Ensure image is in RGB mode (handles PNGs with alpha, etc.)
                img = img.convert('RGB')
                # Resize the image to the target tile size using high-quality resampling
                img_resized = img.resize((tile_w, tile_h), Image.Resampling.LANCZOS)
                # Paste the resized image onto the canvas
                canvas.paste(img_resized, (paste_x, paste_y))
                processed_count += 1
                if processed_count % 50 == 0: # Progress indicator
                    print(f"  Processed {processed_count}/{num_found_files} images...")

            except FileNotFoundError:
                print(f"Error: Image file not found at {img_path}. Skipping.")
                if font:
                    draw.text((paste_x + 5, paste_y + 5), f"Not Found:\n{os.path.basename(img_path)}", fill='white', font=font)
            except Exception as e:
                print(f"Error processing image {img_path}: {e}. Skipping.")
                # Optionally draw a placeholder indicating an error
                draw.rectangle([paste_x, paste_y, paste_x + tile_w -1 , paste_y + tile_h -1], outline='red', width=2)
                if font:
                    draw.text((paste_x + 5, paste_y + 5), f"Error:\n{os.path.basename(img_path)}", fill='red', font=font)
        else:
            # If we ran out of images, fill the remaining grid cells with the background color
            # (already done by Image.new, but could draw a border or something here if desired)
            pass # Keep background color

    print(f"\nFinished processing {processed_count} images.")

    # Save the final image
    try:
        canvas.save(output_filename)
        print(f"Grid image successfully saved to: {output_filename}")
    except Exception as e:
        print(f"Error saving the final image: {e}")

# --- Main execution block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find JPG files and assemble them into a grid image.")
    parser.add_argument('dirs', metavar='DIR', type=str, nargs='+',
                        help='One or more directories to search recursively for JPG/JPEG files.')
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='Filename for the output grid image (e.g., photo_grid.jpg). REQUIRED.')
    parser.add_argument('--grid', metavar=('COLS', 'ROWS'), type=int, nargs=2, default=[25, 25],
                        help='Dimensions of the grid (columns rows). Default: 25 25.')
    parser.add_argument('--tile-size', metavar=('W', 'H'), type=int, nargs=2, default=[100, 100],
                        help='Size (width height) of each tile in pixels. Default: 100 100.')
    parser.add_argument('--bg-color', type=str, default='gray',
                        help='Background color for the grid canvas (e.g., white, black, #CCCCCC). Default: gray.')


    args = parser.parse_args()

    # Calculate the total number of files needed based on grid dimensions
    grid_cols, grid_rows = args.grid
    num_files_needed = grid_cols * grid_rows

    # 1. Find the image files
    found_files = find_jpg_files(args.dirs, num_files_needed)

    # 2. Check if any files were found
    if not found_files:
        print("\nExiting: No JPG files found, cannot create grid.")
        sys.exit(1)

    if len(found_files) < num_files_needed:
         print(f"\nWarning: Only found {len(found_files)} files, but the {grid_cols}x{grid_rows} grid requires {num_files_needed}.")
         print("The grid will be created with the found images, and remaining cells will be filled with the background color.")


    # 3. Create the grid image
    create_photo_grid(
        image_paths=found_files,
        grid_dims=args.grid,
        tile_size=args.tile_size,
        output_filename=args.output,
        background_color=args.bg_color
    )

    print("\nScript finished.")
