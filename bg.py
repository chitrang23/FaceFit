import os
from PIL import Image

# Use current working directory
folder_path = os.getcwd()

# Loop through all files in the folder
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)

    # Only process image files
    if filename.lower().endswith(('.png', '.tif', '.webp')):
        try:
            img = Image.open(file_path).convert("RGBA")  # Ensure image has alpha channel
            alpha_channel = img.split()[-1]  # Get the alpha channel

            # Check if all pixels are fully opaque (no transparency)
            if alpha_channel.getextrema() == (255, 255):
                base, ext = os.path.splitext(filename)
                new_name = f"{base}_not_transparent{ext}"
                new_path = os.path.join(folder_path, new_name)
                os.rename(file_path, new_path)
                print(f"Renamed: {filename} -> {new_name}")

        except Exception as e:
            print(f"Error processing {filename}: {e}")
