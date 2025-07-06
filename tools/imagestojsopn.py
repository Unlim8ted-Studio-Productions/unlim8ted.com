import os
import json

# Path to your local folder
folder_path = "images/Unlim8tedImages"

# File extensions to include
image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff"}

# Get list of image files
image_files = [
    filename for filename in os.listdir(folder_path)
    if os.path.isfile(os.path.join(folder_path, filename)) and os.path.splitext(filename)[1].lower() in image_extensions
]

# Output JSON file path
output_file = os.path.join("tools/data", "images.json")

# Save as JSON
with open(output_file, "w") as f:
    json.dump(image_files, f, indent=2)

print(f"Created {output_file} with {len(image_files)} images.")
