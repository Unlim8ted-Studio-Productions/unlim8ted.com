import os
import piexif
from PIL import Image

# Path to your local folder
folder_path = "images/Unlim8tedImages"

# File extensions to include
image_extensions = {".jpg", ".jpeg"}

# Iterate over each image file
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    if not os.path.isfile(file_path):
        continue

    ext = os.path.splitext(filename)[1].lower()
    if ext not in image_extensions:
        continue

    try:
        # Open image
        image = Image.open(file_path)

        # Try to load EXIF, fallback to empty EXIF structure
        exif_data = image.info.get("exif")
        if exif_data:
            exif_dict = piexif.load(exif_data)
        else:
            exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "Interop": {}, "1st": {}, "thumbnail": None}

        # Set or overwrite the copyright
        exif_dict["0th"][piexif.ImageIFD.Copyright] = "Unlim8ted Studios".encode("utf-8")

        # Save with updated EXIF
        exif_bytes = piexif.dump(exif_dict)
        image.save(file_path, "jpeg", exif=exif_bytes)

        print(f"✅ Updated metadata for: {filename}")

    except Exception as e:
        print(f"❌ Error processing {filename}: {e}")