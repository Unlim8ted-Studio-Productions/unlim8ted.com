import xml.etree.ElementTree as ET
import json


def parse_rss_to_json(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    namespace = {"g": "http://base.google.com/ns/1.0"}
    items = []

    for item in root.findall(".//item"):
        product = {
            "id": (
                item.find("./g:id", namespace).text
                if item.find("./g:id", namespace) is not None
                else None
            ),
            "image": (
                item.find("./g:image_link", namespace).text
                if item.find("./g:image_link", namespace) is not None
                else None
            ),
            "additional_images": [],  # You can extract additional image links if available
            "video": None,  # Placeholder for video links if available
            "additional_videos": [],  # Placeholder for additional videos
            "name": (
                item.find("./g:title", namespace).text
                if item.find("./g:title", namespace) is not None
                else None
            ),
            "description": (
                item.find("./g:description", namespace).text
                if item.find("./g:description", namespace) is not None
                else None
            ),
            "price": (
                float(item.find("./g:price", namespace).text.split()[0])
                if item.find("./g:price", namespace) is not None
                else 0.0
            ),
            "link": (
                item.find("./g:link", namespace).text
                if item.find("./g:link", namespace) is not None
                else None
            ),
        }
        items.append(product)

    return json.dumps(items, indent=2)


# Specify your XML file
xml_file = "google_merchant_products.xml"
output_json = parse_rss_to_json(xml_file)


print(output_json)
