import json

INPUT_FILE = "tools/data/products.json"
OUTPUT_FILE = "tools/output.json"


def extract_details(items):
    for obj in items:
        desc = obj.get("description")

        if not isinstance(desc, str):
            continue

        bullet_index = desc.find("•")
        if bullet_index == -1:
            continue

        end_index = desc.find("\n\n\n", bullet_index)
        if end_index == -1:
            continue

        # Include the delimiter in the cut
        end_index += len("\n\n\n")

        extracted = desc[bullet_index:end_index].strip()

        # Save WITH bullet
        obj["details"] = extracted

        # Remove from description
        obj["description"] = (desc[:bullet_index] + desc[end_index:]).strip()

    return items


def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("JSON root must be a list of objects")

    updated = extract_details(data)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(updated, f, indent=2, ensure_ascii=False)

    print(f"Done. Processed {len(updated)} items → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
