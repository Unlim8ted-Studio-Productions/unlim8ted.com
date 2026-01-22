import os
import sys

OLD = "assets.unlim8ted.com"
NEW = "https://assets.unlim8ted.com"

# File extensions to process (edit if needed)
TEXT_EXTENSIONS = {
    ".html",
    ".htm",
    ".css",
    ".js",
    ".json",
    ".txt",
    ".md",
    ".xml",
    ".svg",
    ".yml",
    ".yaml",
}


def is_text_file(path):
    return os.path.splitext(path)[1].lower() in TEXT_EXTENSIONS


def process_file(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
    except UnicodeDecodeError:
        return 0  # skip binary / non-utf8

    if OLD not in content:
        return 0

    new_content = content.replace(OLD, NEW)

    with open(path, "w", encoding="utf-8") as f:
        f.write(new_content)

    return content.count(OLD)


def walk_and_replace(root):
    total_files = 0
    total_replacements = 0

    for root_dir, _, files in os.walk(root):
        for file in files:
            path = os.path.join(root_dir, file)
            if not is_text_file(path):
                continue

            count = process_file(path)
            if count > 0:
                total_files += 1
                total_replacements += count
                print(f"✔ {path} ({count} replacements)")

    print("\nDone.")
    print(f"Files modified: {total_files}")
    print(f"Total replacements: {total_replacements}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python fix_assets_domain.py <directory>")
        sys.exit(1)

    walk_and_replace(sys.argv[1])
