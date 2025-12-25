import os
import re
import argparse

# Regex: finds URLs inside src="", href="", url("..."), url('...')
ASSET_REGEX = re.compile(
    r'src=["\']([^"\']+)["\']|href=["\']([^"\']+)["\']|url\((?:["\']?)([^)"\']+)(?:["\']?)\)',
    re.IGNORECASE,
)


def find_used_assets(html_path):
    """Return a set of all asset paths referenced in the HTML."""
    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()

    used = set()

    for match in ASSET_REGEX.findall(html):
        for group in match:
            if group.strip():
                # Normalize slashes
                path = group.replace("\\", "/")
                used.add(path)

    return used


def collect_all_assets(asset_dir):
    """Return a set of file paths inside the asset directory."""
    all_assets = set()

    for root, _, files in os.walk(asset_dir):
        for f in files:
            rel = os.path.relpath(os.path.join(root, f), asset_dir)
            rel = rel.replace("\\", "/")
            all_assets.add(rel)

    return all_assets


def main(html_file, asset_dir, delete=False):
    print(f"📄 Reading HTML: {html_file}")
    print(f"📁 Searching assets in: {asset_dir}")

    used = find_used_assets(html_file)
    all_assets = collect_all_assets(asset_dir)

    unused = sorted(all_assets - used)

    print("\n🔍 USED ASSETS FOUND:")
    for u in sorted(used):
        print("  ✔", u)

    print("\n🗑 UNUSED ASSETS:")
    if unused:
        for f in unused:
            print("  ✖", f)
    else:
        print("  🎉 No unused assets found!")

    if delete and unused:
        print("\n⚠️ DELETING UNUSED FILES...")
        for f in unused:
            full = os.path.join(asset_dir, f)
            if os.path.exists(full):
                os.remove(full)
                print("  🗑 Deleted:", f)
        print("✨ Cleanup complete!")
    elif delete:
        print("Nothing to delete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scan HTML & remove unused image/video assets."
    )
    parser.add_argument("html", help="Path to the HTML file")
    parser.add_argument("assets", help="Asset directory to scan")
    parser.add_argument("--delete", action="store_true", help="Delete unused files")

    args = parser.parse_args()
    main(args.html, args.assets, args.delete)
