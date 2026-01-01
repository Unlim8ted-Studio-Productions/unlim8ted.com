#!/usr/bin/env python3
"""
Remove ONLY each variant's `images` list from products.json (PHYSICAL only by default).
Do NOT remove `image`.

Edits:
  - For every product varients[] item:
      delete key "images" if present
      keep "image" unchanged

Usage:
  python tools\\strip_variant_images_list.py --in tools\\data\\products.json --write-inplace

Options:
  --all-products   Apply to all products (not just product-type: physical)
"""

import json
import argparse
from typing import Any, Dict, List


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def is_physical_product(p: Dict[str, Any]) -> bool:
    return str(p.get("product-type") or "").strip().lower() == "physical"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="Path to products.json")
    ap.add_argument(
        "--out", default=None, help="Output path (default: <in>_no_variant_images.json)"
    )
    ap.add_argument("--write-inplace", action="store_true", help="Overwrite input file")
    ap.add_argument(
        "--all-products",
        action="store_true",
        help="Apply to all products (not just physical)",
    )
    args = ap.parse_args()

    data = load_json(args.in_path)
    if not isinstance(data, list):
        raise SystemExit("products.json must be a JSON array")

    removed = 0
    variants_seen = 0

    for p in data:
        if not isinstance(p, dict):
            continue
        if (not args.all_products) and (not is_physical_product(p)):
            continue

        vars_ = p.get("varients") or []
        if not isinstance(vars_, list):
            continue

        for v in vars_:
            if not isinstance(v, dict):
                continue
            variants_seen += 1
            if "images" in v:
                v.pop("images", None)
                removed += 1

    out_path = (
        args.in_path
        if args.write_inplace
        else (args.out or args.in_path.replace(".json", "_no_variant_images.json"))
    )
    write_json(out_path, data)

    print("--- Done ---")
    print(f"variants_seen: {variants_seen}")
    print(f"variant_images_lists_removed: {removed}")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
