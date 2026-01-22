#!/usr/bin/env python3
"""RERUNNING WILL BREAK RIGHT NOW
Download product/variant images referenced in /assets/data/products.json
and rewrite the JSON to point at the new hosted asset URLs.

What it does:
- Loads ./assets/data/products.json
- For each product where product["product-type"] == "physical":
  - Downloads product["image"] (if present)
  - Downloads each URL in product["images"] (if present)
  - Downloads each variant["image"] (if present)
- Saves images under:
    ./assets/images/products/clothes/{product_id}/{variant_id_or_product}/<filename.ext>
- Rewrites those URL fields in JSON to:
    {ASSET_BASE_URL}/images/products/clothes/{product_id}/{variant_id_or_product}/<filename.ext>

Notes:
- Uses the final URL after redirects to derive filename if possible.
- If filename has no extension, it tries to infer from Content-Type.
- Skips re-downloading if the file already exists (unless --force).
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse, unquote

import requests


DEFAULT_JSON_PATH = Path("assets/data/products.json")
DEFAULT_OUT_ROOT = Path("assets/images/products/clothes")
DEFAULT_ASSET_BASE_URL = "https://assets.unlim8ted.com"


def slugify(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^\w\-]+", "-", s)  # keep word chars and hyphen
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s or "item"


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def guess_ext_from_content_type(content_type: str) -> str:
    if not content_type:
        return ""
    ct = content_type.split(";")[0].strip().lower()
    ext = mimetypes.guess_extension(ct) or ""
    # Some servers return image/jpg; normalize
    if ext == ".jpe":
        ext = ".jpg"
    return ext


def filename_from_url(url: str) -> str:
    parsed = urlparse(url)
    name = Path(unquote(parsed.path)).name
    # Strip query-like artifacts if any got into name
    name = name.split("?")[0].split("#")[0]
    return name


def is_http_url(s: str) -> bool:
    return isinstance(s, str) and (s.startswith("http://") or s.startswith("https://"))


@dataclass
class DownloadResult:
    local_path: Path
    public_url: str


class ImageRewriter:
    def __init__(
        self,
        out_root: Path,
        asset_base_url: str,
        timeout_sec: int = 30,
        force: bool = False,
        user_agent: str = "unlim8ted-products-image-migrator/1.0",
    ) -> None:
        self.out_root = out_root
        self.asset_base_url = asset_base_url.rstrip("/")
        self.timeout_sec = timeout_sec
        self.force = force
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})

    def download_to(
        self,
        url: str,
        product_id: str,
        bucket_id: str,
    ) -> DownloadResult:
        """
        Download url into out_root/product_id/bucket_id/<filename> and return new public URL.
        """
        prod = slugify(product_id)
        buck = slugify(bucket_id)

        target_dir = self.out_root / prod / buck
        ensure_dir(target_dir)

        # Determine filename
        original_name = filename_from_url(url)
        if not original_name or original_name in ("/", ".", ".."):
            original_name = "image"

        # If no extension, infer after HEAD/GET
        name_root, name_ext = os.path.splitext(original_name)
        if not name_root:
            name_root = "image"

        # Decide local file path (may need content-type to finalize extension)
        # We'll request and then decide final name if needed.
        resp = self.session.get(
            url, stream=True, timeout=self.timeout_sec, allow_redirects=True
        )
        resp.raise_for_status()

        final_url = resp.url
        content_type = resp.headers.get("Content-Type", "")

        # If original had no ext, try infer from final_url or content-type
        if not name_ext:
            # try from final url path
            final_name = filename_from_url(final_url)
            _, final_ext = os.path.splitext(final_name)
            name_ext = final_ext or guess_ext_from_content_type(content_type) or ".bin"

        # Normalize extension a little
        name_ext = name_ext.lower()
        if name_ext == ".jpeg":
            name_ext = ".jpg"

        safe_filename = f"{slugify(name_root)}{name_ext}"
        local_path = target_dir / safe_filename

        # If exists and not forcing, skip writing but still return public URL
        rel_path = (
            local_path.as_posix().replace("assets/", "", 1)
            if local_path.as_posix().startswith("assets/")
            else local_path.as_posix()
        )
        public_url = f"{self.asset_base_url}/{rel_path}"

        if local_path.exists() and not self.force:
            resp.close()
            return DownloadResult(local_path=local_path, public_url=public_url)

        # Write file
        with open(local_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 256):
                if chunk:
                    f.write(chunk)

        return DownloadResult(local_path=local_path, public_url=public_url)


def load_products_json(path: Path) -> Tuple[Any, List[Dict[str, Any]]]:
    """
    Supports either:
      - a list of products
      - an object { "products": [ ... ] }
    Returns (root_json, products_list_reference)
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data, data
    if isinstance(data, dict) and isinstance(data.get("products"), list):
        return data, data["products"]
    raise ValueError(
        "Unsupported products.json format: expected list or {products:[...]}."
    )


def get_variants(product: Dict[str, Any]) -> List[Dict[str, Any]]:
    # Your file uses "varients" (typo) but support "variants" too.
    if isinstance(product.get("varients"), list):
        return product["varients"]
    if isinstance(product.get("variants"), list):
        return product["variants"]
    return []


def rewrite_product_images(
    rewriter: ImageRewriter,
    product: Dict[str, Any],
    download_product_image: bool,
    download_images_array: bool,
    download_variant_images: bool,
) -> Dict[str, int]:
    stats = {"downloaded": 0, "skipped": 0, "failed": 0, "rewritten": 0}

    product_type = str(product.get("product-type", "")).strip().lower()
    if product_type != "physical":
        return stats

    product_id = str(product.get("id") or "").strip()
    if not product_id:
        # no stable folder name; skip
        return stats

    # 1) product["image"]
    if download_product_image:
        url = product.get("image")
        if is_http_url(url):
            try:
                r = rewriter.download_to(
                    url, product_id=product_id, bucket_id="product"
                )
                if r.public_url != url:
                    product["image"] = r.public_url
                    stats["rewritten"] += 1
                if r.local_path.exists() and not rewriter.force:
                    stats["skipped"] += 1
                else:
                    stats["downloaded"] += 1
            except Exception as e:
                stats["failed"] += 1
                print(
                    f"[WARN] product image failed for {product_id}: {e}",
                    file=sys.stderr,
                )

    # 2) product["images"] array
    if download_images_array and isinstance(product.get("images"), list):
        new_images: List[Any] = []
        for idx, url in enumerate(product["images"]):
            if is_http_url(url):
                try:
                    r = rewriter.download_to(
                        url, product_id=product_id, bucket_id=f"product-images"
                    )
                    if r.public_url != url:
                        stats["rewritten"] += 1
                    if r.local_path.exists() and not rewriter.force:
                        stats["skipped"] += 1
                    else:
                        stats["downloaded"] += 1
                    new_images.append(r.public_url)
                except Exception as e:
                    stats["failed"] += 1
                    print(
                        f"[WARN] product images[{idx}] failed for {product_id}: {e}",
                        file=sys.stderr,
                    )
                    new_images.append(url)
            else:
                new_images.append(url)
        product["images"] = new_images

    # 3) variants[*]["image"]
    if download_variant_images:
        variants = get_variants(product)
        for v in variants:
            url = v.get("image")
            if not is_http_url(url):
                continue
            variant_id = str(
                v.get("id")
                or v.get("printful_cat_id")
                or v.get("printful_catalog_variant_id")
                or "variant"
            )
            try:
                r = rewriter.download_to(
                    url, product_id=product_id, bucket_id=variant_id
                )
                if r.public_url != url:
                    v["image"] = r.public_url
                    stats["rewritten"] += 1
                if r.local_path.exists() and not rewriter.force:
                    stats["skipped"] += 1
                else:
                    stats["downloaded"] += 1
            except Exception as e:
                stats["failed"] += 1
                print(
                    f"[WARN] variant image failed for {product_id}/{variant_id}: {e}",
                    file=sys.stderr,
                )

    return stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--json",
        default=str(DEFAULT_JSON_PATH),
        help="Path to products.json (default: assets/data/products.json)",
    )
    ap.add_argument(
        "--out-root",
        default=str(DEFAULT_OUT_ROOT),
        help="Output root (default: assets/images/products/clothes)",
    )
    ap.add_argument(
        "--asset-base-url",
        default=DEFAULT_ASSET_BASE_URL,
        help="Base URL for rewritten image URLs (default: https://assets.unlim8ted.com)",
    )
    ap.add_argument(
        "--timeout", type=int, default=30, help="HTTP timeout seconds (default: 30)"
    )
    ap.add_argument(
        "--force", action="store_true", help="Redownload and overwrite files"
    )
    ap.add_argument(
        "--no-product-image",
        action="store_true",
        help="Do NOT download/replace product['image']",
    )
    ap.add_argument(
        "--no-images-array",
        action="store_true",
        help="Do NOT download/replace product['images'] array",
    )
    ap.add_argument(
        "--no-variant-images",
        action="store_true",
        help="Do NOT download/replace variant['image']",
    )
    ap.add_argument(
        "--backup", action="store_true", help="Write a .bak backup of the original JSON"
    )
    args = ap.parse_args()

    json_path = Path(args.json)
    out_root = Path(args.out_root)

    if not json_path.exists():
        print(f"ERROR: products.json not found: {json_path}", file=sys.stderr)
        return 2

    root, products = load_products_json(json_path)

    if args.backup:
        bak = json_path.with_suffix(json_path.suffix + ".bak")
        bak.write_text(json_path.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"[OK] wrote backup: {bak}")

    rewriter = ImageRewriter(
        out_root=out_root,
        asset_base_url=args.asset_base_url,
        timeout_sec=args.timeout,
        force=args.force,
    )

    totals = {
        "downloaded": 0,
        "skipped": 0,
        "failed": 0,
        "rewritten": 0,
        "products_processed": 0,
    }
    for p in products:
        if not isinstance(p, dict):
            continue
        if str(p.get("product-type", "")).strip().lower() != "physical":
            continue

        totals["products_processed"] += 1
        stats = rewrite_product_images(
            rewriter,
            p,
            download_product_image=not args.no_product_image,
            download_images_array=not args.no_images_array,
            download_variant_images=not args.no_variant_images,
        )
        for k in ("downloaded", "skipped", "failed", "rewritten"):
            totals[k] += stats[k]

    # Write back JSON nicely
    json_path.write_text(
        json.dumps(root, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(
        "[DONE] "
        f"products={totals['products_processed']} "
        f"downloaded={totals['downloaded']} "
        f"skipped={totals['skipped']} "
        f"rewritten={totals['rewritten']} "
        f"failed={totals['failed']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
