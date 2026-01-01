#!/usr/bin/env python3
"""
Printful → products.json image initializer

Matches:
  Printful sync_variant.external_id  ==  products.json varients[].id

Updates (PHYSICAL products only):
  product.image   (best single URL)
  product.images  (list of URLs)
  variant.image   (best single URL, when match found)
  variant.images  (list of URLs, when match found)

Pull sources (best-effort, varies by Printful data):
  - store product detail: sync_product.thumbnail_url / product files / mockup urls
  - sync variant: files[].preview_url / thumbnail_url / mockup urls

Env:
  PRINTFUL_ACCESS_TOKEN

Usage (Windows CMD):
  set PRINTFUL_ACCESS_TOKEN=YOUR_TOKEN
  python tools\\printful_init_images.py --in tools\\data\\products.json --write-inplace

Options:
  --keep-existing         Keep existing image fields if Printful yields none (default: overwrite with none = remove)
  --sleep 0.08            Delay between requests
"""

import os
import json
import time
import argparse
from typing import Any, Dict, List, Optional, Tuple
import requests

PRINTFUL_BASE = "https://api.printful.com"


# -----------------------------
# IO
# -----------------------------


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


# -----------------------------
# Local helpers
# -----------------------------


def is_physical_product(p: Dict[str, Any]) -> bool:
    return str(p.get("product-type") or "").strip().lower() == "physical"


def norm_url(s: Any) -> Optional[str]:
    if not isinstance(s, str):
        return None
    u = s.strip()
    if not u:
        return None
    if u.startswith("http://") or u.startswith("https://"):
        return u
    return None


def dedupe_keep_order(urls: List[str]) -> List[str]:
    seen = set()
    out = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


# -----------------------------
# Printful HTTP
# -----------------------------


def pf_headers(token: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "Content-Type": "application/json",
        "User-Agent": "unlim8ted-printful-init-images/1.0",
    }


def pf_request(
    token: str, method: str, url: str, *, params=None, timeout=60
) -> requests.Response:
    return requests.request(
        method, url, headers=pf_headers(token), params=params, timeout=timeout
    )


def pf_json(r: requests.Response) -> Any:
    try:
        return r.json()
    except Exception:
        return None


def get_store_product_detail(token: str, store_product_id: str) -> Dict[str, Any]:
    url = f"{PRINTFUL_BASE}/store/products/{store_product_id}"
    r = pf_request(token, "GET", url)
    j = pf_json(r)
    if r.status_code != 200:
        raise RuntimeError(
            f"GET /store/products/{store_product_id} failed HTTP {r.status_code}\n"
            f"{json.dumps(j, indent=2) if isinstance(j,(dict,list)) else r.text[:1200]}"
        )
    if isinstance(j, dict) and isinstance(j.get("result"), dict):
        return j["result"]
    return j if isinstance(j, dict) else {}


# -----------------------------
# Image extraction (best-effort)
# -----------------------------

URL_KEYS_PRIORITY = [
    "thumbnail_url",
    "mockup_url",
    "preview_url",
    "image_url",
    "url",
]


def extract_urls_from_files(files: Any) -> List[str]:
    out: List[str] = []
    if not isinstance(files, list):
        return out
    for f in files:
        if not isinstance(f, dict):
            continue
        for k in URL_KEYS_PRIORITY:
            u = norm_url(f.get(k))
            if u:
                out.append(u)
        # sometimes nested
        for k in ("preview", "thumbnail", "mockup"):
            v = f.get(k)
            if isinstance(v, dict):
                for kk in URL_KEYS_PRIORITY:
                    u = norm_url(v.get(kk))
                    if u:
                        out.append(u)
    return out


def extract_urls_from_obj(obj: Any) -> List[str]:
    """
    Conservative extraction: only pull values from known url-ish keys
    inside dicts/lists. Avoids dragging unrelated URLs.
    """
    out: List[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            kl = str(k).lower()
            if kl in URL_KEYS_PRIORITY or kl.endswith("_url"):
                u = norm_url(v)
                if u:
                    out.append(u)
            # recurse
            out.extend(extract_urls_from_obj(v))
    elif isinstance(obj, list):
        for x in obj:
            out.extend(extract_urls_from_obj(x))
    return out


def choose_best_image(urls: List[str]) -> Optional[str]:
    """
    Heuristic: prefer likely thumbnails/mockups over generic file urls.
    """
    if not urls:
        return None

    # score by substrings
    def score(u: str) -> int:
        ul = u.lower()
        s = 0
        if "thumbnail" in ul:
            s += 50
        if "mockup" in ul:
            s += 40
        if "preview" in ul:
            s += 30
        if "printfile" in ul:
            s -= 10
        return s

    best = sorted(urls, key=lambda u: score(u), reverse=True)[0]
    return best


# -----------------------------
# Core mapping logic
# -----------------------------


def build_externalid_to_variant_images(detail: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Returns: external_id -> [urls...]
    """
    out: Dict[str, List[str]] = {}

    variants = (
        detail.get("sync_variants")
        or detail.get("variants")
        or detail.get("items")
        or []
    )
    if not isinstance(variants, list):
        return out

    for v in variants:
        if not isinstance(v, dict):
            continue
        ext = (
            v.get("external_id") or v.get("external_variant_id") or v.get("externalId")
        )
        ext = str(ext).strip() if ext else ""
        if not ext:
            continue

        urls: List[str] = []

        # common variant file containers
        urls.extend(extract_urls_from_files(v.get("files")))
        urls.extend(extract_urls_from_files(v.get("mockup_files")))
        urls.extend(extract_urls_from_files(v.get("product_files")))

        # direct url fields anywhere in variant
        urls.extend(extract_urls_from_obj(v))

        urls = [u for u in urls if u]
        urls = dedupe_keep_order(urls)
        out[ext] = urls

    return out


def extract_product_images(detail: Dict[str, Any]) -> List[str]:
    """
    Best-effort product-level images from sync_product and top-level.
    """
    urls: List[str] = []

    sync_product = (
        detail.get("sync_product")
        if isinstance(detail.get("sync_product"), dict)
        else None
    )
    if sync_product:
        # most common single image fields
        for k in ("thumbnail_url", "mockup_url"):
            u = norm_url(sync_product.get(k))
            if u:
                urls.append(u)

        # file arrays
        urls.extend(extract_urls_from_files(sync_product.get("files")))
        urls.extend(extract_urls_from_files(sync_product.get("mockup_files")))
        urls.extend(extract_urls_from_files(sync_product.get("product_files")))

        # conservative crawl
        urls.extend(extract_urls_from_obj(sync_product))

    # sometimes product images appear at top-level too
    urls.extend(extract_urls_from_obj(detail))

    urls = [u for u in urls if u]
    urls = dedupe_keep_order(urls)
    return urls


def apply_images_to_products(
    products: List[Dict[str, Any]],
    token: str,
    *,
    keep_existing: bool,
    sleep_s: float,
) -> Dict[str, int]:
    stats = {
        "products_total": len(products),
        "physical_products_seen": 0,
        "physical_products_with_printful_id": 0,
        "printful_detail_fetched": 0,
        "product_images_set": 0,
        "variant_images_set": 0,
        "variants_matched": 0,
        "variants_skipped_no_id": 0,
        "products_skipped_no_printful_id": 0,
        "errors": 0,
    }

    for p in products:
        if not is_physical_product(p):
            continue
        stats["physical_products_seen"] += 1

        pfid = str(p.get("printful_id") or "").strip()
        if not pfid:
            stats["products_skipped_no_printful_id"] += 1
            continue
        stats["physical_products_with_printful_id"] += 1

        try:
            detail = get_store_product_detail(token, pfid)
            stats["printful_detail_fetched"] += 1
        except Exception as e:
            stats["errors"] += 1
            print(f"[WARN] {p.get('id')} printful_id={pfid} error: {e}")
            continue

        # product images
        prod_urls = extract_product_images(detail)
        prod_urls = dedupe_keep_order(prod_urls)
        prod_main = choose_best_image(prod_urls)

        if prod_main or prod_urls:
            p["image"] = prod_main or (prod_urls[0] if prod_urls else None)
            p["images"] = prod_urls
            stats["product_images_set"] += 1
        else:
            if not keep_existing:
                p.pop("image", None)
                p.pop("images", None)

        # variant images by external_id match
        ext_to_urls = build_externalid_to_variant_images(detail)
        vars_ = p.get("varients") or []
        if isinstance(vars_, list):
            for v in vars_:
                if not isinstance(v, dict):
                    continue
                vid = str(v.get("id") or "").strip()
                if not vid:
                    stats["variants_skipped_no_id"] += 1
                    continue

                urls = ext_to_urls.get(vid)
                if urls is None:
                    # no match - optionally remove if not keeping
                    if not keep_existing:
                        v.pop("image", None)
                        v.pop("images", None)
                    continue

                stats["variants_matched"] += 1
                urls = dedupe_keep_order([u for u in urls if u])
                main = choose_best_image(urls)

                if main or urls:
                    v["image"] = main or (urls[0] if urls else None)
                    v["images"] = urls
                    stats["variant_images_set"] += 1
                else:
                    if not keep_existing:
                        v.pop("image", None)
                        v.pop("images", None)

        time.sleep(max(0.0, sleep_s))

    return stats


# -----------------------------
# CLI
# -----------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="Path to products.json")
    ap.add_argument(
        "--out", default=None, help="Output path (default: <in>_with_images.json)"
    )
    ap.add_argument("--write-inplace", action="store_true", help="Overwrite input file")
    ap.add_argument(
        "--keep-existing",
        action="store_true",
        help="If Printful yields no images, keep existing fields",
    )
    ap.add_argument(
        "--sleep",
        type=float,
        default=0.08,
        help="Sleep seconds between Printful product detail requests",
    )
    ap.add_argument(
        "--token",
        default=None,
        help="Printful token (or use PRINTFUL_ACCESS_TOKEN env var)",
    )
    args = ap.parse_args()

    token = (args.token or os.environ.get("PRINTFUL_ACCESS_TOKEN") or "").strip()
    if not token:
        raise SystemExit("Missing PRINTFUL_ACCESS_TOKEN (set env var) or pass --token")

    products = load_json(args.in_path)
    if not isinstance(products, list):
        raise SystemExit("products.json must be a JSON array")

    print("Initializing images from Printful (physical only)…")
    stats = apply_images_to_products(
        products, token, keep_existing=args.keep_existing, sleep_s=args.sleep
    )

    out_path = (
        args.in_path
        if args.write_inplace
        else (args.out or args.in_path.replace(".json", "_with_images.json"))
    )
    write_json(out_path, products)

    print("\n--- Done ---")
    for k in sorted(stats.keys()):
        print(f"{k}: {stats[k]}")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
