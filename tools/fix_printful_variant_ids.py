import os
import json
import argparse
import time
import requests
from typing import Any, Dict, List

PRINTFUL_BASE = "https://api.printful.com"


def pf_headers(token: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }


def pf_request(token: str, method: str, url: str, *, params=None):
    r = requests.request(
        method, url, headers=pf_headers(token), params=params, timeout=60
    )
    r.raise_for_status()
    return r.json()


def list_store_products(token: str) -> List[Dict[str, Any]]:
    out = []
    offset = 0
    limit = 100

    while True:
        result = pf_request(
            token,
            "GET",
            f"{PRINTFUL_BASE}/store/products",
            params={"offset": offset, "limit": limit},
        )
        batch = result.get("result") or []
        if not batch:
            break
        out.extend(batch)
        offset += limit
        time.sleep(0.1)
    return out


def get_store_product_detail(token: str, store_product_id: str) -> Dict[str, Any]:
    result = pf_request(
        token, "GET", f"{PRINTFUL_BASE}/store/products/{store_product_id}"
    )
    return result.get("result") or {}


def build_sync_to_catalog_map(token: str) -> Dict[str, int]:
    sync_to_catalog = {}
    print("Fetching store products from Printful...")

    products = list_store_products(token)
    print(f" → {len(products)} store products found")

    for p in products:
        pid = str(p.get("id")).strip()
        if not pid:
            continue

        detail = get_store_product_detail(token, pid)
        variants = detail.get("sync_variants") or detail.get("variants") or []
        for v in variants:
            ext_id = v.get("external_id") or v.get("external_variant_id")
            if not ext_id:
                continue
            ext_id = str(ext_id).strip()

            catalog_id = v.get("variant_id")  # catalog variant id
            if catalog_id:
                sync_to_catalog[ext_id] = catalog_id

        time.sleep(0.08)

    return sync_to_catalog


def update_products_json(in_path: str, out_path: str, mapping: Dict[str, int]):
    with open(in_path, "r", encoding="utf-8") as f:
        products = json.load(f)

    for product in products:
        for v in product.get("varients") or []:
            vid = str(v.get("id") or "").strip()
            if not vid:
                continue

            # Always keep original sync id
            v["printful_cat_id"] = vid

            # Add catalog variant id if known
            if vid in mapping:
                v["printful_catalog_variant_id"] = mapping[vid]

        # You could also add printful catalog product id at product level if needed
        # For store products the store detail payload does NOT return one universally,
        # but if your data does include it you can attach it here.

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(products, f, indent=2, ensure_ascii=False)

    print(f"Updated products JSON written to: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in", dest="in_path", required=True, help="Input products.json"
    )
    parser.add_argument(
        "--out", dest="out_path", default=None, help="Output updated JSON"
    )
    parser.add_argument("--token", default=None, help="Printful API token")
    args = parser.parse_args()

    token = args.token or os.environ.get("PRINTFUL_ACCESS_TOKEN") or ""
    if not token:
        raise SystemExit("Missing Printful API token")

    mapping = build_sync_to_catalog_map(token)
    print(f"\nBuilt sync_id → catalog_variant_id map ({len(mapping)} entries)")

    out_path = args.out_path or args.in_path.replace(".json", "_updated.json")
    update_products_json(args.in_path, out_path, mapping)


if __name__ == "__main__":
    main()
