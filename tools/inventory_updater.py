#!/usr/bin/env python3
"""
Update products.json with Square Online style availability:

available = (not deleted) AND sellable AND present_at_location AND (
              if stockable: IN_STOCK > 0
              else: True
           )

Input JSON shape:
[
  {
    "id": "...",
    "varients": [
      {"id": "SQUARE_VARIATION_ID", ...}
    ]
  }
]

Requires:
  pip install requests

Env:
  SQUARE_ACCESS_TOKEN (required)  production token
  SQUARE_LOCATION_ID (optional)   fulfillment location id (recommended)
                                 If omitted, uses the first Square location.

Usage:
  python tools/inventory_updater.py --in tools/data/products.json --write-inplace
  python tools/inventory_updater.py --in tools/data/products.json --write-inplace --location L4KPR2BE0PAA4
"""

import os
import json
import argparse
from typing import Dict, List, Any

import requests

SQUARE_BASE = "https://connect.squareup.com/v2"


def headers(token: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def get_locations(token: str) -> List[Dict[str, Any]]:
    r = requests.get(f"{SQUARE_BASE}/locations", headers=headers(token), timeout=30)
    r.raise_for_status()
    return r.json().get("locations") or []


def get_first_location_id(token: str) -> str:
    locs = get_locations(token)
    if not locs:
        raise RuntimeError("No Square locations found.")
    return locs[0]["id"]


def load_products(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Expected top-level JSON to be a list of products.")
    return data


def write_products(path: str, data: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def collect_variation_ids(products: List[Dict[str, Any]]) -> List[str]:
    ids: List[str] = []
    for p in products:
        vars_ = p.get("varients") or []
        if not isinstance(vars_, list):
            continue
        for v in vars_:
            vid = (v.get("id") or "").strip()
            if vid:
                ids.append(vid)

    # de-dupe while preserving order
    seen = set()
    uniq: List[str] = []
    for vid in ids:
        if vid not in seen:
            seen.add(vid)
            uniq.append(vid)
    return uniq


def chunk(lst: List[str], n: int = 200):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def fetch_variations(token: str, ids: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    POST /v2/catalog/batch-retrieve
    Returns map: variation_id -> ITEM_VARIATION object
    """
    out: Dict[str, Dict[str, Any]] = {}
    for part in chunk(ids, 200):
        r = requests.post(
            f"{SQUARE_BASE}/catalog/batch-retrieve",
            headers=headers(token),
            json={"object_ids": part, "include_related_objects": False},
            timeout=60,
        )
        r.raise_for_status()
        for obj in r.json().get("objects") or []:
            if obj.get("type") == "ITEM_VARIATION" and obj.get("id"):
                out[obj["id"]] = obj
    return out


def present_in_location(obj: Dict[str, Any], location_id: str) -> bool:
    """
    Square presence logic:
    - If present_at_all_locations == True:
        present unless absent_at_location_ids contains location
    - If present_at_all_locations == False:
        if present_at_location_ids exists -> MUST include location
        else if absent_at_location_ids exists -> MUST NOT include location
        else assume present
    """
    if not location_id:
        return True

    pal = obj.get("present_at_all_locations", True)
    present_ids = obj.get("present_at_location_ids") or []
    absent_ids = obj.get("absent_at_location_ids") or []

    if pal is True:
        return location_id not in absent_ids

    # pal is False
    if present_ids:
        return location_id in present_ids

    if absent_ids:
        return location_id not in absent_ids

    return True


def fetch_instock_counts(
    token: str, location_id: str, variation_ids: List[str]
) -> Dict[str, int]:
    """
    POST /v2/inventory/counts/batch-retrieve
    Returns map: variation_id -> sum(IN_STOCK) at that location.
    If inventory tracking is OFF, Square commonly returns no entry for that id.
    """
    stock: Dict[str, int] = {}
    for part in chunk(variation_ids, 200):
        payload = {
            "catalog_object_ids": part,
            "location_ids": [location_id],
            "states": ["IN_STOCK"],
        }
        r = requests.post(
            f"{SQUARE_BASE}/inventory/counts/batch-retrieve",
            headers=headers(token),
            json=payload,
            timeout=60,
        )
        r.raise_for_status()
        for c in r.json().get("counts") or []:
            vid = c.get("catalog_object_id")
            if not vid:
                continue
            try:
                qty = int(float(c.get("quantity", "0")))
            except ValueError:
                qty = 0
            stock[vid] = stock.get(vid, 0) + qty
    return stock


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--write-inplace", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--location", default=None, help="Square location ID (fulfillment origin)"
    )
    args = ap.parse_args()

    token = (os.environ.get("SQUARE_ACCESS_TOKEN") or "").strip()
    if not token:
        raise SystemExit("Missing SQUARE_ACCESS_TOKEN (set env var).")

    # pick location
    location_id = (args.location or os.environ.get("SQUARE_LOCATION_ID") or "").strip()
    if not location_id:
        location_id = get_first_location_id(token)

    products = load_products(args.in_path)
    var_ids = collect_variation_ids(products)
    if not var_ids:
        raise SystemExit("No varients[].id found in JSON.")

    catalog = fetch_variations(token, var_ids)

    # We only need inventory counts for stockable variations,
    # but easiest is to fetch once for all ids and then use conditionally.
    instock_map = fetch_instock_counts(token, location_id, var_ids)

    checked = 0
    unavailable = 0
    missing_in_catalog = 0
    stockable_count = 0

    for p in products:
        vars_ = p.get("varients") or []
        if not isinstance(vars_, list):
            continue

        for v in vars_:
            vid = (v.get("id") or "").strip()
            if not vid:
                continue

            obj = catalog.get(vid)

            # annotate
            v["available_location_id"] = location_id
            v["availability_source"] = "square_catalog_plus_inventory"

            if not obj:
                # Not found in catalog => treat unavailable (or keep old value)
                v["available"] = False
                v["stock"] = None
                v["stockable"] = False
                missing_in_catalog += 1
                unavailable += 1
                checked += 1
                continue

            if obj.get("is_deleted"):
                v["available"] = False
                v["stock"] = None
                v["stockable"] = False
                unavailable += 1
                checked += 1
                continue

            iv = obj.get("item_variation_data", {}) or {}
            sellable = bool(iv.get("sellable", True))
            stockable = bool(iv.get("stockable", False))
            loc_ok = present_in_location(obj, location_id)

            # stock handling
            if stockable:
                stockable_count += 1
                stock = int(instock_map.get(vid, 0))
                v["stock"] = stock
                v["stockable"] = True
                v["available"] = bool(sellable and loc_ok and stock > 0)
            else:
                # Printful commonly lands here
                v["stock"] = None
                v["stockable"] = False
                v["available"] = bool(sellable and loc_ok)

            if not v["available"]:
                unavailable += 1

            checked += 1

    out_path = (
        args.in_path
        if args.write_inplace
        else (args.out or args.in_path.replace(".json", "_updated.json"))
    )
    write_products(out_path, products)

    print(f"Location used: {location_id}")
    print(f"Variants checked: {len(var_ids)}")
    print(f"Stockable variants (inventory tracked): {stockable_count}")
    print(f"Missing in catalog: {missing_in_catalog}")
    print(f"Unavailable at this location: {unavailable}")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
