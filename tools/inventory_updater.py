#!/usr/bin/env python3
"""
Square production: fetch inventory for varients[].id and write into products.json.

Adds:
  variant["stock"] (sum of IN_STOCK across locations)
  variant["stock_debug"] (states seen, per location)

This version debugs why you're getting 0s:
- queries ALL locations by default (unless --location is set)
- retrieves counts WITHOUT states filter so we can see returned states
- optional: verifies IDs are ITEM_VARIATION via Catalog API

pip install requests
Env:
  SQUARE_ACCESS_TOKEN (required)

Usage:
  python square_write_stock_to_json.py --in products.json --write-inplace
  python square_write_stock_to_json.py --in products.json --out products_with_stock.json
  python square_write_stock_to_json.py --in products.json --location YOUR_LOCATION_ID --write-inplace
  python square_write_stock_to_json.py --in products.json --verify-catalog
"""

import os, json, argparse
from typing import Dict, List, Any, Tuple
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


def load_products(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Expected top-level JSON to be a list of products.")
    return data


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

    # de-dupe keep order
    seen = set()
    uniq = []
    for vid in ids:
        if vid not in seen:
            seen.add(vid)
            uniq.append(vid)
    return uniq


def chunk(lst: List[str], n: int):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def batch_retrieve_counts(
    token: str, location_ids: List[str], variation_ids: List[str]
) -> List[Dict[str, Any]]:
    """
    POST /v2/inventory/counts/batch-retrieve
    Return raw counts list entries.
    We do NOT filter states so we can see what Square returns.
    """
    all_counts: List[Dict[str, Any]] = []
    for part in chunk(variation_ids, 200):
        payload = {
            "catalog_object_ids": part,
            "location_ids": location_ids,
            # intentionally omit "states" for debugging
        }
        r = requests.post(
            f"{SQUARE_BASE}/inventory/counts/batch-retrieve",
            headers=headers(token),
            json=payload,
            timeout=60,
        )
        r.raise_for_status()
        all_counts.extend(r.json().get("counts") or [])
    return all_counts


def parse_counts(
    counts: List[Dict[str, Any]],
) -> Tuple[Dict[str, int], Dict[str, Any], set]:
    """
    Build:
      stock_map[variation_id] = sum(IN_STOCK quantities)
      debug_map[variation_id] = { location_id: {state: qty, ...}, ... }
      seen_ids = set of variation ids that returned any count entry
    """
    stock_map: Dict[str, int] = {}
    debug_map: Dict[str, Any] = {}
    seen_ids = set()

    for c in counts:
        vid = c.get("catalog_object_id")
        loc = c.get("location_id")
        state = c.get("state")
        qty_str = c.get("quantity", "0")

        try:
            qty = int(float(qty_str))
        except ValueError:
            qty = 0

        if not vid or not loc or not state:
            continue

        seen_ids.add(vid)
        debug_map.setdefault(vid, {}).setdefault(loc, {})
        debug_map[vid][loc][state] = debug_map[vid][loc].get(state, 0) + qty

        if state == "IN_STOCK":
            stock_map[vid] = stock_map.get(vid, 0) + qty

    return stock_map, debug_map, seen_ids


def catalog_verify_variations(token: str, variation_ids: List[str]) -> Dict[str, str]:
    """
    Uses Catalog batch-retrieve to confirm object types.
    POST /v2/catalog/batch-retrieve
    Returns map: id -> type (or 'MISSING')
    """
    out: Dict[str, str] = {}
    for part in chunk(variation_ids, 200):
        payload = {"object_ids": part, "include_related_objects": False}
        r = requests.post(
            f"{SQUARE_BASE}/catalog/batch-retrieve",
            headers=headers(token),
            json=payload,
            timeout=60,
        )
        r.raise_for_status()
        objs = r.json().get("objects") or []
        found = {o.get("id"): o.get("type") for o in objs if o.get("id")}
        for vid in part:
            out[vid] = found.get(vid, "MISSING")
    return out


def apply_to_products(
    products: List[Dict[str, Any]], stock_map: Dict[str, int], debug_map: Dict[str, Any]
) -> int:
    updated = 0
    for p in products:
        vars_ = p.get("varients") or []
        if not isinstance(vars_, list):
            continue
        for v in vars_:
            vid = (v.get("id") or "").strip()
            if not vid:
                continue
            v["stock"] = int(stock_map.get(vid, 0))
            v["stock_debug"] = debug_map.get(vid, {})
            updated += 1
    return updated


def write_products(path: str, products: List[Dict[str, Any]]):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(products, f, ensure_ascii=False, indent=2)
        f.write("\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_path", default=None)
    ap.add_argument("--write-inplace", action="store_true")
    ap.add_argument(
        "--location", default=None, help="If set, query ONLY this location id"
    )
    ap.add_argument(
        "--verify-catalog",
        action="store_true",
        help="Verify IDs are ITEM_VARIATION via Catalog API",
    )
    args = ap.parse_args()

    token = ""
    if not token:
        raise SystemExit("Missing env var SQUARE_ACCESS_TOKEN (production token).")

    if args.write_inplace and args.out_path:
        raise SystemExit("Use either --out or --write-inplace, not both.")

    products = load_products(args.in_path)
    variation_ids = collect_variation_ids(products)
    if not variation_ids:
        raise SystemExit("No varients[].id found in JSON.")

    locs = get_locations(token)
    if not locs:
        raise SystemExit("No Square locations found.")

    if args.location:
        location_ids = [args.location.strip()]
        loc_name = next(
            (l.get("name") for l in locs if l.get("id") == location_ids[0]), None
        )
        print(f"Using location: {location_ids[0]} ({loc_name or 'unknown'})")
    else:
        location_ids = [l["id"] for l in locs if l.get("id")]
        print(
            f"Using ALL locations ({len(location_ids)}): "
            + ", ".join([l.get("name", "?") for l in locs])
        )

    # Inventory counts
    raw_counts = batch_retrieve_counts(token, location_ids, variation_ids)
    stock_map, debug_map, seen_ids = parse_counts(raw_counts)

    # Report missing counts (this is your 'everything is 0' smoking gun)
    missing = [vid for vid in variation_ids if vid not in seen_ids]
    if missing:
        print(
            f"\nWARNING: {len(missing)} variation IDs returned NO inventory counts at queried location(s)."
        )
        print(
            "This usually means inventory tracking is OFF for those items, or these IDs are not catalog variation IDs."
        )
        print("First 20 missing IDs:\n  " + "\n  ".join(missing[:20]))

    # Optional catalog verify
    if args.verify_catalog:
        types = catalog_verify_variations(token, variation_ids)
        bad = [(vid, t) for vid, t in types.items() if t != "ITEM_VARIATION"]
        if bad:
            print(
                f"\nCatalog verify: {len(bad)} IDs are not ITEM_VARIATION (or missing). First 20:"
            )
            for vid, t in bad[:20]:
                print(f"  {vid}: {t}")
        else:
            print("\nCatalog verify: all IDs are ITEM_VARIATION ✅")

    # Apply + write
    updated = apply_to_products(products, stock_map, debug_map)
    out_path = (
        args.in_path
        if args.write_inplace
        else (args.out_path or args.in_path.replace(".json", "_with_stock.json"))
    )
    write_products(out_path, products)

    # Quick summary
    nonzero = sum(1 for vid in variation_ids if stock_map.get(vid, 0) > 0)
    print(f"\nVariants total: {len(variation_ids)}")
    print(f"Variants with stock>0: {nonzero}")
    print(f"Variants updated in JSON: {updated}")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
