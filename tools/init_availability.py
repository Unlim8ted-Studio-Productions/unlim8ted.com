#!/usr/bin/env python3
"""
Init/Update availability flags in products.json using Printful as source of truth.

Match rule:
  Printful sync_variant.external_id  ==  products.json varients[].id

Edits:
  - Updates ONLY varients[].available (bool)
  - Does NOT touch ids, names, variants, or product fields.

Scope:
  - PHYSICAL products only: product["product-type"] == "physical"

Env:
  PRINTFUL_ACCESS_TOKEN

Usage (Windows CMD):
  set PRINTFUL_ACCESS_TOKEN=YOUR_TOKEN
  python tools\printful_init_availability.py --in tools\data\products.json --write-inplace

Options:
  --default-missing false   (default) if a local variant id isn't found in Printful map, set available=false
  --default-missing keep    keep existing value if not found
  --default-missing true    set available=true if not found (NOT recommended)
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
# Printful HTTP
# -----------------------------


def pf_headers(token: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "Content-Type": "application/json",
        "User-Agent": "unlim8ted-printful-availability/2.0",
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


def extract_list_and_paging(
    j: Any,
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    # v1: {"code":200, "result":[...], "paging":{...}}
    if isinstance(j, dict):
        res = j.get("result")
        paging = j.get("paging") if isinstance(j.get("paging"), dict) else None
        if isinstance(res, list):
            return res, paging
        if isinstance(res, dict):
            for k in ("items", "data", "results"):
                if isinstance(res.get(k), list):
                    return res[k], paging
    if isinstance(j, list):
        return j, None
    return [], None


def list_store_products_all(token: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    offset = 0
    limit = 100
    total = None

    while True:
        url = f"{PRINTFUL_BASE}/store/products"
        r = pf_request(token, "GET", url, params={"offset": offset, "limit": limit})
        j = pf_json(r)
        if r.status_code != 200:
            raise SystemExit(
                f"GET /store/products failed HTTP {r.status_code}\n"
                f"{json.dumps(j, indent=2) if isinstance(j,(dict,list)) else r.text[:1200]}"
            )

        batch, paging = extract_list_and_paging(j)
        if paging and isinstance(paging.get("total"), int):
            total = paging["total"]

        out.extend(batch)

        if total is not None:
            offset += limit
            if offset >= total:
                break
        else:
            if not batch or len(batch) < limit:
                break
            offset += limit

        time.sleep(0.12)

    return out


def get_store_product_detail(token: str, store_product_id: str) -> Dict[str, Any]:
    url = f"{PRINTFUL_BASE}/store/products/{store_product_id}"
    r = pf_request(token, "GET", url)
    j = pf_json(r)
    if r.status_code != 200:
        raise SystemExit(
            f"GET /store/products/{store_product_id} failed HTTP {r.status_code}\n"
            f"{json.dumps(j, indent=2) if isinstance(j,(dict,list)) else r.text[:1200]}"
        )
    if isinstance(j, dict) and isinstance(j.get("result"), dict):
        return j["result"]
    return j if isinstance(j, dict) else {}


# -----------------------------
# Availability inference
# -----------------------------


def infer_available(obj: Dict[str, Any]) -> Optional[bool]:
    # direct boolean fields
    for k in ("available", "is_available", "in_stock", "is_in_stock", "sellable"):
        if k in obj and isinstance(obj[k], bool):
            return bool(obj[k])

    # status-like fields
    for k in ("availability_status", "availability", "status", "stock_status"):
        if k in obj and isinstance(obj[k], str):
            s = obj[k].strip().lower()
            if s in ("in_stock", "instock", "available", "ok", "active", "enabled"):
                return True
            if s in (
                "out_of_stock",
                "outofstock",
                "unavailable",
                "disabled",
                "inactive",
                "paused",
            ):
                return False

    # quantity-like fields (fallback)
    for k in ("quantity", "stock", "stock_count", "available_stock", "inventory"):
        if k in obj and isinstance(obj[k], (int, float)):
            return obj[k] > 0

    return None


# -----------------------------
# Build external_id -> available map
# -----------------------------


def build_externalid_to_availability(token: str) -> Dict[str, bool]:
    mapping: Dict[str, bool] = {}

    store_products = list_store_products_all(token)

    for sp in store_products:
        spid = sp.get("id")
        if not spid:
            continue
        spid = str(spid).strip()

        detail = get_store_product_detail(token, spid)

        variants = (
            detail.get("sync_variants")
            or detail.get("variants")
            or detail.get("items")
            or []
        )
        if not isinstance(variants, list):
            variants = []

        for v in variants:
            if not isinstance(v, dict):
                continue
            ext = (
                v.get("external_id")
                or v.get("external_variant_id")
                or v.get("externalId")
            )
            if not ext:
                continue
            ext = str(ext).strip()

            av = infer_available(v)
            if av is None:
                for nested_key in ("warehouse", "availability", "stock", "variant"):
                    nested = v.get(nested_key)
                    if isinstance(nested, dict):
                        av = infer_available(nested)
                        if av is not None:
                            break

            if av is None:
                # safest: unknown => False (prevents selling something Printful may reject)
                av = False

            mapping[ext] = bool(av)

        time.sleep(0.08)

    return mapping


# -----------------------------
# Apply to products.json
# -----------------------------


def is_physical_product(p: Dict[str, Any]) -> bool:
    return str(p.get("product-type") or "").strip().lower() == "physical"


def apply_availability(
    products: List[Dict[str, Any]],
    ext_to_av: Dict[str, bool],
    default_missing: str,
) -> Dict[str, int]:
    stats = {
        "products_total": len(products),
        "physical_products": 0,
        "variants_seen": 0,
        "variants_with_id": 0,
        "variants_matched": 0,
        "available_set": 0,
        "available_unchanged": 0,
        "missing_in_printful": 0,
        "missing_action_keep": 0,
        "missing_action_set_false": 0,
        "missing_action_set_true": 0,
    }

    for p in products:
        if not is_physical_product(p):
            continue
        stats["physical_products"] += 1

        vars_ = p.get("varients") or []
        if not isinstance(vars_, list):
            continue

        for v in vars_:
            if not isinstance(v, dict):
                continue
            stats["variants_seen"] += 1

            vid = str(v.get("id") or "").strip()
            if not vid:
                continue
            stats["variants_with_id"] += 1

            if vid in ext_to_av:
                stats["variants_matched"] += 1
                new_av = bool(ext_to_av[vid])
                old_av = v.get("available", None)
                if old_av is None or bool(old_av) != new_av:
                    v["available"] = new_av
                    stats["available_set"] += 1
                else:
                    # ensure it's a real bool
                    v["available"] = new_av
                    stats["available_unchanged"] += 1
            else:
                stats["missing_in_printful"] += 1
                if default_missing == "keep":
                    stats["missing_action_keep"] += 1
                elif default_missing == "true":
                    v["available"] = True
                    stats["missing_action_set_true"] += 1
                else:
                    v["available"] = False
                    stats["missing_action_set_false"] += 1

    return stats


# -----------------------------
# CLI
# -----------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="Path to products.json")
    ap.add_argument(
        "--out", default=None, help="Output path (default: <in>_with_available.json)"
    )
    ap.add_argument("--write-inplace", action="store_true", help="Overwrite input file")
    ap.add_argument(
        "--default-missing",
        choices=["false", "keep", "true"],
        default="false",
        help="If a local variant id isn't found in Printful map: set available=false (default), keep, or set true.",
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

    print("Building availability map from Printful (external_id -> available)...")
    ext_to_av = build_externalid_to_availability(token)
    print(f"Mapped external_ids: {len(ext_to_av)}")

    print("Applying availability to products.json (physical only)...")
    stats = apply_availability(products, ext_to_av, args.default_missing)

    out_path = (
        args.in_path
        if args.write_inplace
        else (args.out or args.in_path.replace(".json", "_with_available.json"))
    )
    write_json(out_path, products)

    print("\n--- Done ---")
    for k in sorted(stats.keys()):
        print(f"{k}: {stats[k]}")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
