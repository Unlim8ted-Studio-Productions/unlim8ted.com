#!/usr/bin/env python3
"""
Printful → products.json sync (NO GUI)

What this script does:
- Uses Printful as source of truth for PHYSICAL catalog + variants.
- Matches Printful products to your products.json by slugifying the Printful product name:
    e.g. "Life of a Meatball Unisex T-Shirt" -> "life-of-a-meatball-unisex-t-shirt"
  and comparing to your local product ids:
    e.g. "unlim8ted-bubble-free-stickers"

- For PHYSICAL products in products.json:
    • If no matching Printful product => REMOVE from products.json (physical only)
    • If match => REPLACE its variants from Printful variants (and set available bool)

- For Printful products not in products.json => ADD to products.json as new PHYSICAL products.

- Removes "buy links" fields from PHYSICAL products + their variants.

Requires:
  PRINTFUL_ACCESS_TOKEN env var

Usage (Windows CMD):
  set PRINTFUL_ACCESS_TOKEN=YOUR_TOKEN
  python tools\printful_sync_products.py --in tools\data\products.json --write-inplace

Notes:
- This expects Printful variant "name" format like:
    "Life of a Meatball Unisex T-Shirt / White / XS"
  so we can split product / color / size.
- Variant id set to Printful sync_variant.external_id (Square variation id) if present.
- Availability inferred from common fields: available/in_stock/status/quantity etc.
"""

import os
import re
import json
import time
import argparse
from typing import Any, Dict, List, Optional, Tuple
import requests

PRINTFUL_BASE = "https://api.printful.com"


# -----------------------------
# Utility
# -----------------------------


def slugify(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[’']", "", s)
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def is_physical_product(p: Dict[str, Any]) -> bool:
    return str(p.get("product-type") or "").strip().lower() == "physical"


# -----------------------------
# Printful HTTP
# -----------------------------


def pf_headers(token: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "Content-Type": "application/json",
        "User-Agent": "unlim8ted-printful-sync/1.0",
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
    for k in ("available", "is_available", "in_stock", "is_in_stock", "sellable"):
        if k in obj and isinstance(obj[k], bool):
            return bool(obj[k])

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

    for k in ("quantity", "stock", "stock_count", "available_stock", "inventory"):
        if k in obj and isinstance(obj[k], (int, float)):
            return obj[k] > 0

    return None


# -----------------------------
# Parsing Printful variant names
# -----------------------------


def parse_variant_display_name(v: Dict[str, Any]) -> str:
    for k in ("name", "variant_name", "title"):
        s = v.get(k)
        if isinstance(s, str) and s.strip():
            return s.strip()
    return ""


def split_pf_variant_name(full: str) -> Tuple[str, str, str]:
    """
    "Life of a Meatball Unisex T-Shirt / White / XS" -> (product_name, color, size)
    If it doesn't match, returns best-effort.
    """
    parts = [p.strip() for p in (full or "").split("/") if p.strip()]
    if len(parts) >= 3:
        return parts[0], parts[1], parts[2]
    if len(parts) == 2:
        return parts[0], parts[1], ""
    if len(parts) == 1:
        return parts[0], "", ""
    return "", "", ""


# -----------------------------
# Remove "buy links"
# -----------------------------

BUY_LINK_KEYS = {
    "buyLink",
    "buy_link",
    "buyUrl",
    "buy_url",
    "squareUrl",
    "square_url",
    "paymentLink",
    "payment_link",
    "checkoutUrl",
    "checkout_url",
    "url",
    "link",
    "square",
    "squareLink",
}


def strip_buy_links_from_obj(d: Dict[str, Any]) -> int:
    """
    Removes known buy-link keys and any key that *looks* like a checkout link.
    Keeps image/imageUrl/downloadUrl/accessUrl/etc.
    """
    removed = 0
    keys = list(d.keys())
    for k in keys:
        kl = k.lower()

        # explicit known keys
        if k in BUY_LINK_KEYS or kl in {x.lower() for x in BUY_LINK_KEYS}:
            d.pop(k, None)
            removed += 1
            continue

        # remove fields containing "square" or "checkout" or "payment" (but not images)
        if ("square" in kl or "checkout" in kl or "payment" in kl) and (
            "image" not in kl
        ):
            d.pop(k, None)
            removed += 1
            continue

        # remove generic "*Url" fields if they look like buy links (but not imageUrl/accessUrl/downloadUrl)
        if (
            kl.endswith("url")
            and ("image" not in kl)
            and ("download" not in kl)
            and ("access" not in kl)
        ):
            # if you want to be stricter: only delete if value contains "square.link" etc.
            val = d.get(k)
            if isinstance(val, str) and val.strip():
                d.pop(k, None)
                removed += 1
                continue

    return removed


# -----------------------------
# Build Printful catalog (slug -> data)
# -----------------------------


def pf_product_name_from_list_item(sp: Dict[str, Any]) -> str:
    for k in ("name", "title", "product_name"):
        v = sp.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def build_printful_catalog(token: str) -> Dict[str, Dict[str, Any]]:
    """
    Returns:
      slug_id -> {
        "store_product_id": "...",
        "product_name": "...",
        "variants": [
          {"external_id": "...", "color": "...", "size": "...", "available": bool, "variantLabel": "...", "optionParts":[...]}
        ]
      }
    """
    catalog: Dict[str, Dict[str, Any]] = {}

    store_products = list_store_products_all(token)
    for sp in store_products:
        spid = str(sp.get("id") or "").strip()
        if not spid:
            continue

        # Pull detail for variants
        detail = get_store_product_detail(token, spid)

        # Prefer a stable product name from variants (more reliable than list name in some cases)
        list_name = pf_product_name_from_list_item(sp)
        product_name = list_name

        pf_vars = (
            detail.get("sync_variants")
            or detail.get("variants")
            or detail.get("items")
            or []
        )
        if not isinstance(pf_vars, list):
            pf_vars = []

        variants_out: List[Dict[str, Any]] = []

        for v in pf_vars:
            if not isinstance(v, dict):
                continue

            full = parse_variant_display_name(v)
            pn, color, size = split_pf_variant_name(full)
            if pn and not product_name:
                product_name = pn

            ext = (
                v.get("external_id")
                or v.get("external_variant_id")
                or v.get("externalId")
            )
            ext = str(ext).strip() if ext else ""

            av = infer_available(v)
            if av is None:
                for nested_key in ("warehouse", "availability", "stock", "variant"):
                    nested = v.get(nested_key)
                    if isinstance(nested, dict):
                        av = infer_available(nested)
                        if av is not None:
                            break
            if av is None:
                # unknown -> default True? safer to default False so you don't sell something Printful can't fulfill
                av = False

            option_parts = []
            if color:
                option_parts.append(color)
            if size:
                option_parts.append(size)

            variant_label = (
                ", ".join([x for x in option_parts if x]) if option_parts else ""
            )

            variants_out.append(
                {
                    "external_id": ext,
                    "color": color,
                    "size": size,
                    "available": bool(av),
                    "variantLabel": variant_label,
                    "optionParts": option_parts,
                    "fullName": full,
                }
            )

        # Decide slug id:
        # 1) use product_name from list item if present
        # 2) else use first variant's product name portion
        if not product_name and variants_out:
            pn, _, _ = split_pf_variant_name(variants_out[0].get("fullName", ""))
            product_name = pn or ""

        slug = slugify(product_name)
        if not slug:
            # fallback: make a deterministic slug based on store product id
            slug = f"printful-{slugify(spid)}"

        catalog[slug] = {
            "store_product_id": spid,
            "product_name": product_name or slug,
            "variants": variants_out,
        }

        time.sleep(0.08)

    return catalog


# -----------------------------
# Sync products.json
# -----------------------------


def build_local_variant_from_pf(v: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    ext = str(v.get("external_id") or "").strip()
    # if no external_id, we can't map to Square; skip variant
    if not ext:
        return None

    out: Dict[str, Any] = {
        "id": ext,
        "available": bool(v.get("available", False)),
    }
    # keep your local fields consistent
    if v.get("variantLabel"):
        out["variantLabel"] = v["variantLabel"]
    if isinstance(v.get("optionParts"), list) and v["optionParts"]:
        out["optionParts"] = v["optionParts"]
    return out


def sync_products(
    local: List[Dict[str, Any]], pf_catalog: Dict[str, Dict[str, Any]]
) -> Dict[str, int]:
    """
    Mutates local list in-place.

    Rules:
    - Only physical products are managed.
    - Remove physical products with no match in Printful.
    - Add physical products from Printful not present locally.
    - For matched physical products, replace variants with Printful-derived variants.
    - Remove buy links from physical products + variants.

    Returns stats dict.
    """
    stats = {
        "local_total": len(local),
        "local_physical_before": 0,
        "removed_physical_no_match": 0,
        "matched_physical": 0,
        "added_from_printful": 0,
        "variants_written": 0,
        "variants_skipped_no_external_id": 0,
        "buy_link_fields_removed": 0,
    }

    # index local by id
    local_by_id: Dict[str, Dict[str, Any]] = {}
    for p in local:
        pid = str(p.get("id") or "").strip()
        if pid:
            local_by_id[pid] = p

    # Remove unmatched physical products
    keep: List[Dict[str, Any]] = []
    for p in local:
        if not is_physical_product(p):
            keep.append(p)
            continue

        stats["local_physical_before"] += 1
        pid = str(p.get("id") or "").strip()

        if pid in pf_catalog:
            keep.append(p)
        else:
            stats["removed_physical_no_match"] += 1

    # replace list in-place
    local[:] = keep

    # refresh local_by_id after removals
    local_by_id = {
        str(p.get("id") or "").strip(): p
        for p in local
        if str(p.get("id") or "").strip()
    }

    # Update matched + add missing
    for slug_id, pf in pf_catalog.items():
        if slug_id in local_by_id:
            p = local_by_id[slug_id]
            stats["matched_physical"] += 1

            # enforce required fields
            p["product-type"] = "physical"
            p["name"] = p.get("name") or pf.get("product_name") or slug_id
            p["printful_id"] = pf["store_product_id"]

            # replace variants from Printful
            new_vars: List[Dict[str, Any]] = []
            for pv in pf.get("variants", []):
                lv = build_local_variant_from_pf(pv)
                if lv is None:
                    stats["variants_skipped_no_external_id"] += 1
                    continue
                new_vars.append(lv)

            p["varients"] = new_vars
            stats["variants_written"] += len(new_vars)

            # remove buy links fields from product + variants
            stats["buy_link_fields_removed"] += strip_buy_links_from_obj(p)
            for v in p.get("varients", []) or []:
                if isinstance(v, dict):
                    stats["buy_link_fields_removed"] += strip_buy_links_from_obj(v)

        else:
            # add new product from Printful
            new_p: Dict[str, Any] = {
                "id": slug_id,
                "name": pf.get("product_name") or slug_id,
                "product-type": "physical",
                "printful_id": pf["store_product_id"],
                "varients": [],
            }

            new_vars: List[Dict[str, Any]] = []
            for pv in pf.get("variants", []):
                lv = build_local_variant_from_pf(pv)
                if lv is None:
                    stats["variants_skipped_no_external_id"] += 1
                    continue
                new_vars.append(lv)
            new_p["varients"] = new_vars
            stats["variants_written"] += len(new_vars)

            stats["buy_link_fields_removed"] += strip_buy_links_from_obj(new_p)
            for v in new_p.get("varients", []) or []:
                if isinstance(v, dict):
                    stats["buy_link_fields_removed"] += strip_buy_links_from_obj(v)

            local.append(new_p)
            local_by_id[slug_id] = new_p
            stats["added_from_printful"] += 1

    return stats


# -----------------------------
# CLI
# -----------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="Path to products.json")
    ap.add_argument(
        "--out", default=None, help="Output path (default: <in>_synced.json)"
    )
    ap.add_argument("--write-inplace", action="store_true", help="Overwrite input file")
    ap.add_argument(
        "--token",
        default=None,
        help="Printful token (or use PRINTFUL_ACCESS_TOKEN env var)",
    )
    args = ap.parse_args()

    token = (args.token or os.environ.get("PRINTFUL_ACCESS_TOKEN") or "").strip()
    if not token:
        raise SystemExit("Missing PRINTFUL_ACCESS_TOKEN (set env var) or pass --token")

    local = load_json(args.in_path)
    if not isinstance(local, list):
        raise SystemExit("products.json must be a JSON array")

    print("Fetching Printful store products + variants...")
    pf_catalog = build_printful_catalog(token)
    print(f"Printful products discovered: {len(pf_catalog)}")

    print("Syncing products.json (physical only)...")
    stats = sync_products(local, pf_catalog)

    out_path = (
        args.in_path
        if args.write_inplace
        else (args.out or args.in_path.replace(".json", "_synced.json"))
    )
    write_json(out_path, local)

    print("\n--- Done ---")
    for k in sorted(stats.keys()):
        print(f"{k}: {stats[k]}")
    print(f"Wrote: {out_path}")
    print("\nIMPORTANT:")
    print(
        "- Any PHYSICAL product ids in products.json that don't match Printful slug(product_name) were removed."
    )
    print("- Any Printful products not present locally were added.")
    print(
        "- Variants without external_id were skipped (cannot map to Square variation id)."
    )


if __name__ == "__main__":
    main()
