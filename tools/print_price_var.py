#!/usr/bin/env python3
"""
Init/Update variant prices in products.json using Printful as source of truth.

Match rule:
  Printful sync_variant.external_id  ==  products.json varients[].id  (Square variation ID)

Edits:
  - Updates ONLY varients[].price (number) for PHYSICAL products
  - Optionally writes varients[].currency (string) if found

What "price" means here:
  - We prefer Printful's *retail* price for the synced store variant (when present)
  - Fallback to other common price fields if retail isn't present

Env:
  PRINTFUL_ACCESS_TOKEN

Usage (Windows CMD):
  set PRINTFUL_ACCESS_TOKEN=YOUR_TOKEN
  python tools\\printful_init_prices.py --in tools\\data\\products.json --write-inplace

Options:
  --default-missing keep    keep existing price if variant id not found in Printful map
  --default-missing clear   remove price if not found (default)
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


def to_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        s = x.strip().replace("$", "").replace(",", "")
        try:
            return float(s)
        except Exception:
            return None
    return None


# -----------------------------
# Printful HTTP
# -----------------------------


def pf_headers(token: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "Content-Type": "application/json",
        "User-Agent": "unlim8ted-printful-init-prices/1.0",
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
# Price inference (defensive)
# -----------------------------


def infer_price_and_currency(
    v: Dict[str, Any],
) -> Tuple[Optional[float], Optional[str]]:
    """
    Tries common Printful fields for synced variant pricing.

    Priority (best guess):
      1) retail_price (+ currency)
      2) retail_price.value / retail_price.amount (+ currency)
      3) price / retail_price (string/number)
      4) variant.price / variant.retail_price
    """
    currency = None

    # direct currency fields (common patterns)
    for ck in (
        "currency",
        "retail_currency",
        "retail_price_currency",
        "store_currency",
    ):
        cv = v.get(ck)
        if isinstance(cv, str) and cv.strip():
            currency = cv.strip().upper()
            break

    # 1) direct retail_price
    if "retail_price" in v:
        rp = v.get("retail_price")
        if isinstance(rp, dict):
            # possible shapes: {value: "19.99", currency: "USD"} or {amount: 1999, currency: "USD"}
            if not currency:
                c2 = rp.get("currency") or rp.get("curr")
                if isinstance(c2, str) and c2.strip():
                    currency = c2.strip().upper()

            val = rp.get("value")
            amt = rp.get("amount")
            f = to_float(val if val is not None else amt)
            # if amount looks like cents (integer >= 100), you can tweak here,
            # but we won't assume without strong evidence.
            if f is not None:
                return f, currency
        else:
            f = to_float(rp)
            if f is not None:
                return f, currency

    # 2) other likely retail fields
    for k in ("retail", "store_price", "retailPrice", "retail_price_value"):
        if k in v:
            f = to_float(v.get(k))
            if f is not None:
                return f, currency

    # 3) generic price fields
    for k in ("price", "variant_price", "sync_variant_price"):
        if k in v:
            f = to_float(v.get(k))
            if f is not None:
                return f, currency

    # 4) nested variant
    nested = v.get("variant")
    if isinstance(nested, dict):
        p, c = infer_price_and_currency(nested)
        return p, c or currency

    return None, currency


# -----------------------------
# Build external_id -> price map
# -----------------------------


def build_externalid_to_price(token: str) -> Dict[str, Tuple[float, Optional[str]]]:
    """
    Returns mapping:
      external_id (Square variation id) -> (price_float, currency_or_none)
    """
    mapping: Dict[str, Tuple[float, Optional[str]]] = {}

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

            price, cur = infer_price_and_currency(v)
            if price is None:
                # try nested containers where some stores keep pricing
                for nk in ("pricing", "prices", "retail"):
                    nested = v.get(nk)
                    if isinstance(nested, dict):
                        price2, cur2 = infer_price_and_currency(nested)
                        if price2 is not None:
                            price, cur = price2, cur2 or cur
                            break

            if price is None:
                continue

            mapping[ext] = (float(price), cur)

        time.sleep(0.08)

    return mapping


# -----------------------------
# Apply to products.json
# -----------------------------


def apply_prices(
    products: List[Dict[str, Any]],
    ext_to_price: Dict[str, Tuple[float, Optional[str]]],
    default_missing: str,
) -> Dict[str, int]:
    stats = {
        "products_total": len(products),
        "physical_products": 0,
        "variants_seen": 0,
        "variants_with_id": 0,
        "variants_matched": 0,
        "prices_set": 0,
        "prices_unchanged": 0,
        "currency_set": 0,
        "missing_in_printful": 0,
        "missing_action_keep": 0,
        "missing_action_clear": 0,
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

            if vid in ext_to_price:
                stats["variants_matched"] += 1
                new_price, cur = ext_to_price[vid]

                old_price = to_float(v.get("price"))
                if old_price is None or abs(old_price - new_price) > 1e-9:
                    v["price"] = new_price
                    stats["prices_set"] += 1
                else:
                    v["price"] = new_price
                    stats["prices_unchanged"] += 1

                if cur:
                    if v.get("currency") != cur:
                        v["currency"] = cur
                        stats["currency_set"] += 1
                # if no cur returned, we don't overwrite currency
            else:
                stats["missing_in_printful"] += 1
                if default_missing == "keep":
                    stats["missing_action_keep"] += 1
                else:
                    v.pop("price", None)
                    v.pop(
                        "currency", None
                    )  # optional; comment out if you want to keep currency
                    stats["missing_action_clear"] += 1

    return stats


# -----------------------------
# CLI
# -----------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="Path to products.json")
    ap.add_argument(
        "--out", default=None, help="Output path (default: <in>_with_prices.json)"
    )
    ap.add_argument("--write-inplace", action="store_true", help="Overwrite input file")
    ap.add_argument(
        "--default-missing",
        choices=["keep", "clear"],
        default="clear",
        help="If a local variant id isn't found in Printful map: keep existing price or clear it (default).",
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

    print("Building price map from Printful (external_id -> price)...")
    ext_to_price = build_externalid_to_price(token)
    print(f"Mapped external_ids with price: {len(ext_to_price)}")

    print("Applying prices to products.json (physical only)...")
    stats = apply_prices(products, ext_to_price, args.default_missing)

    out_path = (
        args.in_path
        if args.write_inplace
        else (args.out or args.in_path.replace(".json", "_with_prices.json"))
    )
    write_json(out_path, products)

    print("\n--- Done ---")
    for k in sorted(stats.keys()):
        print(f"{k}: {stats[k]}")
    print(f"Wrote: {out_path}")
    print("\nNote: If your Printful store variants do NOT include retail price fields,")
    print(
        "we can switch to using the Catalog Variant Prices endpoint (v2) but that requires catalog_variant_id mapping."
    )


if __name__ == "__main__":
    main()
