#!/usr/bin/env python3
"""
Update ONE product object in products.json from Printful, but ONLY these fields:

PRODUCT-level (overwrite):
  - image
  - images
  - additional_images (best-effort from discovered URLs)

VARIANT-level (for each item in varients[], matched by id):
  - available
  - price
  - currency
  - image
  - variantLabel (derived from optionParts -> "Red / XL" style)

DO NOT touch:
  - name, description, details, notes (or any other copy fields)
  - product-type, printful_id, id
  - additional_videos, video, etc. (left as-is unless you edit manually)

Match rules:
  - Select product by --product-id (your slug) OR --printful-id
  - Match variants by:
      Printful sync_variant.id  (store variant id)  ==  products.json varients[].id
    (Your JSON shows variant ids like "6956aecd234d33" which look like Printful store variant IDs.)

Env:
  PRINTFUL_ACCESS_TOKEN

Usage:
  set PRINTFUL_ACCESS_TOKEN=YOUR_TOKEN
  python tools\\printful_update_one_fields.py --in tools\\data\\products.json --write-inplace --product-id unlim8ted-organic-ribbed-beanie
"""

import os, json, argparse, time
from typing import Any, Dict, List, Optional, Tuple
import requests

PRINTFUL_BASE = "https://api.printful.com"


# ---------------- IO ----------------


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


# ---------------- Helpers ----------------


def is_physical_product(p: Dict[str, Any]) -> bool:
    return str(p.get("product-type") or "").strip().lower() == "physical"


def norm_url(x: Any) -> Optional[str]:
    if not isinstance(x, str):
        return None
    u = x.strip()
    if not u:
        return None
    if u.startswith("https://") or u.startswith("http://"):
        return u
    return None


def dedupe(urls: List[str]) -> List[str]:
    seen = set()
    out = []
    for u in urls:
        if u and u not in seen:
            seen.add(u)
            out.append(u)
    return out


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


def build_variant_label_from_optionparts(option_parts: Any) -> Optional[str]:
    if not isinstance(option_parts, list):
        return None
    parts = []
    for p in option_parts:
        ps = str(p).strip()
        if ps:
            parts.append(ps)
    if not parts:
        return None
    return " / ".join(parts)  # "Red / XL"


# ---------------- Printful HTTP ----------------


def pf_headers(token: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "Content-Type": "application/json",
        "User-Agent": "unlim8ted-printful-update-one-fields/1.0",
    }


def pf_get(token: str, path: str, params=None) -> Dict[str, Any]:
    url = f"{PRINTFUL_BASE}{path}"
    r = requests.get(url, headers=pf_headers(token), params=params, timeout=60)
    try:
        j = r.json()
    except Exception:
        j = None
    if r.status_code != 200:
        raise SystemExit(
            f"GET {path} failed HTTP {r.status_code}\n"
            f"{json.dumps(j, indent=2) if isinstance(j,(dict,list)) else r.text[:1200]}"
        )
    if isinstance(j, dict) and isinstance(j.get("result"), dict):
        return j["result"]
    return j if isinstance(j, dict) else {}


# ---------------- Inference (availability, price, images) ----------------


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
                "inactive",
                "disabled",
                "paused",
            ):
                return False
    for k in ("quantity", "stock", "stock_count", "available_stock", "inventory"):
        if k in obj and isinstance(obj[k], (int, float)):
            return obj[k] > 0
    return None


def infer_price_and_currency(
    v: Dict[str, Any],
) -> Tuple[Optional[float], Optional[str]]:
    currency = None
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

    if "retail_price" in v:
        rp = v.get("retail_price")
        if isinstance(rp, dict):
            if not currency:
                c2 = rp.get("currency") or rp.get("curr")
                if isinstance(c2, str) and c2.strip():
                    currency = c2.strip().upper()
            val = rp.get("value")
            amt = rp.get("amount")
            f = to_float(val if val is not None else amt)
            if f is not None:
                return f, currency
        else:
            f = to_float(rp)
            if f is not None:
                return f, currency

    for k in ("price", "variant_price", "store_price"):
        if k in v:
            f = to_float(v.get(k))
            if f is not None:
                return f, currency

    nested = v.get("variant")
    if isinstance(nested, dict):
        p2, c2 = infer_price_and_currency(nested)
        return p2, c2 or currency

    return None, currency


URL_KEYS = ("thumbnail_url", "mockup_url", "preview_url", "image_url", "url")


def extract_urls(obj: Any) -> List[str]:
    out: List[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            kl = str(k).lower()
            if kl in URL_KEYS or kl.endswith("_url"):
                u = norm_url(v)
                if u:
                    out.append(u)
            out.extend(extract_urls(v))
    elif isinstance(obj, list):
        for x in obj:
            out.extend(extract_urls(x))
    return out


def choose_best(urls: List[str]) -> Optional[str]:
    if not urls:
        return None

    def score(u: str) -> int:
        ul = u.lower()
        s = 0
        if "preview" in ul:
            s += 40
        if "mockup" in ul:
            s += 35
        if "thumbnail" in ul or "_thumb" in ul:
            s += 20
        return s

    return sorted(urls, key=score, reverse=True)[0]


# ---------------- Core update ----------------


def find_product_index(
    products: List[Dict[str, Any]],
    product_id: Optional[str],
    printful_id: Optional[str],
) -> int:
    if product_id:
        for i, p in enumerate(products):
            if str(p.get("id") or "").strip() == product_id:
                return i
    if printful_id:
        for i, p in enumerate(products):
            if str(p.get("printful_id") or "").strip() == str(printful_id).strip():
                return i
    raise SystemExit(
        "Product not found. Use --product-id or --printful-id that exists in products.json"
    )


def update_one_product_fields(prod: Dict[str, Any], token: str) -> Dict[str, int]:
    if not is_physical_product(prod):
        raise SystemExit("Selected product is not product-type: physical")

    pfid = str(prod.get("printful_id") or "").strip()
    if not pfid:
        raise SystemExit("Selected product has no printful_id")

    # Pull Printful store product detail (v1)
    detail = pf_get(token, f"/store/products/{pfid}")

    # PRODUCT images
    all_prod_urls = dedupe([u for u in extract_urls(detail) if u])
    best = choose_best(all_prod_urls)
    if best:
        prod["image"] = best
    if all_prod_urls:
        prod["images"] = all_prod_urls

    # additional_images best-effort: prefer non Printful CDN if present
    addl = [u for u in all_prod_urls if ("items-images-production" in u or "s3." in u)]
    if addl:
        prod["additional_images"] = dedupe(addl)

    # VARIANTS: your ids look like Printful store variant ids, so match on sync_variant.id
    pf_variants = (
        detail.get("sync_variants")
        or detail.get("variants")
        or detail.get("items")
        or []
    )
    if not isinstance(pf_variants, list):
        pf_variants = []

    pf_by_id: Dict[str, Dict[str, Any]] = {}
    for sv in pf_variants:
        if isinstance(sv, dict) and sv.get("id") is not None:
            pf_by_id[str(sv["id"]).strip()] = sv

    stats = {
        "variants_seen": 0,
        "variants_matched": 0,
        "availability_set": 0,
        "price_set": 0,
        "currency_set": 0,
        "variant_image_set": 0,
        "variant_label_set": 0,
    }

    local_vars = prod.get("varients") or []
    if not isinstance(local_vars, list):
        return stats

    for lv in local_vars:
        if not isinstance(lv, dict):
            continue

        stats["variants_seen"] += 1
        vid = str(lv.get("id") or "").strip()
        if not vid:
            continue

        sv = pf_by_id.get(vid)
        if not sv:
            continue

        stats["variants_matched"] += 1

        # available
        av = infer_available(sv)
        if av is not None:
            lv["available"] = bool(av)
            stats["availability_set"] += 1

        # price/currency
        price, cur = infer_price_and_currency(sv)
        if price is not None:
            lv["price"] = float(price)
            stats["price_set"] += 1
        if cur:
            lv["currency"] = cur
            stats["currency_set"] += 1

        # variant image
        vurls = dedupe([u for u in extract_urls(sv) if u])
        vbest = choose_best(vurls)
        if vbest:
            lv["image"] = vbest
            stats["variant_image_set"] += 1

        # variantLabel from optionParts -> "Red / XL"
        new_label = build_variant_label_from_optionparts(lv.get("optionParts"))
        if new_label:
            lv["variantLabel"] = new_label
            stats["variant_label_set"] += 1

    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--write-inplace", action="store_true")
    ap.add_argument("--product-id", default=None)
    ap.add_argument("--printful-id", default=None)
    ap.add_argument("--token", default=None)
    args = ap.parse_args()

    token = (args.token or os.environ.get("PRINTFUL_ACCESS_TOKEN") or "").strip()
    if not token:
        raise SystemExit("Missing PRINTFUL_ACCESS_TOKEN env var (or pass --token).")

    products = load_json(args.in_path)
    if not isinstance(products, list):
        raise SystemExit("products.json must be a JSON ARRAY")

    idx = find_product_index(products, args.product_id, args.printful_id)
    prod = products[idx]

    print(
        f"Updating product index={idx} id={prod.get('id')} printful_id={prod.get('printful_id')}"
    )

    stats = update_one_product_fields(prod, token)

    out_path = (
        args.in_path
        if args.write_inplace
        else (args.out or args.in_path.replace(".json", "_one_fields_updated.json"))
    )
    write_json(out_path, products)

    print("\n--- Done ---")
    for k in sorted(stats.keys()):
        print(f"{k}: {stats[k]}")
    print(f"Wrote: {out_path}")
    print(
        "\nUpdated ONLY: product image(s) + variant image/price/currency/available/variantLabel."
    )


if __name__ == "__main__":
    main()
