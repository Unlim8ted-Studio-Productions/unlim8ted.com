#!/usr/bin/env python3
"""
REBUILD variants for ONE product from Printful, but ONLY touch the fields you approved.

What it does (for ONE product):
✅ Replaces the entire `varients` array with fresh data from Printful store product detail.
✅ Updates product `image`, `images`, `additional_images` (best-effort)
✅ Updates each variant:
   - id            (Printful store variant id)
   - available     (bool inferred)
   - optionParts   (parsed from Printful variant name; e.g. "Red / XL" => ["Red","XL"])
   - variantLabel  ("Red / XL")
   - image         (best-effort)
   - price         (float)
   - currency      ("USD" etc)

❌ Does NOT touch: name, description, details, notes, product-type, printful_id, videos, etc.

Env:
  PRINTFUL_ACCESS_TOKEN

Usage (CMD):
  set PRINTFUL_ACCESS_TOKEN=YOUR_TOKEN
  python tools\\printful_rebuild_variants_one.py --in tools\\data\\products.json --write-inplace --product-id unlim8ted-organic-ribbed-beanie

Or:
  python tools\\printful_rebuild_variants_one.py --in tools\\data\\products.json --write-inplace --printful-id 411558074
"""

import os, json, argparse
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


def s(x: Any) -> str:
    return "" if x is None else str(x)


def is_physical_product(p: Dict[str, Any]) -> bool:
    return s(p.get("product-type")).strip().lower() == "physical"


def norm_url(x: Any) -> Optional[str]:
    if not isinstance(x, str):
        return None
    u = x.strip()
    if u.startswith("http://") or u.startswith("https://"):
        return u
    return None


def dedupe(urls: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
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
        t = x.strip().replace("$", "").replace(",", "")
        try:
            return float(t)
        except Exception:
            return None
    return None


# ---------------- Printful HTTP ----------------


def pf_headers(token: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "Content-Type": "application/json",
        "User-Agent": "unlim8ted-printful-rebuild-variants/1.0",
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
    # classic Printful format: {"code":200,"result":{...}}
    if isinstance(j, dict) and isinstance(j.get("result"), dict):
        return j["result"]
    return j if isinstance(j, dict) else {}


# ---------------- Inference ----------------


def infer_available(obj: Dict[str, Any]) -> Optional[bool]:
    for k in ("available", "is_available", "in_stock", "is_in_stock", "sellable"):
        if k in obj and isinstance(obj[k], bool):
            return bool(obj[k])
    for k in ("availability_status", "availability", "status", "stock_status"):
        if k in obj and isinstance(obj[k], str):
            st = obj[k].strip().lower()
            if st in ("in_stock", "instock", "available", "ok", "active", "enabled"):
                return True
            if st in (
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
    for ck in ("currency", "retail_currency", "store_currency"):
        cv = v.get(ck)
        if isinstance(cv, str) and cv.strip():
            currency = cv.strip().upper()
            break

    rp = v.get("retail_price")
    if isinstance(rp, dict):
        if not currency:
            c2 = rp.get("currency")
            if isinstance(c2, str) and c2.strip():
                currency = c2.strip().upper()
        val = rp.get("value") if rp.get("value") is not None else rp.get("amount")
        f = to_float(val)
        if f is not None:
            return f, currency
    elif rp is not None:
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
        sc = 0
        if "preview" in ul:
            sc += 40
        if "mockup" in ul:
            sc += 30
        if "thumbnail" in ul or "_thumb" in ul:
            sc += 15
        if "printfile-preview" in ul:
            sc -= 5
        return sc

    return sorted(urls, key=score, reverse=True)[0]


def parse_option_parts_from_variant_name(
    full_name: str, product_name_hint: Optional[str]
) -> List[str]:
    """
    Your Printful display names look like:
      "Life of a Meatball Unisex T-Shirt / White / XS"
    We want optionParts: ["White","XS"]
    Strategy:
      - Split by " / "
      - If first segment matches product name (or contains it), drop first segment
      - Return remaining segments
    """
    parts = [p.strip() for p in full_name.split(" / ") if p.strip()]
    if not parts:
        return []
    if product_name_hint:
        pnh = product_name_hint.strip().lower()
        if parts and pnh and parts[0].lower() == pnh:
            return parts[1:]
        # sometimes full product name is embedded; keep simple & safe:
        if parts and pnh and pnh in parts[0].lower() and len(parts) > 1:
            return parts[1:]
    # if it doesn't look prefixed, just return all (better than empty)
    return parts


# ---------------- Core ----------------


def find_product_index(
    products: List[Dict[str, Any]],
    product_id: Optional[str],
    printful_id: Optional[str],
) -> int:
    if product_id:
        for i, p in enumerate(products):
            if s(p.get("id")).strip() == product_id:
                return i
    if printful_id:
        for i, p in enumerate(products):
            if s(p.get("printful_id")).strip() == s(printful_id).strip():
                return i
    raise SystemExit(
        "Product not found. Use --product-id or --printful-id that exists in products.json."
    )


def rebuild_variants_for_one_product(
    prod: Dict[str, Any], token: str
) -> Dict[str, int]:
    if not is_physical_product(prod):
        raise SystemExit("Selected product is not product-type: physical")

    pfid = s(prod.get("printful_id")).strip()
    if not pfid:
        raise SystemExit("Selected product has no printful_id")

    detail = pf_get(token, f"/store/products/{pfid}")

    # --- product images (only fields you approved) ---
    prod_urls = dedupe([u for u in extract_urls(detail) if u])
    best = choose_best(prod_urls)
    if best:
        prod["image"] = best
    if prod_urls:
        prod["images"] = prod_urls

    # best-effort additional_images: prioritize non Printful CDN if present
    addl = [u for u in prod_urls if ("items-images-production" in u or "s3." in u)]
    if addl:
        prod["additional_images"] = dedupe(addl)

    # --- rebuild variants ---
    pf_variants = (
        detail.get("sync_variants")
        or detail.get("variants")
        or detail.get("items")
        or []
    )
    if not isinstance(pf_variants, list):
        pf_variants = []

    # use product name from your json as hint for parsing optionParts
    name_hint = s(prod.get("name")).strip() or None

    new_vars: List[Dict[str, Any]] = []
    seen_ids = set()

    for sv in pf_variants:
        if not isinstance(sv, dict):
            continue

        vid = sv.get("id")
        if vid is None:
            continue
        vid_s = s(vid).strip()
        if not vid_s or vid_s in seen_ids:
            continue
        seen_ids.add(vid_s)

        av = infer_available(sv)
        if av is None:
            # if unknown, default to True (safer for storefront?) — but you can change to False
            av = True

        price, cur = infer_price_and_currency(sv)
        if cur is None:
            cur = "USD"  # sensible default; keep if Printful doesn't send

        # variant label + optionParts
        # prefer sv["name"] if present, else try external name fields
        full_name = s(
            sv.get("name") or sv.get("variant_name") or sv.get("title")
        ).strip()
        opt_parts: List[str] = []

        if full_name:
            opt_parts = parse_option_parts_from_variant_name(full_name, name_hint)

        variant_label = " / ".join(opt_parts) if opt_parts else ""

        # variant image best-effort
        vurls = dedupe([u for u in extract_urls(sv) if u])
        vbest = choose_best(vurls)

        vobj: Dict[str, Any] = {
            "id": vid_s,
            "available": bool(av),
        }

        if opt_parts:
            vobj["optionParts"] = opt_parts
        if variant_label:
            vobj["variantLabel"] = variant_label
        if vbest:
            vobj["image"] = vbest
        if price is not None:
            vobj["price"] = float(price)
        if cur:
            vobj["currency"] = str(cur).upper()

        new_vars.append(vobj)

    # Replace the entire variants list
    prod["varients"] = new_vars

    return {
        "pf_variants_total": len(pf_variants),
        "variants_written": len(new_vars),
        "product_images_found": len(prod_urls),
    }


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
        f"Rebuilding varients for product index={idx} id={prod.get('id')} printful_id={prod.get('printful_id')}"
    )

    stats = rebuild_variants_for_one_product(prod, token)

    out_path = (
        args.in_path
        if args.write_inplace
        else (args.out or args.in_path.replace(".json", "_rebuilt_one.json"))
    )
    write_json(out_path, products)

    print("\n--- Done ---")
    for k, v in stats.items():
        print(f"{k}: {v}")
    print(f"Wrote: {out_path}")
    print(
        "\nRebuilt ONLY: product images + varients (id/available/price/currency/image/variantLabel/optionParts)."
    )
    print("Did NOT touch: name/description/details/notes and other product fields.")


if __name__ == "__main__":
    main()
