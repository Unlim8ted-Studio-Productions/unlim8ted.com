import os
import json
import time
import uuid
from typing import Any, Dict, List, Optional

import requests


# ----------------------------
# Config
# ----------------------------
ACCESS_TOKEN = ""
if not ACCESS_TOKEN:
    raise SystemExit("Missing env var: SQUARE_ACCESS_TOKEN")
LOCATION_ID = ""
if not LOCATION_ID:
    raise SystemExit("Missing env var: SQUARE_LOCATION_ID (required for payment links)")

ENV = "production"
BASE_URL = (
    "https://connect.squareupsandbox.com"
    if ENV == "sandbox"
    else "https://connect.squareup.com"
)

SQUARE_VERSION = (os.getenv("SQUARE_VERSION") or "2025-01-23").strip()
REDIRECT_URL = (os.getenv("SQUARE_REDIRECT_URL") or "").strip() or None

SLEEP_BETWEEN_CALLS_SEC = float(os.getenv("SQUARE_SLEEP_SEC") or "0.25")

OUT_JSON = os.getenv("OUT_JSON") or "products_with_variant_buy_links.json"


# ----------------------------
# HTTP helpers
# ----------------------------
def headers() -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {ACCESS_TOKEN}",
        "Content-Type": "application/json",
        "Square-Version": SQUARE_VERSION,
    }


def square_get(path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    r = requests.get(f"{BASE_URL}{path}", headers=headers(), params=params, timeout=60)
    if not r.ok:
        raise RuntimeError(f"GET {path} failed: {r.status_code} {r.text}")
    return r.json() if r.text else {}


def square_post(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    r = requests.post(f"{BASE_URL}{path}", headers=headers(), json=payload, timeout=60)
    if not r.ok:
        raise RuntimeError(f"POST {path} failed: {r.status_code} {r.text}")
    return r.json() if r.text else {}


# ----------------------------
# Catalog loading
# ----------------------------
def list_catalog_all(types: List[str], limit: int = 200) -> List[Dict[str, Any]]:
    """
    Loads the full catalog for the requested types, using cursor pagination.
    """
    objects: List[Dict[str, Any]] = []
    cursor: Optional[str] = None

    while True:
        params = {"types": ",".join(types), "limit": str(limit)}
        if cursor:
            params["cursor"] = cursor

        data = square_get("/v2/catalog/list", params=params)
        page = data.get("objects") or []
        objects.extend(page)

        cursor = data.get("cursor")
        if not cursor:
            break

        time.sleep(SLEEP_BETWEEN_CALLS_SEC)

    return objects


def money_to_float(price_money: Optional[Dict[str, Any]]) -> Optional[float]:
    """
    Square money is in the smallest currency unit (e.g., cents).
    """
    if not price_money:
        return None
    amount = price_money.get("amount")
    if amount is None:
        return None
    try:
        return round(float(amount) / 100.0, 2)
    except Exception:
        return None


# ----------------------------
# Payment Link creation
# ----------------------------
def create_payment_link_for_variation(
    variation_id: str, item_name: str, variation_name: str
) -> str:
    """
    Creates a Square Payment Link that purchases exactly 1 of this catalog variation.
    Returns the URL.
    """

    payload = {
        "idempotency_key": str(uuid.uuid4()),
        "order": {
            "location_id": LOCATION_ID,
            "line_items": [
                {
                    "quantity": "1",
                    "catalog_object_id": variation_id,
                }
            ],
        },
    }

    if REDIRECT_URL:
        payload["checkout_options"] = {"redirect_url": REDIRECT_URL}

    # Optional nice-to-have metadata (safe to remove)
    payload["description"] = f"{item_name} — {variation_name}"

    data = square_post("/v2/online-checkout/payment-links", payload)
    pl = data.get("payment_link") or {}
    url = pl.get("url")
    if not url:
        raise RuntimeError(
            f"Payment link created but URL missing for variation {variation_id}"
        )
    return url


# ----------------------------
# Build the requested JSON shape
# ----------------------------
def build_products_json() -> List[Dict[str, Any]]:
    # Pull Items, Variations, Images
    catalog = list_catalog_all(["ITEM", "ITEM_VARIATION", "IMAGE"], limit=200)
    print("TOTAL objects returned:", len(catalog))
    print(
        "Types sample:",
        sorted({o.get("type") for o in catalog if isinstance(o, dict)})[:20],
    )

    # Map image_id -> url
    image_url_by_id: Dict[str, str] = {}
    for obj in catalog:
        if (
            obj.get("type") == "IMAGE"
            and not obj.get("is_deleted")
            and not obj.get("is_archived")
        ):
            url = (obj.get("image_data") or {}).get("url")
            if url:
                image_url_by_id[obj["id"]] = url

    # Map variation_id -> variation object
    variation_by_id: Dict[str, Dict[str, Any]] = {
        obj["id"]: obj
        for obj in catalog
        if obj.get("type") == "ITEM_VARIATION"
        and not obj.get("is_deleted")
        and not obj.get("is_archived")
    }

    # Filter items: unarchived, not deleted
    items = [
        obj
        for obj in catalog
        if obj.get("type") == "ITEM"
        and not obj.get("is_deleted")
        and not obj.get("is_archived")
    ]

    out: List[Dict[str, Any]] = []

    for item in items:
        item_data = item.get("item_data") or {}
        item_id = item.get("id")
        name = item_data.get("name") or ""
        description = item_data.get("description") or ""

        # Item-level images
        item_image_ids = item_data.get("image_ids") or []
        item_images = [
            image_url_by_id[iid] for iid in item_image_ids if iid in image_url_by_id
        ]

        main_image = item_images[0] if item_images else None
        additional_images = item_images[1:] if len(item_images) > 1 else []

        # Variants: every variation reference listed on the item
        variants_out: List[Dict[str, Any]] = []
        variations = item_data.get("variations") or []

        # Some catalogs also store a single "variations" array of objects with just id.
        for vref in variations:
            variation_id = vref.get("id")
            if not variation_id:
                continue

            vobj = variation_by_id.get(variation_id)
            if not vobj:
                continue

            vdata = vobj.get("item_variation_data") or {}
            vname = vdata.get("name") or "Default"
            price = money_to_float(vdata.get("price_money"))

            # Attempt to find variation-specific image IDs, if present.
            # Many catalogs won't have these; in that case images=[]
            v_image_ids = vdata.get("image_ids") or []
            v_images = [
                image_url_by_id[iid] for iid in v_image_ids if iid in image_url_by_id
            ]

            # Create payment link for this specific variation
            try:
                buy_link = create_payment_link_for_variation(variation_id, name, vname)
            except Exception as e:
                # Keep the variant, but record failure for debugging
                buy_link = None
                print(
                    f"[WARN] Payment link failed for {name} / {vname} ({variation_id}): {e}"
                )

            variants_out.append(
                {
                    "id": variation_id,
                    "name": vname,
                    "images": v_images,
                    "price": price,
                    "buy_link": buy_link,
                }
            )

            time.sleep(SLEEP_BETWEEN_CALLS_SEC)

        # Only include items that actually have variations
        if not variants_out:
            continue

        out.append(
            {
                "id": item_id,
                "image": main_image,
                "varients": variants_out,  # keeping your exact key spelling
                "additional_images": additional_images,
                "video": None,
                "additional_videos": [],
                "name": name,
                "description": description,
                "product-type": "physical",
            }
        )

    return out


def main() -> None:
    products = build_products_json()
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(products, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(products)} products to {OUT_JSON}")


if __name__ == "__main__":
    main()
