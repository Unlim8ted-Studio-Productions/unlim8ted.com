import os
import json
import time
import argparse
from typing import Any, Dict, List, Optional, Set

import requests

PRINTFUL_API = "https://api.printful.com"


def pf_headers(token: str, store_id: Optional[str] = None) -> Dict[str, str]:
    h = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "unlim8ted-printful-webhook-setup/1.0",
    }
    # Some Printful v2 calls require store header (depends on token type)
    if store_id:
        h["X-PF-Store-Id"] = str(store_id)
    return h


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_printful_product_ids(products: Any) -> List[int]:
    """
    From your sample:
      product.printful_id is a string like "411547155"
    """
    if not isinstance(products, list):
        raise ValueError("products.json must be a JSON array (list)")

    ids: Set[int] = set()
    for p in products:
        if not isinstance(p, dict):
            continue
        v = p.get("printful_id")
        if isinstance(v, int):
            ids.add(v)
        elif isinstance(v, str) and v.strip().isdigit():
            ids.add(int(v.strip()))
    return sorted(ids)


def list_webhooks(token: str, store_id: Optional[str]) -> Dict[str, Any]:
    url = f"{PRINTFUL_API}/v2/webhooks"
    r = requests.get(url, headers=pf_headers(token, store_id), timeout=30)
    r.raise_for_status()
    return r.json()


def delete_webhook(token: str, store_id: Optional[str], webhook_id: int) -> None:
    url = f"{PRINTFUL_API}/v2/webhooks/{webhook_id}"
    r = requests.delete(url, headers=pf_headers(token, store_id), timeout=30)
    r.raise_for_status()


def create_webhook(
    token: str, store_id: Optional[str], webhook_url: str, product_ids: List[int]
) -> Dict[str, Any]:
    """
    Printful v2 webhook config subscribing to catalog_stock_updated filtered by products list.
    """
    url = f"{PRINTFUL_API}/v2/webhooks"
    payload = {
        "default_url": webhook_url,
        "expires_at": None,
        "events": [
            {
                "type": "catalog_stock_updated",
                "params": [
                    {
                        "name": "products",
                        "value": [{"id": int(pid)} for pid in product_ids],
                    }
                ],
            }
        ],
    }

    r = requests.post(
        url, headers=pf_headers(token, store_id), data=json.dumps(payload), timeout=30
    )
    r.raise_for_status()
    return r.json()


def main():
    ap = argparse.ArgumentParser(
        description="Create/replace Printful v2 webhook (catalog_stock_updated)"
    )
    ap.add_argument(
        "--products-json",
        default=os.getenv("PRODUCTS_JSON_PATH", "tools/data/products.json"),
    )
    ap.add_argument(
        "--webhook-url",
        default=os.getenv(
            "WEBHOOK_URL", "https://api.unlim8ted.com/printful-stock-update"
        ),
    )
    ap.add_argument("--store-id", default=os.getenv("PRINTFUL_STORE_ID"))
    ap.add_argument(
        "--replace-existing",
        action="store_true",
        help="Delete existing v2 webhooks then create a fresh one",
    )
    args = ap.parse_args()

    token = os.getenv("PRINTFUL_ACCESS_TOKEN")
    if not token:
        raise SystemExit("Missing env var PRINTFUL_ACCESS_TOKEN")

    products = load_json(args.products_json)
    product_ids = extract_printful_product_ids(products)

    if not product_ids:
        raise SystemExit(
            f"No printful_id values found in {args.products_json}.\n"
            "Each physical product should have a numeric string/int 'printful_id'."
        )

    print(f"Found {len(product_ids)} Printful product IDs in {args.products_json}")
    print(f"Webhook URL: {args.webhook_url}")
    if args.store_id:
        print(f"Using X-PF-Store-Id: {args.store_id}")

    if args.replace_existing:
        existing = list_webhooks(token, args.store_id)
        data = existing.get("data") or []
        if isinstance(data, list) and data:
            print(f"Deleting {len(data)} existing v2 webhook(s)...")
            for wh in data:
                wid = wh.get("id")
                if isinstance(wid, int):
                    delete_webhook(token, args.store_id, wid)
                    time.sleep(0.2)

    created = create_webhook(token, args.store_id, args.webhook_url, product_ids)

    print("\n=== Printful response ===")
    print(json.dumps(created, indent=2))

    # Printful returns secret_key (hex) on create; save into Worker secret.
    # Depending on response wrapper, it may be at created["data"]["secret_key"].
    secret_hex = None
    if isinstance(created, dict):
        d = created.get("data")
        if isinstance(d, dict):
            secret_hex = d.get("secret_key")
        if not secret_hex:
            secret_hex = created.get("secret_key")

    if secret_hex:
        print("\n=== SAVE THIS IN CLOUDFLARE WORKER SECRETS ===")
        print(f"PF_WEBHOOK_SECRET_HEX={secret_hex}")
        print(
            "Note: secret_key is hex; Worker must decode hex -> bytes for HMAC verification."
        )
    else:
        print(
            "\n(Heads up) secret_key not found in response. If you already created one earlier,"
        )
        print(
            "use the secret_key from that create response or re-create with --replace-existing."
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
