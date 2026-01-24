from pathlib import Path
import json
import sys
from typing import Any, Dict, List, Optional

import requests
import firebase_admin
from firebase_admin import credentials, firestore

SERVICE_ACCOUNT_PATH = Path(
    r"D:\projects\certs\unlim8ted-db-firebase-adminsdk-nhsr0-3bd37e315e.json"
)

PRODUCTS_URL = "https://assets.unlim8ted.com/data/products.json"
COLLECTION_NAME = "products"


def fetch_products(url: str) -> List[Dict[str, Any]]:
    r = requests.get(url, headers={"Cache-Control": "no-store"}, timeout=30)
    if not r.ok:
        raise RuntimeError(
            "Failed to fetch products.json (HTTP "
            + str(r.status_code)
            + " "
            + r.reason
            + ")"
        )
    data = r.json()
    products = (
        data
        if isinstance(data, list)
        else data.get("products") if isinstance(data, dict) else None
    )
    if not isinstance(products, list):
        raise RuntimeError(
            "products.json must be a list or an object with a 'products' list"
        )
    # Ensure items are dicts
    out: List[Dict[str, Any]] = []
    for p in products:
        if isinstance(p, dict):
            out.append(p)
    return out


def init_firestore(service_account_path: Path):
    if not service_account_path.exists():
        raise FileNotFoundError(
            "Service account file not found: " + str(service_account_path)
        )

    # Prevent double-init if you run in an interactive session
    if not firebase_admin._apps:
        cred = credentials.Certificate(str(service_account_path))
        firebase_admin.initialize_app(cred)

    return firestore.client()


def upload_products(
    db, products: List[Dict[str, Any]], collection_name: str
) -> Dict[str, int]:
    col_ref = db.collection(collection_name)
    uploaded = 0
    skipped = 0

    for product in products:
        pid = product.get("id")
        if not pid:
            skipped += 1
            continue

        doc_id = str(pid)
        col_ref.document(doc_id).set(product)
        uploaded += 1

        name = product.get("name") or doc_id
        print("Uploaded product: " + str(name))

    return {"uploaded": uploaded, "skipped": skipped, "total": len(products)}


def main() -> int:
    try:
        products = fetch_products(PRODUCTS_URL)
        db = init_firestore(SERVICE_ACCOUNT_PATH)
        stats = upload_products(db, products, COLLECTION_NAME)
        print(
            "Done. Uploaded: "
            + str(stats["uploaded"])
            + " | Skipped: "
            + str(stats["skipped"])
            + " | Total fetched: "
            + str(stats["total"])
        )
        return 0
    except Exception as e:
        print("ERROR: " + str(e), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
