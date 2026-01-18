import json
from pathlib import Path

import firebase_admin
from firebase_admin import credentials, firestore
import requests


# ---- Config ----
PROJECT_ROOT = Path(__file__).resolve().parent

# Make this a Path (NOT a raw string), so .exists() works
SERVICE_ACCOUNT_PATH = Path(
    r"D:\projects\certs\unlim8ted-db-firebase-adminsdk-nhsr0-3bd37e315e.json"
)

PRODUCTS_URL = "https://unlim8ted.com/tools/data/products.json"

# Adjust this if your script isn't at the website root
LOCAL_PRODUCTS_PATH = PROJECT_ROOT / "data" / "products.json"

USE_REMOTE_URL = False  # False = local file, True = fetch from site


def load_products() -> list[dict]:
    if USE_REMOTE_URL:
        r = requests.get(
            PRODUCTS_URL, headers={"Cache-Control": "no-cache"}, timeout=30
        )
        r.raise_for_status()
        data = r.json()
    else:
        if not LOCAL_PRODUCTS_PATH.exists():
            raise FileNotFoundError(
                f"Local products.json not found at: {LOCAL_PRODUCTS_PATH}"
            )
        data = json.loads(LOCAL_PRODUCTS_PATH.read_text(encoding="utf-8"))

    # Accept either raw array OR { "products": [...] }
    products = data if isinstance(data, list) else data.get("products")
    if not isinstance(products, list):
        raise ValueError(
            "products.json must be an array, or an object with a 'products' array."
        )
    return products


def init_firestore():
    if not SERVICE_ACCOUNT_PATH.exists():
        raise FileNotFoundError(
            f"Service account JSON not found: {SERVICE_ACCOUNT_PATH}"
        )

    cred = credentials.Certificate(str(SERVICE_ACCOUNT_PATH))

    # Avoid "The default Firebase app already exists" on reruns
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred)
    else:
        # Reuse existing default app
        firebase_admin.get_app()

    return firestore.client()


def main():
    db = init_firestore()
    products = load_products()
    products_col = db.collection("products")

    uploaded = 0
    skipped = 0

    for product in products:
        if not isinstance(product, dict):
            print("Skipping non-object product:", product)
            skipped += 1
            continue

        pid = product.get("id")
        if not pid:
            print("Skipping product without 'id':", product)
            skipped += 1
            continue

        # Admin SDK -> bypasses Firestore rules
        products_col.document(str(pid)).set(product)
        print(f"Uploaded product: {product.get('name') or pid}")
        uploaded += 1

    print(
        f"\nDone. Uploaded: {uploaded}, Skipped: {skipped}, Total in file: {len(products)}"
    )


if __name__ == "__main__":
    main()
