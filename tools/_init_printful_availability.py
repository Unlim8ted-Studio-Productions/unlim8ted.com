#!/usr/bin/env python3
"""
Printful ↔ products.json Mapper GUI (Tkinter) — PHYSICAL ONLY + RESET + NO AUTO-COLLAPSE (REAL FIX)

Why it was still closing:
- Any approach that *rebuilds the entire TreeView* (delete + reinsert) will collapse nodes on some Tk builds,
  even if you try to restore "open" state.

This version fixes it properly:
✅ On MATCH, we do NOT rebuild either tree.
✅ We update only the affected row(s) in-place.
✅ If "Hide matched Printful items" is ON, we remove only the matched Printful row(s) in-place.

Still available:
- Fetch ALL Printful products (paging.total)
- Fetch ALL Printful variants (bulk product detail fetch)
- RESET mappings for physical only
- Search + Apply still rebuilds the trees (expected)
- Only displays LOCAL products where "product-type": "physical"

Env:
  set PRINTFUL_ACCESS_TOKEN=YOUR_TOKEN

Run:
  python tools\\printful_mapper_gui.py
"""

import os
import json
import time
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Set
import requests

PRINTFUL_BASE = "https://api.printful.com"


# -----------------------------
# Printful API helpers
# -----------------------------


def pf_headers(token: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "Content-Type": "application/json",
        "User-Agent": "unlim8ted-printful-mapper-gui/6.0",
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


def parse_result_and_paging(
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


def best_pf_name(sp: Dict[str, Any]) -> str:
    for k in ("name", "title", "product_name"):
        v = sp.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def best_pf_variant_name(v: Dict[str, Any]) -> str:
    for k in ("name", "variant_name", "title"):
        s = v.get(k)
        if isinstance(s, str) and s.strip():
            return s.strip()
    color = v.get("color")
    size = v.get("size")
    parts = []
    if isinstance(color, str) and color.strip():
        parts.append(color.strip())
    if isinstance(size, str) and size.strip():
        parts.append(size.strip())
    return " / ".join(parts) if parts else ""


def list_store_products_all(token: str, progress_cb=None) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    offset = 0
    limit = 100
    total = None

    while True:
        url = f"{PRINTFUL_BASE}/store/products"
        r = pf_request(token, "GET", url, params={"offset": offset, "limit": limit})
        j = pf_json(r)
        if r.status_code != 200:
            raise RuntimeError(
                f"GET /store/products failed HTTP {r.status_code}\n"
                f"{json.dumps(j, indent=2) if isinstance(j,(dict,list)) else r.text[:1200]}"
            )

        batch, paging = parse_result_and_paging(j)
        if paging and isinstance(paging.get("total"), int):
            total = paging["total"]

        out.extend(batch)
        if progress_cb:
            progress_cb(len(out), total)

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
        raise RuntimeError(
            f"GET /store/products/{store_product_id} failed HTTP {r.status_code}\n"
            f"{json.dumps(j, indent=2) if isinstance(j,(dict,list)) else r.text[:1200]}"
        )
    if isinstance(j, dict) and isinstance(j.get("result"), dict):
        return j["result"]
    return j if isinstance(j, dict) else {}


# -----------------------------
# Local JSON helpers
# -----------------------------


def load_local_products(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("products.json must be a JSON array")
    return data


def save_local_products(path: str, products: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(products, f, indent=2, ensure_ascii=False)
        f.write("\n")


def best_local_product_name(p: Dict[str, Any]) -> str:
    for k in ("name", "title", "productName"):
        v = p.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return str(p.get("id") or "").strip()


def best_local_variant_name(v: Dict[str, Any]) -> str:
    for k in ("variantLabel", "label", "name", "title", "optionLabel"):
        s = v.get(k)
        if isinstance(s, str) and s.strip():
            return s.strip()
    parts = v.get("optionParts") or v.get("options")
    if isinstance(parts, list) and all(isinstance(x, str) for x in parts):
        parts2 = [x.strip() for x in parts if x.strip()]
        if parts2:
            return ", ".join(parts2)
    return str(v.get("id") or "").strip()


def is_physical_product(p: Dict[str, Any]) -> bool:
    return str(p.get("product-type") or "").strip().lower() == "physical"


# -----------------------------
# Matched sets for filtering
# -----------------------------


def compute_matched_printful_products(local_products: List[Dict[str, Any]]) -> Set[str]:
    s: Set[str] = set()
    for p in local_products:
        if not is_physical_product(p):
            continue
        pfid = str(p.get("printful_id") or "").strip()
        if pfid:
            s.add(pfid)
    return s


def compute_matched_external_ids(local_products: List[Dict[str, Any]]) -> Set[str]:
    s: Set[str] = set()
    for p in local_products:
        if not is_physical_product(p):
            continue
        vars_ = p.get("varients") or []
        if isinstance(vars_, list):
            for v in vars_:
                vid = str(v.get("id") or "").strip()
                if vid:
                    s.add(vid)
    return s


# -----------------------------
# GUI selections
# -----------------------------


@dataclass
class LocalSelection:
    product_index: Optional[int] = None
    variant_index: Optional[int] = None


@dataclass
class PrintfulSelection:
    store_product_id: Optional[str] = None
    variant_index: Optional[int] = None


# -----------------------------
# GUI
# -----------------------------


class MapperGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(
            "Unlim8ted Printful Mapper (Physical Only + Reset + No Collapse)"
        )
        self.root.geometry("1440x860")

        self.token = os.environ.get("PRINTFUL_ACCESS_TOKEN", "").strip()

        self.local_path: Optional[str] = None
        self.local_products: List[Dict[str, Any]] = []

        self.pf_products_list: List[Dict[str, Any]] = []
        self.pf_details_cache: Dict[str, Dict[str, Any]] = {}

        self.local_sel = LocalSelection()
        self.pf_sel = PrintfulSelection()

        self._stop_flag = False

        self._build_ui()

    # ---------- UI ----------

    def _build_ui(self):
        top = ttk.Frame(self.root, padding=10)
        top.pack(fill="x")

        ttk.Label(top, text="Local products.json:").grid(row=0, column=0, sticky="w")
        self.local_path_var = tk.StringVar(value="(not loaded)")
        ttk.Entry(top, textvariable=self.local_path_var, width=95).grid(
            row=0, column=1, padx=8, sticky="we"
        )
        ttk.Button(top, text="Open…", command=self.open_local).grid(
            row=0, column=2, padx=4
        )
        ttk.Button(top, text="Save", command=self.save_local).grid(
            row=0, column=3, padx=4
        )
        ttk.Button(top, text="Save As…", command=self.save_local_as).grid(
            row=0, column=4, padx=4
        )

        ttk.Label(top, text="Printful token:").grid(
            row=1, column=0, sticky="w", pady=(8, 0)
        )
        self.token_var = tk.StringVar(value=self.token)
        ttk.Entry(top, textvariable=self.token_var, width=95, show="•").grid(
            row=1, column=1, padx=8, sticky="we", pady=(8, 0)
        )
        ttk.Button(
            top, text="Fetch ALL Products", command=self.fetch_products_async
        ).grid(row=1, column=2, padx=4, pady=(8, 0))
        ttk.Button(
            top, text="Fetch ALL Variants", command=self.fetch_all_variants_async
        ).grid(row=1, column=3, padx=4, pady=(8, 0))
        ttk.Button(top, text="Stop", command=self.stop_work).grid(
            row=1, column=4, padx=4, pady=(8, 0)
        )

        top.grid_columnconfigure(1, weight=1)

        prog = ttk.Frame(self.root, padding=(10, 0, 10, 10))
        prog.pack(fill="x")

        self.progress = ttk.Progressbar(prog, mode="determinate")
        self.progress.pack(fill="x")

        toggles = ttk.Frame(prog)
        toggles.pack(fill="x", pady=(6, 0))

        # Default OFF to avoid confusion
        self.hide_matched_printful_var = tk.BooleanVar(value=False)
        self.hide_matched_local_var = tk.BooleanVar(value=False)

        ttk.Checkbutton(
            toggles,
            text="Hide matched Printful items",
            variable=self.hide_matched_printful_var,
            command=self.refresh_pf_tree,
        ).pack(side="left")
        ttk.Checkbutton(
            toggles,
            text="Hide matched Local variants",
            variable=self.hide_matched_local_var,
            command=self.refresh_local_tree,
        ).pack(side="left", padx=14)

        ttk.Button(
            toggles, text="RESET MAPPINGS (physical only)", command=self.reset_mappings
        ).pack(side="right")

        self.progress_label_var = tk.StringVar(
            value="Load products.json, then Fetch Printful."
        )
        ttk.Label(prog, textvariable=self.progress_label_var).pack(anchor="w")

        mid = ttk.PanedWindow(self.root, orient="horizontal")
        mid.pack(fill="both", expand=True, padx=10, pady=10)

        # LEFT
        left = ttk.Frame(mid, padding=8)
        mid.add(left, weight=1)

        ttk.Label(
            left,
            text='LOCAL (only "product-type": "physical")',
            font=("Segoe UI", 11, "bold"),
        ).pack(anchor="w")

        ls = ttk.Frame(left)
        ls.pack(fill="x", pady=(6, 6))
        self.local_search_var = tk.StringVar()
        ttk.Label(ls, text="Search:").pack(side="left")
        ttk.Entry(ls, textvariable=self.local_search_var).pack(
            side="left", fill="x", expand=True, padx=6
        )
        ttk.Button(ls, text="Apply", command=self.refresh_local_tree).pack(side="left")

        self.local_tree = ttk.Treeview(
            left,
            columns=("type", "pid", "name", "printful_id", "vid"),
            show="tree headings",
        )
        for col, title, w in [
            ("type", "Type", 70),
            ("pid", "Product ID", 240),
            ("name", "Name / Variant", 370),
            ("printful_id", "printful_id", 260),
            ("vid", "Variant ID", 260),
        ]:
            self.local_tree.heading(col, text=title)
            self.local_tree.column(col, width=w, stretch=True)
        self.local_tree.column("#0", width=26, stretch=False)
        self.local_tree.pack(fill="both", expand=True)
        self.local_tree.bind("<<TreeviewSelect>>", self.on_local_select)

        # RIGHT
        right = ttk.Frame(mid, padding=8)
        mid.add(right, weight=1)

        ttk.Label(
            right, text="PRINTFUL (products + variants)", font=("Segoe UI", 11, "bold")
        ).pack(anchor="w")

        ps = ttk.Frame(right)
        ps.pack(fill="x", pady=(6, 6))
        self.pf_search_var = tk.StringVar()
        ttk.Label(ps, text="Search:").pack(side="left")
        ttk.Entry(ps, textvariable=self.pf_search_var).pack(
            side="left", fill="x", expand=True, padx=6
        )
        ttk.Button(ps, text="Apply", command=self.refresh_pf_tree).pack(side="left")

        self.pf_tree = ttk.Treeview(
            right, columns=("type", "spid", "name", "external_id"), show="tree headings"
        )
        for col, title, w in [
            ("type", "Type", 70),
            ("spid", "Store Product ID", 230),
            ("name", "Name / Variant", 440),
            ("external_id", "external_id (Square Variation ID)", 320),
        ]:
            self.pf_tree.heading(col, text=title)
            self.pf_tree.column(col, width=w, stretch=True)
        self.pf_tree.column("#0", width=26, stretch=False)
        self.pf_tree.pack(fill="both", expand=True)
        self.pf_tree.bind("<<TreeviewSelect>>", self.on_pf_select)

        # Bottom actions
        bottom = ttk.Frame(self.root, padding=10)
        bottom.pack(fill="x")

        ttk.Button(
            bottom,
            text="Match PRODUCT  ➜  set local.printful_id",
            command=self.match_product,
        ).pack(side="left")
        ttk.Button(
            bottom,
            text="Match VARIANT  ➜  set local variant.id = external_id",
            command=self.match_variant,
        ).pack(side="left", padx=8)
        ttk.Button(bottom, text="Save", command=self.save_local).pack(
            side="left", padx=8
        )

        self.status_var = tk.StringVar(value="")
        ttk.Label(bottom, textvariable=self.status_var).pack(side="right")

    # ---------- status/progress ----------

    def set_progress(self, cur: int, total: Optional[int], label: str):
        if total and total > 0:
            self.progress.configure(maximum=total, value=min(cur, total))
            self.progress_label_var.set(f"{label} {cur}/{total}")
        else:
            self.progress.configure(maximum=100, value=0)
            self.progress_label_var.set(f"{label} {cur}")

    def status(self, msg: str):
        self.status_var.set(msg)

    def stop_work(self):
        self._stop_flag = True
        self.status("Stopping…")

    # ---------- reset mappings ----------

    def reset_mappings(self):
        if not self.local_products:
            messagebox.showinfo("Reset", "Load products.json first.")
            return

        if not messagebox.askyesno(
            "RESET MAPPINGS",
            "This will CLEAR mappings for PHYSICAL products only:\n\n"
            "• remove product.printful_id\n"
            "• remove variant.id\n"
            "• remove variant.available\n\n"
            "Continue?",
        ):
            return

        cleared_products = 0
        cleared_variants = 0

        for p in self.local_products:
            if not is_physical_product(p):
                continue

            if "printful_id" in p and str(p.get("printful_id") or "").strip():
                p.pop("printful_id", None)
                cleared_products += 1

            vars_ = p.get("varients") or []
            if isinstance(vars_, list):
                for v in vars_:
                    if isinstance(v, dict):
                        if "id" in v and str(v.get("id") or "").strip():
                            v.pop("id", None)
                            cleared_variants += 1
                        if "available" in v:
                            v.pop("available", None)

        # After reset, rebuild views (fine to collapse here — user asked reset)
        self.hide_matched_printful_var.set(False)
        self.hide_matched_local_var.set(False)
        self.refresh_local_tree()
        self.refresh_pf_tree()
        self.status(
            f"Reset complete. Cleared product mappings: {cleared_products}, variant ids: {cleared_variants}"
        )

    # ---------- local file ----------

    def open_local(self):
        path = filedialog.askopenfilename(
            title="Open products.json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            self.local_products = load_local_products(path)
            self.local_path = path
            self.local_path_var.set(path)
            self.refresh_local_tree()
            self.refresh_pf_tree()
            self.status(f"Loaded local products: {len(self.local_products)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load JSON:\n{e}")

    def save_local(self):
        if not self.local_path:
            messagebox.showinfo("Save", "No file loaded. Use Save As…")
            return
        try:
            save_local_products(self.local_path, self.local_products)
            self.status("Saved.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save:\n{e}")

    def save_local_as(self):
        if not self.local_products:
            messagebox.showinfo("Save As", "Load a products.json first.")
            return
        path = filedialog.asksaveasfilename(
            title="Save products.json as",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")],
        )
        if not path:
            return
        try:
            save_local_products(path, self.local_products)
            self.local_path = path
            self.local_path_var.set(path)
            self.status(f"Saved as {path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save:\n{e}")

    # ---------- tree rebuild (used for Apply/search, fetches, reset) ----------

    def refresh_local_tree(self):
        self.local_tree.delete(*self.local_tree.get_children())
        q = (self.local_search_var.get() or "").strip().lower()
        hide_matched = bool(self.hide_matched_local_var.get())

        for pi, p in enumerate(self.local_products):
            if not is_physical_product(p):
                continue

            pid = str(p.get("id") or "").strip()
            pname = best_local_product_name(p)
            pfid = str(p.get("printful_id") or "").strip()
            prod_hay = f"{pid} {pname} {pfid}".lower()

            prod_iid = self.local_tree.insert(
                "", "end", iid=f"lp:{pi}", values=("PRODUCT", pid, pname, pfid, "")
            )

            vars_ = p.get("varients") or []
            any_variant_shown = False
            if isinstance(vars_, list):
                for vi, v in enumerate(vars_):
                    if not isinstance(v, dict):
                        continue
                    vid = str(v.get("id") or "").strip()
                    vname = best_local_variant_name(v)
                    vhay = f"{pid} {pname} {vid} {vname}".lower()

                    if hide_matched and vid:
                        continue
                    if q and (q not in prod_hay and q not in vhay):
                        continue

                    self.local_tree.insert(
                        prod_iid,
                        "end",
                        iid=f"lv:{pi}:{vi}",
                        values=("VARIANT", pid, vname, "", vid),
                    )
                    any_variant_shown = True

            if q and (q not in prod_hay) and (not any_variant_shown):
                self.local_tree.delete(prod_iid)

    def refresh_pf_tree(self):
        self.pf_tree.delete(*self.pf_tree.get_children())
        q = (self.pf_search_var.get() or "").strip().lower()

        hide_matched = bool(self.hide_matched_printful_var.get())
        matched_products = (
            compute_matched_printful_products(self.local_products)
            if hide_matched
            else set()
        )
        matched_external_ids = (
            compute_matched_external_ids(self.local_products) if hide_matched else set()
        )

        for sp in self.pf_products_list:
            spid = str(sp.get("id") or "").strip()
            name = best_pf_name(sp)
            hay = f"{spid} {name}".lower()

            if hide_matched and spid in matched_products:
                continue

            prod_iid = self.pf_tree.insert(
                "", "end", iid=f"pp:{spid}", values=("PRODUCT", spid, name, "")
            )

            detail = self.pf_details_cache.get(spid)
            if detail:
                vars_ = (
                    detail.get("sync_variants")
                    or detail.get("variants")
                    or detail.get("items")
                    or []
                )
                any_variant_shown = False
                if isinstance(vars_, list):
                    for vi, v in enumerate(vars_):
                        if not isinstance(v, dict):
                            continue
                        vname = best_pf_variant_name(v)
                        ext = str(v.get("external_id") or "").strip()
                        vhay = f"{spid} {name} {vname} {ext}".lower()

                        if hide_matched and ext and ext in matched_external_ids:
                            continue
                        if q and q not in hay and q not in vhay:
                            continue

                        self.pf_tree.insert(
                            prod_iid,
                            "end",
                            iid=f"pv:{spid}:{vi}",
                            values=("VARIANT", spid, vname, ext),
                        )
                        any_variant_shown = True

                if q and (q not in hay) and (not any_variant_shown):
                    self.pf_tree.delete(prod_iid)
            else:
                if q and q not in hay:
                    self.pf_tree.delete(prod_iid)

    # ---------- selection ----------

    def on_local_select(self, _evt=None):
        sel = self.local_tree.selection()
        if not sel:
            return
        iid = sel[0]
        if iid.startswith("lp:"):
            pi = int(iid.split(":")[1])
            self.local_sel = LocalSelection(product_index=pi, variant_index=None)
            p = self.local_products[pi]
            self.status(
                f"Local PRODUCT: {p.get('id')} (printful_id={p.get('printful_id')})"
            )
        elif iid.startswith("lv:"):
            _, pi, vi = iid.split(":")
            pi = int(pi)
            vi = int(vi)
            self.local_sel = LocalSelection(product_index=pi, variant_index=vi)
            p = self.local_products[pi]
            v = (p.get("varients") or [])[vi]
            self.status(
                f"Local VARIANT: {best_local_variant_name(v)} (id={v.get('id')})"
            )

    def on_pf_select(self, _evt=None):
        sel = self.pf_tree.selection()
        if not sel:
            return
        iid = sel[0]
        if iid.startswith("pp:"):
            spid = iid.split(":", 1)[1]
            self.pf_sel = PrintfulSelection(store_product_id=spid, variant_index=None)
            self.status(f"Printful PRODUCT: {spid}")
        elif iid.startswith("pv:"):
            _, spid, vi = iid.split(":")
            self.pf_sel = PrintfulSelection(
                store_product_id=spid, variant_index=int(vi)
            )
            detail = self.pf_details_cache.get(spid)
            ext = ""
            name = ""
            if detail:
                vars_ = (
                    detail.get("sync_variants")
                    or detail.get("variants")
                    or detail.get("items")
                    or []
                )
                if isinstance(vars_, list) and 0 <= int(vi) < len(vars_):
                    pv = vars_[int(vi)]
                    ext = str(pv.get("external_id") or "").strip()
                    name = best_pf_variant_name(pv)
            self.status(f"Printful VARIANT: {name} (external_id={ext})")

    # ---------- fetching ----------

    def fetch_products_async(self):
        token = (self.token_var.get() or "").strip()
        if not token:
            messagebox.showerror(
                "Token missing",
                "Set PRINTFUL_ACCESS_TOKEN or paste it into the token field.",
            )
            return
        self.token = token
        self._stop_flag = False

        def progress_cb(count, total):
            self.root.after(
                0, lambda: self.set_progress(count, total, "Fetched products:")
            )

        def worker():
            try:
                self.root.after(
                    0, lambda: self.status("Fetching all Printful products…")
                )
                products = list_store_products_all(self.token, progress_cb=progress_cb)
                if self._stop_flag:
                    return
                self.pf_products_list = products
                self.pf_details_cache = {}
                self.root.after(0, self.refresh_pf_tree)
                self.root.after(
                    0,
                    lambda: self.status(f"Fetched Printful products: {len(products)}"),
                )
            except Exception as e:
                self.root.after(
                    0, lambda: messagebox.showerror("Printful error", str(e))
                )
                self.root.after(0, lambda: self.status("Fetch products failed."))

        threading.Thread(target=worker, daemon=True).start()

    def fetch_all_variants_async(self):
        if not self.pf_products_list:
            messagebox.showinfo("Fetch Variants", "Fetch ALL Products first.")
            return
        token = (self.token_var.get() or "").strip()
        if not token:
            messagebox.showerror(
                "Token missing",
                "Set PRINTFUL_ACCESS_TOKEN or paste it into the token field.",
            )
            return
        self.token = token
        self._stop_flag = False

        ids = [
            str(sp.get("id") or "").strip()
            for sp in self.pf_products_list
            if str(sp.get("id") or "").strip()
        ]
        total = len(ids)

        def worker():
            try:
                self.root.after(
                    0, lambda: self.status("Fetching ALL variants for ALL products…")
                )
                self.root.after(
                    0, lambda: self.set_progress(0, total, "Fetched variant sets:")
                )
                for i, spid in enumerate(ids, start=1):
                    if self._stop_flag:
                        self.root.after(0, lambda: self.status("Stopped."))
                        return
                    if spid in self.pf_details_cache:
                        self.root.after(
                            0,
                            lambda i=i: self.set_progress(
                                i, total, "Fetched variant sets:"
                            ),
                        )
                        continue
                    detail = get_store_product_detail(self.token, spid)
                    self.pf_details_cache[spid] = detail
                    self.root.after(
                        0,
                        lambda i=i: self.set_progress(
                            i, total, "Fetched variant sets:"
                        ),
                    )
                    time.sleep(0.10)

                if self._stop_flag:
                    return
                self.root.after(0, self.refresh_pf_tree)
                self.root.after(0, lambda: self.status("Fetched ALL variants."))
            except Exception as e:
                self.root.after(
                    0, lambda: messagebox.showerror("Printful error", str(e))
                )
                self.root.after(0, lambda: self.status("Fetch variants failed."))

        threading.Thread(target=worker, daemon=True).start()

    # ---------- in-place updates (NO COLLAPSE ON MATCH) ----------

    def _update_local_product_row_inplace(self, pi: int):
        iid = f"lp:{pi}"
        if not self.local_tree.exists(iid):
            return
        p = self.local_products[pi]
        pid = str(p.get("id") or "").strip()
        pname = best_local_product_name(p)
        pfid = str(p.get("printful_id") or "").strip()
        self.local_tree.item(iid, values=("PRODUCT", pid, pname, pfid, ""))

    def _update_local_variant_row_inplace(self, pi: int, vi: int):
        iid = f"lv:{pi}:{vi}"
        if not self.local_tree.exists(iid):
            return
        p = self.local_products[pi]
        pid = str(p.get("id") or "").strip()
        vars_ = p.get("varients") or []
        if not isinstance(vars_, list) or not (0 <= vi < len(vars_)):
            return
        v = vars_[vi]
        vname = best_local_variant_name(v)
        vid = str(v.get("id") or "").strip()
        self.local_tree.item(iid, values=("VARIANT", pid, vname, "", vid))

    def _remove_printful_product_inplace_if_hidden(self, spid: str):
        if not self.hide_matched_printful_var.get():
            return
        pid = f"pp:{spid}"
        if self.pf_tree.exists(pid):
            self.pf_tree.delete(pid)

    def _remove_printful_variant_inplace_if_hidden(self, spid: str, pvi: int):
        if not self.hide_matched_printful_var.get():
            return
        vid = f"pv:{spid}:{pvi}"
        if self.pf_tree.exists(vid):
            self.pf_tree.delete(vid)
        # if parent has no children left, remove parent too
        pid = f"pp:{spid}"
        if self.pf_tree.exists(pid) and len(self.pf_tree.get_children(pid)) == 0:
            self.pf_tree.delete(pid)

    # ---------- matching ----------

    def match_product(self):
        if self.local_sel.product_index is None:
            messagebox.showinfo("Match Product", "Select a LOCAL product (left).")
            return
        if not self.pf_sel.store_product_id or self.pf_sel.variant_index is not None:
            messagebox.showinfo(
                "Match Product", "Select a PRINTFUL PRODUCT (right), not a variant."
            )
            return

        pi = self.local_sel.product_index
        if not is_physical_product(self.local_products[pi]):
            messagebox.showinfo(
                "Match Product", "Selected local product is not physical."
            )
            return

        spid = self.pf_sel.store_product_id
        self.local_products[pi]["printful_id"] = spid

        # ✅ In-place update (no collapse)
        self._update_local_product_row_inplace(pi)
        self._remove_printful_product_inplace_if_hidden(spid)

        self.status(f"Matched product: local.printful_id={spid}")

    def match_variant(self):
        if self.local_sel.product_index is None or self.local_sel.variant_index is None:
            messagebox.showinfo("Match Variant", "Select a LOCAL variant (left).")
            return
        if not self.pf_sel.store_product_id or self.pf_sel.variant_index is None:
            messagebox.showinfo("Match Variant", "Select a PRINTFUL variant (right).")
            return

        pi = self.local_sel.product_index
        if not is_physical_product(self.local_products[pi]):
            messagebox.showinfo(
                "Match Variant", "Selected local product is not physical."
            )
            return

        spid = self.pf_sel.store_product_id
        if spid not in self.pf_details_cache:
            messagebox.showinfo(
                "Match Variant", "Fetch variants first (Fetch ALL Variants)."
            )
            return

        detail = self.pf_details_cache[spid]
        pf_vars = (
            detail.get("sync_variants")
            or detail.get("variants")
            or detail.get("items")
            or []
        )
        if not isinstance(pf_vars, list):
            messagebox.showerror(
                "Match Variant", "Printful product detail has no variants array."
            )
            return

        pvi = self.pf_sel.variant_index
        if pvi is None or not (0 <= pvi < len(pf_vars)):
            messagebox.showerror("Match Variant", "Invalid Printful variant selection.")
            return

        pfv = pf_vars[pvi]
        ext = str(pfv.get("external_id") or "").strip()
        if not ext:
            messagebox.showerror(
                "Match Variant",
                "This Printful variant has NO external_id.\n\n"
                "Set external_id in Printful to the Square variation ID, then retry.",
            )
            return

        lvi = self.local_sel.variant_index
        local_vars = self.local_products[pi].get("varients") or []
        if not isinstance(local_vars, list) or not (0 <= lvi < len(local_vars)):
            messagebox.showerror("Match Variant", "Invalid local variant selection.")
            return

        local_vars[lvi]["id"] = ext

        # ✅ In-place update (no collapse)
        self._update_local_variant_row_inplace(pi, lvi)
        self._remove_printful_variant_inplace_if_hidden(spid, pvi)

        self.status(f"Matched variant: local.variant.id={ext}")


def main():
    root = tk.Tk()
    try:
        style = ttk.Style()
        style.theme_use("clam")
    except Exception:
        pass
    MapperGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
