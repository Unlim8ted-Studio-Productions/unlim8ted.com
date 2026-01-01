#!/usr/bin/env python3
"""
GUI: Build per-product `variation_types` for products.json (Tkinter)

Goal:
- Load products.json (array of products).
- For each product, infer candidate variation type labels from varients[].optionParts length.
- Let you quickly set `variation_types` like ["Color","Size"] (one per product).
- Writes back into each product object.

Works with your structure:
{
  "id": "...",
  "image": "...",
  "varients": [
    { "variantLabel": "Navy, White", "optionParts": ["Navy","White"], ... }
  ],
  "variation_types": ["Color","Size"]   <-- we add/update this
}

Notes:
- This is NOT auto-perfect. It gives you a fast UI:
  • select product
  • set the number of variation slots based on max optionParts length
  • choose labels per slot from dropdowns (Color/Size/Style/etc) or type custom
  • apply and move on
- No auto-collapsing on apply (we update row in place).
"""

import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Any, Dict, List, Optional


COMMON_TYPES = [
    "Color",
    "Size",
    "Style",
    "Material",
    "Fit",
    "Gender",
    "Length",
    "Width",
    "Capacity",
    "Pack",
    "Scent",
    "Flavor",
    "Finish",
    "Model",
    "Edition",
    "Custom",
]


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def safe_str(x: Any) -> str:
    return "" if x is None else str(x)


def is_products_array(data: Any) -> bool:
    return isinstance(data, list) and all(isinstance(p, dict) for p in data)


def max_option_parts(product: Dict[str, Any]) -> int:
    mx = 0
    vars_ = product.get("varients") or []
    if isinstance(vars_, list):
        for v in vars_:
            if not isinstance(v, dict):
                continue
            parts = v.get("optionParts")
            if isinstance(parts, list):
                mx = max(mx, len(parts))
    return mx


def sample_option_parts(product: Dict[str, Any], limit: int = 6) -> List[List[str]]:
    out = []
    vars_ = product.get("varients") or []
    if isinstance(vars_, list):
        for v in vars_:
            if not isinstance(v, dict):
                continue
            parts = v.get("optionParts")
            if isinstance(parts, list) and parts:
                out.append([safe_str(x).strip() for x in parts if safe_str(x).strip()])
            if len(out) >= limit:
                break
    return out


class VariationTypesGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Variation Types Builder (products.json)")
        self.root.geometry("1200x750")

        self.path: Optional[str] = None
        self.data: List[Dict[str, Any]] = []
        self.filtered_indices: List[int] = []

        self.current_index: Optional[int] = None
        self.type_vars: List[tk.StringVar] = []
        self.custom_entries: List[ttk.Entry] = []

        self._build_ui()

    def _build_ui(self):
        top = ttk.Frame(self.root, padding=10)
        top.pack(fill="x")

        ttk.Button(top, text="Open products.json…", command=self.open_file).pack(
            side="left"
        )
        self.path_var = tk.StringVar(value="(no file loaded)")
        ttk.Entry(top, textvariable=self.path_var, width=95).pack(
            side="left", padx=8, fill="x", expand=True
        )
        ttk.Button(top, text="Save", command=self.save).pack(side="left", padx=4)
        ttk.Button(top, text="Save As…", command=self.save_as).pack(side="left", padx=4)

        mid = ttk.PanedWindow(self.root, orient="horizontal")
        mid.pack(fill="both", expand=True, padx=10, pady=10)

        # LEFT: products list
        left = ttk.Frame(mid, padding=8)
        mid.add(left, weight=1)

        ttk.Label(left, text="Products", font=("Segoe UI", 11, "bold")).pack(anchor="w")

        search_row = ttk.Frame(left)
        search_row.pack(fill="x", pady=(6, 6))
        ttk.Label(search_row, text="Search:").pack(side="left")
        self.search_var = tk.StringVar()
        ent = ttk.Entry(search_row, textvariable=self.search_var)
        ent.pack(side="left", fill="x", expand=True, padx=6)
        ent.bind("<Return>", lambda _e: self.refresh_list())
        ttk.Button(search_row, text="Apply", command=self.refresh_list).pack(
            side="left"
        )

        self.tree = ttk.Treeview(
            left, columns=("id", "mx", "current"), show="headings", selectmode="browse"
        )
        self.tree.heading("id", text="Product ID")
        self.tree.heading("mx", text="Max optionParts")
        self.tree.heading("current", text="variation_types")
        self.tree.column("id", width=420, stretch=True)
        self.tree.column("mx", width=120, stretch=False, anchor="center")
        self.tree.column("current", width=340, stretch=True)
        self.tree.pack(fill="both", expand=True)
        self.tree.bind("<<TreeviewSelect>>", self.on_select)

        # RIGHT: editor
        right = ttk.Frame(mid, padding=8)
        mid.add(right, weight=2)

        ttk.Label(right, text="Editor", font=("Segoe UI", 11, "bold")).pack(anchor="w")

        self.info_var = tk.StringVar(value="Load a file, then select a product.")
        ttk.Label(right, textvariable=self.info_var).pack(anchor="w", pady=(6, 8))

        self.samples_var = tk.StringVar(value="")
        ttk.Label(right, textvariable=self.samples_var, foreground="#666").pack(
            anchor="w", pady=(0, 12)
        )

        self.editor_frame = ttk.Frame(right)
        self.editor_frame.pack(fill="x")

        btn_row = ttk.Frame(right)
        btn_row.pack(fill="x", pady=(14, 0))

        ttk.Button(btn_row, text="Auto-suggest", command=self.auto_suggest).pack(
            side="left"
        )
        ttk.Button(
            btn_row, text="Apply to product", command=self.apply_to_product
        ).pack(side="left", padx=8)
        ttk.Button(
            btn_row, text="Clear variation_types", command=self.clear_types
        ).pack(side="left", padx=8)

        ttk.Separator(right).pack(fill="x", pady=14)

        nav_row = ttk.Frame(right)
        nav_row.pack(fill="x")
        ttk.Button(nav_row, text="◀ Prev", command=self.prev_product).pack(side="left")
        ttk.Button(nav_row, text="Next ▶", command=self.next_product).pack(
            side="left", padx=8
        )

        self.status_var = tk.StringVar(value="")
        ttk.Label(right, textvariable=self.status_var).pack(anchor="w", pady=(10, 0))

    # ---------- file ----------

    def open_file(self):
        path = filedialog.askopenfilename(
            title="Open products.json",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            data = load_json(path)
            if not is_products_array(data):
                raise ValueError("Expected a JSON array of product objects.")
            self.data = data
            self.path = path
            self.path_var.set(path)
            self.refresh_list()
            self.status_var.set(f"Loaded {len(self.data)} products.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load:\n{e}")

    def save(self):
        if not self.path:
            messagebox.showinfo("Save", "No file loaded. Use Save As…")
            return
        try:
            save_json(self.path, self.data)
            self.status_var.set("Saved.")
        except Exception as e:
            messagebox.showerror("Error", f"Save failed:\n{e}")

    def save_as(self):
        if not self.data:
            messagebox.showinfo("Save As", "Load a file first.")
            return
        path = filedialog.asksaveasfilename(
            title="Save products.json as",
            defaultextension=".json",
            filetypes=[("JSON", "*.json")],
        )
        if not path:
            return
        try:
            save_json(path, self.data)
            self.path = path
            self.path_var.set(path)
            self.status_var.set(f"Saved as {path}")
        except Exception as e:
            messagebox.showerror("Error", f"Save As failed:\n{e}")

    # ---------- list ----------

    def refresh_list(self):
        self.tree.delete(*self.tree.get_children())
        self.filtered_indices = []

        q = (self.search_var.get() or "").strip().lower()

        for i, p in enumerate(self.data):
            pid = safe_str(p.get("id")).strip()
            if not pid:
                continue
            mx = max_option_parts(p)
            vt = p.get("variation_types")
            vt_str = ", ".join(vt) if isinstance(vt, list) else ""

            hay = f"{pid} {vt_str}".lower()
            if q and q not in hay:
                continue

            self.filtered_indices.append(i)
            self.tree.insert("", "end", iid=str(i), values=(pid, mx, vt_str))

        self.current_index = None
        self._clear_editor()
        self.info_var.set("Select a product to edit variation_types.")
        self.samples_var.set("")
        self.status_var.set(f"Showing {len(self.filtered_indices)} products.")

    # ---------- selection/editor ----------

    def on_select(self, _evt=None):
        sel = self.tree.selection()
        if not sel:
            return
        idx = int(sel[0])
        self.current_index = idx
        p = self.data[idx]
        pid = safe_str(p.get("id")).strip()

        mx = max_option_parts(p)
        existing = (
            p.get("variation_types")
            if isinstance(p.get("variation_types"), list)
            else []
        )
        existing = [safe_str(x).strip() for x in existing if safe_str(x).strip()]

        self.info_var.set(f"Product: {pid}   |   Max optionParts length: {mx}")
        samples = sample_option_parts(p, limit=6)
        if samples:
            lines = ["Examples of optionParts:"] + [
                f"  - {', '.join(s)}" for s in samples
            ]
            self.samples_var.set("\n".join(lines))
        else:
            self.samples_var.set("No optionParts found on variants for this product.")

        self._build_editor_slots(mx, existing)

    def _clear_editor(self):
        for child in self.editor_frame.winfo_children():
            child.destroy()
        self.type_vars = []
        self.custom_entries = []

    def _build_editor_slots(self, mx: int, existing: List[str]):
        self._clear_editor()

        if mx <= 0:
            ttk.Label(
                self.editor_frame, text="No variations detected (max optionParts = 0)."
            ).pack(anchor="w")
            return

        # Build rows for each slot: dropdown + optional custom
        for slot in range(mx):
            row = ttk.Frame(self.editor_frame)
            row.pack(fill="x", pady=4)

            ttk.Label(row, text=f"Slot {slot+1}:", width=10).pack(side="left")

            var = tk.StringVar(value=existing[slot] if slot < len(existing) else "")
            self.type_vars.append(var)

            cb = ttk.Combobox(row, textvariable=var, values=COMMON_TYPES, width=18)
            cb.pack(side="left", padx=6)

            ttk.Label(row, text="(or type custom):").pack(side="left", padx=(10, 4))

            custom = ttk.Entry(row, width=28)
            custom.pack(side="left", fill="x", expand=True)

            # If existing isn't one of common types, put it in custom too
            ex = existing[slot] if slot < len(existing) else ""
            if ex and ex not in COMMON_TYPES:
                custom.insert(0, ex)
                var.set("Custom")

            self.custom_entries.append(custom)

    def auto_suggest(self):
        """
        Very simple heuristic:
        - If max slots == 1 -> Color
        - If max slots == 2 -> Color, Size
        - If max slots == 3 -> Color, Size, Style
        """
        if self.current_index is None:
            return
        p = self.data[self.current_index]
        mx = max_option_parts(p)
        if mx <= 0:
            return

        # ensure slots exist
        if len(self.type_vars) != mx:
            self._build_editor_slots(mx, [])

        suggestion = []
        if mx == 1:
            suggestion = ["Color"]
        elif mx == 2:
            suggestion = ["Color", "Size"]
        else:
            suggestion = ["Color", "Size", "Style"] + ["Custom"] * max(0, mx - 3)

        for i in range(mx):
            s = suggestion[i] if i < len(suggestion) else "Custom"
            self.type_vars[i].set(s)
            self.custom_entries[i].delete(0, "end")

        self.status_var.set("Auto-suggest applied (you can adjust).")

    def apply_to_product(self):
        if self.current_index is None:
            messagebox.showinfo("Apply", "Select a product first.")
            return

        mx = len(self.type_vars)
        if mx == 0:
            messagebox.showinfo("Apply", "No variation slots to apply.")
            return

        vt: List[str] = []
        for i in range(mx):
            choice = safe_str(self.type_vars[i].get()).strip()
            custom = safe_str(self.custom_entries[i].get()).strip()

            if choice == "Custom":
                val = custom
            else:
                val = (
                    custom if custom else choice
                )  # allow custom override even if dropdown chosen

            val = val.strip()
            if not val:
                continue
            vt.append(val)

        # If user left everything blank, don't write junk
        if not vt:
            if messagebox.askyesno(
                "Apply", "No types selected. Clear variation_types instead?"
            ):
                self.clear_types()
            return

        self.data[self.current_index]["variation_types"] = vt

        # Update row in-place (no collapsing)
        iid = str(self.current_index)
        if self.tree.exists(iid):
            pid, mxv, _ = self.tree.item(iid, "values")
            self.tree.item(iid, values=(pid, mxv, ", ".join(vt)))

        self.status_var.set(f"Applied variation_types: {vt}")

    def clear_types(self):
        if self.current_index is None:
            return
        self.data[self.current_index].pop("variation_types", None)

        iid = str(self.current_index)
        if self.tree.exists(iid):
            pid, mxv, _ = self.tree.item(iid, "values")
            self.tree.item(iid, values=(pid, mxv, ""))

        self.status_var.set("Cleared variation_types for this product.")

    # ---------- navigation ----------

    def _find_filtered_pos(self) -> Optional[int]:
        if self.current_index is None:
            return None
        try:
            return self.filtered_indices.index(self.current_index)
        except ValueError:
            return None

    def prev_product(self):
        if not self.filtered_indices:
            return
        pos = self._find_filtered_pos()
        if pos is None:
            # select first
            self._select_index(self.filtered_indices[0])
            return
        pos2 = max(0, pos - 1)
        self._select_index(self.filtered_indices[pos2])

    def next_product(self):
        if not self.filtered_indices:
            return
        pos = self._find_filtered_pos()
        if pos is None:
            self._select_index(self.filtered_indices[0])
            return
        pos2 = min(len(self.filtered_indices) - 1, pos + 1)
        self._select_index(self.filtered_indices[pos2])

    def _select_index(self, idx: int):
        iid = str(idx)
        if not self.tree.exists(iid):
            return
        self.tree.selection_set(iid)
        self.tree.focus(iid)
        self.tree.see(iid)
        self.on_select(None)


def main():
    root = tk.Tk()
    try:
        style = ttk.Style()
        style.theme_use("clam")
    except Exception:
        pass
    VariationTypesGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
