#!/usr/bin/env python3
"""
Repo HTML refactorer (static-site friendly)

For every *.html file (excluding:
  - anything under /tools/ or /components/
  - error pages like 404.html, 500.html, 502.html, 403.html
  - root index.html (by default)
), this script will:

1) Move:
     cart.html  -> cart/index.html
     about.html -> about/index.html
   (same directory level)

2) Create per-page folders:
     cart/recources/css/
     cart/recources/js/

3) Extract inline <style>...</style> blocks into separate CSS files,
   replacing each <style> with a <link rel="stylesheet" href="./recources/css/..."> placed
   in the same spot.

4) Extract inline <script>...</script> blocks (only those WITHOUT src=, and excluding JSON-LD)
   into separate JS files, replacing each with a <script ... src="./recources/js/..."></script>
   keeping the *same* attributes (type/module/etc) and keeping it in the same spot.

5) Update references ONLY for src-like things and fetch/src usage:
   - HTML attributes: src, srcset, poster, data-src
       "/images/..."   -> "/assets/images/..."
       "/music/..."    -> "/assets/music/..."
       "/podcasts/..." -> "/assets/podcasts/..."
     (does NOT touch href=, meta refresh, redirects, etc.)
   - In extracted JS, updates only conservative patterns like:
       fetch("/images/...") -> fetch("/assets/images/...")
       x.src="/images/..."  -> x.src="/assets/images/..."
       { src: "/images/..." } -> { src: "/assets/images/..." }

USAGE:
  python refactor_pages.py --root . --apply
  python refactor_pages.py --root .            # dry-run
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

# --- Config ---
RES_DIR_NAME = "resources"  # actually correct spelling this time
EXCLUDE_DIRS = {
    "tools",
    "components",
    ".git",
    ".github",
}  # won't process html inside these
EXCLUDE_BASENAMES = {"404.html", "500.html", "502.html", "403.html", "50x.html"}
EXCLUDE_ROOT_INDEX = True  # keep /index.html at root
ASSET_PREFIX_MAP = {
    "/images/": "/assets/images/",
    "/music/": "/assets/music/",
    "/podcasts/": "/assets/podcasts/",
}

# Conservative HTML attributes to rewrite (NOT href)
REWRITE_ATTRS = {"src", "srcset", "poster", "data-src"}

# --- Regex helpers ---
STYLE_BLOCK_RE = re.compile(r"(?is)<style(?P<attrs>[^>]*)>(?P<css>.*?)</style>")

# Matches <script ...>...</script> where there is NO src= in the opening tag.
# We later exclude JSON-LD and non-JS types.
SCRIPT_BLOCK_RE = re.compile(r"(?is)<script(?P<attrs>[^>]*)>(?P<js>.*?)</script>")

ATTR_RE = re.compile(r"""(\s+[\w:-]+)(\s*=\s*(".*?"|'.*?'|[^\s"'>]+))?""", re.S)


# For rewriting specific HTML attrs without parsing full HTML
def rewrite_asset_prefix_in_value(val: str) -> str:
    for k, v in ASSET_PREFIX_MAP.items():
        if val.startswith(k):
            return v + val[len(k) :]
    return val


def rewrite_html_src_like_attrs(html: str) -> str:
    # Rewrite only certain attributes (src, srcset, poster, data-src) and only when value begins with /images/ /music/ /podcasts/
    def repl_attr(m: re.Match) -> str:
        attr = m.group(1)
        eq = m.group(2) or ""
        raw_val = m.group(3) or ""
        if not eq:
            return m.group(0)

        # strip quotes if present
        v = raw_val
        quote = ""
        if (v.startswith('"') and v.endswith('"')) or (
            v.startswith("'") and v.endswith("'")
        ):
            quote = v[0]
            v = v[1:-1]

        if attr.lower() in REWRITE_ATTRS:
            new_v = v
            # srcset can contain multiple URLs; rewrite each token that starts with /images/ etc.
            if attr.lower() == "srcset":
                parts = []
                for part in re.split(r"\s*,\s*", v.strip()):
                    if not part:
                        continue
                    # "url 1x" pattern
                    toks = part.strip().split()
                    if toks:
                        toks[0] = rewrite_asset_prefix_in_value(toks[0])
                    parts.append(" ".join(toks))
                new_v = ", ".join(parts)
            else:
                new_v = rewrite_asset_prefix_in_value(v)

            if new_v != v:
                if quote:
                    return f"{attr}={quote}{new_v}{quote}"
                return f"{attr}={new_v}"
        return m.group(0)

    # Replace attributes in a simple tag-attribute pass (not a full HTML parser, but safe enough for this targeted rewrite)
    # Pattern: attr="value" or attr='value'
    return re.sub(
        r"(?i)\b("
        + "|".join(map(re.escape, REWRITE_ATTRS))
        + r')\s*=\s*(".*?"|\'.*?\')',
        lambda mm: repl_attr(
            re.match(r"(?is)\b(?P<a>[\w:-]+)\s*=\s*(?P<v>.*)", mm.group(0)).groupdict()  # type: ignore
        ),
        html,
    )


def rewrite_html_src_attrs_strict(html: str) -> str:
    # A stricter implementation that only touches the chosen attrs and preserves quotes exactly.
    def attr_sub(match: re.Match) -> str:
        attr = match.group("attr")
        quote = match.group("q")
        val = match.group("val")

        if attr.lower() not in REWRITE_ATTRS:
            return match.group(0)

        # srcset special case
        if attr.lower() == "srcset":
            parts = []
            for part in re.split(r"\s*,\s*", val.strip()):
                if not part:
                    continue
                toks = part.strip().split()
                if toks:
                    toks[0] = rewrite_asset_prefix_in_value(toks[0])
                parts.append(" ".join(toks))
            new_val = ", ".join(parts)
        else:
            new_val = rewrite_asset_prefix_in_value(val)

        if new_val == val:
            return match.group(0)
        return f"{attr}={quote}{new_val}{quote}"

    attr_pat = re.compile(
        r'(?P<attr>(?:src|srcset|poster|data-src))\s*=\s*(?P<q>"|\')(?P<val>.*?)(?P=q)',
        re.IGNORECASE | re.DOTALL,
    )
    return attr_pat.sub(attr_sub, html)


def rewrite_js_fetch_and_src_patterns(js: str) -> str:
    # Only rewrite in:
    #   fetch("..."), fetch('...')
    #   .src="..." / src:"..." / src='...'
    # NOT rewriting href.
    out = js

    def sub_prefix(s: str) -> str:
        for k, v in ASSET_PREFIX_MAP.items():
            s = s.replace(k, v) if False else s  # no-op; keep explicit in regex below
        return s

    # fetch("/images/..")
    for k, v in ASSET_PREFIX_MAP.items():
        out = re.sub(
            rf'(\bfetch\s*\(\s*[\'"])({re.escape(k)})([^\'"]*)',
            lambda m: m.group(1) + v + m.group(3),
            out,
        )

    # .src = "/images/.."
    for k, v in ASSET_PREFIX_MAP.items():
        out = re.sub(
            rf'(\.\s*src\s*=\s*[\'"])({re.escape(k)})([^\'"]*)',
            lambda m: m.group(1) + v + m.group(3),
            out,
        )

    # src: "/images/.."
    for k, v in ASSET_PREFIX_MAP.items():
        out = re.sub(
            rf'(\bsrc\s*:\s*[\'"])({re.escape(k)})([^\'"]*)',
            lambda m: m.group(1) + v + m.group(3),
            out,
        )

    return out


@dataclass
class ExtractedBlock:
    kind: str  # "css" or "js"
    attrs: str
    content: str
    replacement_html: str
    out_rel_path: str  # relative to page folder


def is_excluded(path: Path, root: Path) -> bool:
    rel = path.relative_to(root)
    parts = set(rel.parts)
    if parts & EXCLUDE_DIRS:
        return True
    if path.name in EXCLUDE_BASENAMES:
        return True
    if EXCLUDE_ROOT_INDEX and rel.as_posix() == "index.html":
        return True
    return False


def ensure_dir(p: Path, apply: bool) -> None:
    if apply:
        p.mkdir(parents=True, exist_ok=True)


def safe_write_text(path: Path, text: str, apply: bool) -> None:
    if apply:
        path.write_text(text, encoding="utf-8")


def split_script_attrs(attrs: str) -> Tuple[str, Optional[str]]:
    """
    Returns (attrs_str_preserved, script_type_value_if_present)
    attrs includes leading spaces (from regex).
    """
    # Find type=...
    m = re.search(r'(?is)\btype\s*=\s*("([^"]+)"|\'([^\']+)\'|([^\s>]+))', attrs)
    if not m:
        return attrs, None
    t = m.group(2) or m.group(3) or m.group(4)
    return attrs, (t.strip() if t else None)


def has_src_attr(attrs: str) -> bool:
    return re.search(r"(?is)\bsrc\s*=", attrs) is not None


def extract_blocks(html: str, page_name: str) -> Tuple[str, List[ExtractedBlock]]:
    """
    Extracts inline <style> blocks and eligible inline <script> blocks,
    returning (new_html, extracted_blocks_in_order_appearance).
    """
    extracted: List[ExtractedBlock] = []

    # We’ll extract in a single pass by locating all style/script blocks with their spans,
    # then rewriting from start->end.
    candidates: List[Tuple[int, int, str, re.Match]] = []

    for m in STYLE_BLOCK_RE.finditer(html):
        candidates.append((m.start(), m.end(), "style", m))

    for m in SCRIPT_BLOCK_RE.finditer(html):
        candidates.append((m.start(), m.end(), "script", m))

    if not candidates:
        return html, extracted

    candidates.sort(key=lambda x: x[0])

    out_parts = []
    cursor = 0
    css_i = 0
    js_i = 0

    for start, end, kind, m in candidates:
        out_parts.append(html[cursor:start])

        if kind == "style":
            css = (m.group("css") or "").strip("\n")
            attrs = m.group("attrs") or ""
            css_i += 1
            css_name = f"{page_name}.{css_i:02d}.css"
            rel = f"./{RES_DIR_NAME}/css/{css_name}"

            replacement = f'<link rel="stylesheet" href="{rel}">'
            extracted.append(
                ExtractedBlock(
                    kind="css",
                    attrs=attrs,
                    content=(
                        rewrite_js_fetch_and_src_patterns(css) if False else css
                    ),  # no JS rewrite inside CSS
                    replacement_html=replacement,
                    out_rel_path=f"{RES_DIR_NAME}/css/{css_name}",
                )
            )
            out_parts.append(replacement)

        else:
            attrs = m.group("attrs") or ""
            js = (m.group("js") or "").strip("\n")

            # Skip if script already has src=
            if has_src_attr(attrs):
                out_parts.append(html[start:end])
                cursor = end
                continue

            # Skip JSON-LD or other non-JS types
            attrs_preserved, script_type = split_script_attrs(attrs)
            if script_type and script_type.lower() in {
                "application/ld+json",
                "application/json",
            }:
                out_parts.append(html[start:end])
                cursor = end
                continue

            js_i += 1
            js_name = f"{page_name}.{js_i:02d}.js"
            rel = f"./{RES_DIR_NAME}/js/{js_name}"

            # Keep exact attributes, add src, remove inline content
            # Ensure we don't double-space weirdly
            attrs_clean = attrs_preserved.rstrip()

            replacement = f'<script{attrs_clean} src="{rel}"></script>'
            extracted.append(
                ExtractedBlock(
                    kind="js",
                    attrs=attrs_preserved,
                    content=rewrite_js_fetch_and_src_patterns(js),
                    replacement_html=replacement,
                    out_rel_path=f"{RES_DIR_NAME}/js/{js_name}",
                )
            )
            out_parts.append(replacement)

        cursor = end

    out_parts.append(html[cursor:])
    return "".join(out_parts), extracted


def refactor_file(html_path: Path, root: Path, apply: bool) -> None:
    rel = html_path.relative_to(root)
    parent = html_path.parent
    stem = html_path.stem

    # destination folder is "same directory level / <stem> / index.html"
    dest_dir = parent / stem
    dest_html = dest_dir / "index.html"

    # If it's already an index.html inside a folder named same as its parent, don't rewrap
    # (optional safety)
    if html_path.name == "index.html":
        # leave index.html files alone
        return

    # Read
    original = html_path.read_text(encoding="utf-8", errors="replace")

    # Rewrite src-like HTML attributes (NOT href)
    updated = rewrite_html_src_attrs_strict(original)

    # Extract inline CSS/JS into per-page recources/
    updated, extracted = extract_blocks(updated, page_name=stem)

    # Create folders
    css_dir = dest_dir / RES_DIR_NAME / "css"
    js_dir = dest_dir / RES_DIR_NAME / "js"

    # Plan output
    print(f"\n[PROCESS] {rel.as_posix()}")
    print(f"  -> move to: {dest_html.relative_to(root).as_posix()}")
    if extracted:
        print(f"  -> extracted: {len(extracted)} block(s)")
        for b in extracted:
            print(f"     - {b.kind}: {b.out_rel_path}")

    if not apply:
        return

    ensure_dir(css_dir, apply=True)
    ensure_dir(js_dir, apply=True)

    # Write extracted files
    for b in extracted:
        out_path = dest_dir / b.out_rel_path
        safe_write_text(out_path, b.content.strip() + "\n", apply=True)

    # Write updated HTML into new location
    ensure_dir(dest_dir, apply=True)
    safe_write_text(dest_html, updated.strip() + "\n", apply=True)

    # Remove original file
    html_path.unlink()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="Repo root (published folder)")
    ap.add_argument(
        "--apply",
        action="store_true",
        help="Actually write changes (otherwise dry-run)",
    )
    args = ap.parse_args()

    root = Path(args.root).resolve()
    apply = bool(args.apply)

    html_files: List[Path] = []
    for p in root.rglob("*.html"):
        if not p.is_file():
            continue
        if is_excluded(p, root):
            continue
        # Also exclude anything already inside /tools/ or /components/ by path parts (handled)
        html_files.append(p)

    # Sort: shallow files first (more predictable)
    html_files.sort(key=lambda x: (len(x.relative_to(root).parts), x.as_posix()))

    print(f"Found {len(html_files)} HTML file(s) to refactor.")
    if not apply:
        print("DRY-RUN mode. Add --apply to write changes.")

    for f in html_files:
        # If this file is already in a folder that matches its stem, skip to avoid double wrapping
        # e.g. /cart/index.html shouldn’t become /cart/index/index.html
        rel = f.relative_to(root)
        if f.name == "index.html":
            continue

        # Also skip if target would collide with an existing different page
        dest_dir = f.parent / f.stem
        dest_html = dest_dir / "index.html"
        if dest_html.exists():
            print(
                f"\n[SKIP] {rel.as_posix()} -> target exists: {dest_html.relative_to(root).as_posix()}"
            )
            continue

        refactor_file(f, root, apply)

    print("\nDone.")


if __name__ == "__main__":
    main()
