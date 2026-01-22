#!/usr/bin/env python3
"""
tools/update_asset_paths.py

Rewrites references in text/code files:

A) products.json / spinner.json / images.json -> assets.unlim8ted.com/data/{file}
B) /images /music /podcast or \\images \\music \\podcast
   -> assets.unlim8ted.com/{folder}/{rest}

Stability improvements:
- Prints progress: shows the file BEFORE processing (so you see what it gets "stuck" on)
- Skips common heavy dirs (node_modules, dist, build, etc.)
- Skips big files via --max-mb (default 2MB)
- Skips minified/bundle artifacts by default (override with --no-skip-minified)
- Processes line-by-line (does not load entire file into memory)
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Set, Tuple

SEP = r"[\\/]"  # / or \

JSON_REF_RE = re.compile(
    rf"""(?P<full>(?:(?:https?://[^\s"'()<>\]]+?{SEP})|(?:[^\s"'()<>\]]+?{SEP}))?(?P<file>products\.json|spinner\.json|images\.json))""",
    re.IGNORECASE,
)


SEP = r"[\\/]"  # / or \

# Matches:
#   https://unlim8ted.com/assets/images/...
#   https://anything.com/images/...
#   /assets/images/...
#   /images/...
#   \images\...
# and captures folder + rest.
FOLDER_REF_RE = re.compile(
    rf"""
    (?P<full>
        (?:(?P<scheme>https?://)[^\s"'()<>\]]+?)?   # optional scheme+domain
        (?P<prefix>{SEP}assets)?                   # optional /assets
        {SEP}
        (?P<folder>images|music|podcast)
        (?P<rest>(?:{SEP}+[^\s"'()<>\]]*)?)        # rest of path, allowing multiple slashes
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)


DEFAULT_SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    "node_modules",
    "bower_components",
    "dist",
    "build",
    "out",
    ".next",
    ".nuxt",
    ".cache",
    "coverage",
    "venv",
    ".venv",
    "__pycache__",
    "target",
}

DEFAULT_SKIP_NAME_PATTERNS = (
    r"\.min\.",  # foo.min.js, foo.min.css
    r"\.bundle\.",  # foo.bundle.js
    r"\.chunk\.",  # chunk files
    r"-bundle\.",  # foo-bundle.js
    r"\.map$",  # sourcemaps
)

DEFAULT_EXTS = {
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".html",
    ".htm",
    ".css",
    ".scss",
    ".sass",
    ".less",
    ".md",
    ".txt",
}


@dataclass
class Stats:
    scanned: int = 0
    changed_files: int = 0
    replacements: int = 0
    skipped_large: int = 0
    skipped_binary: int = 0
    skipped_name: int = 0


def looks_binary(sample: bytes) -> bool:
    return b"\x00" in sample


def rewrite_line(line: str) -> Tuple[str, int]:
    reps = 0

    def json_sub(m: re.Match) -> str:
        nonlocal reps
        reps += 1
        return f"assets.unlim8ted.com/data/{m.group('file').lower()}"

    def folder_sub(m: re.Match) -> str:
        full = m.group("full")

        # If it's already rewritten, leave it alone (prevents double-prefixing)
        if "assets.unlim8ted.com" in full.lower():
            return full

        folder = m.group("folder").lower()
        rest = m.group("rest") or ""

        # Normalize rest to use forward slashes and collapse multiple slashes
        rest = rest.replace("\\", "/")
        rest = re.sub(r"/{2,}", "/", rest)

        # Ensure rest begins with exactly one slash if non-empty
        if rest and not rest.startswith("/"):
            rest = "/" + rest

        return f"assets.unlim8ted.com/{folder}{rest}"

    line2 = JSON_REF_RE.sub(json_sub, line)
    line3 = FOLDER_REF_RE.sub(folder_sub, line2)
    return line3, reps


def iter_files(
    root: Path, skip_dirs: Set[str], allowed_exts: Set[str] | None, include_hidden: bool
) -> Iterable[Path]:
    for dirpath, dirnames, filenames in os.walk(root):
        # prune dirs for speed
        pruned = []
        for d in dirnames:
            if d in skip_dirs:
                continue
            if not include_hidden and d.startswith("."):
                continue
            pruned.append(d)
        dirnames[:] = pruned

        for name in filenames:
            if not include_hidden and name.startswith("."):
                continue
            p = Path(dirpath) / name
            if allowed_exts is not None and p.suffix.lower() not in allowed_exts:
                continue
            yield p


def should_skip_name(path: Path, skip_name_res: list[re.Pattern]) -> bool:
    s = path.name.lower()
    return any(r.search(s) for r in skip_name_res)


def process_file(
    path: Path,
    dry_run: bool,
    max_bytes: int,
    skip_name_res: list[re.Pattern],
    stats: Stats,
    verbose: bool,
) -> None:
    stats.scanned += 1

    if verbose:
        print(f"-> {path}", flush=True)

    try:
        st = path.stat()
    except OSError:
        return

    if st.st_size > max_bytes:
        stats.skipped_large += 1
        return

    if should_skip_name(path, skip_name_res):
        stats.skipped_name += 1
        return

    # read a small sample to detect binary quickly
    try:
        with path.open("rb") as f:
            sample = f.read(8192)
            if looks_binary(sample):
                stats.skipped_binary += 1
                return
    except OSError:
        return

    # Stream line-by-line; write to temp only if we actually change something.
    tmp_path = path.with_suffix(path.suffix + ".tmp_rewrite")
    changed = False
    file_reps = 0

    try:
        with path.open("r", encoding="utf-8", errors="strict", newline="") as fin:
            # optimistic: try UTF-8 first
            lines_iter = fin
            with tmp_path.open("w", encoding="utf-8", newline="") as fout:
                for line in lines_iter:
                    new_line, reps = rewrite_line(line)
                    if reps:
                        changed = True
                        file_reps += reps
                    fout.write(new_line)
    except UnicodeDecodeError:
        # fallback (keeps bytes-ish stable for weird encodings)
        try:
            with path.open("r", encoding="latin-1", newline="") as fin:
                with tmp_path.open("w", encoding="latin-1", newline="") as fout:
                    for line in fin:
                        new_line, reps = rewrite_line(line)
                        if reps:
                            changed = True
                            file_reps += reps
                        fout.write(new_line)
        except OSError:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            return
    except OSError:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        return

    if changed:
        stats.changed_files += 1
        stats.replacements += file_reps
        if dry_run:
            # don’t leave temp files around
            tmp_path.unlink(missing_ok=True)
            print(f"CHANGED ({file_reps:4d}): {path}", flush=True)
        else:
            # atomic-ish replace
            tmp_path.replace(path)
            print(f"CHANGED ({file_reps:4d}): {path}", flush=True)
    else:
        tmp_path.unlink(missing_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default=".", help="Root folder (default: .)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--include-hidden", action="store_true")
    ap.add_argument(
        "--include-ext",
        nargs="*",
        default=None,
        help="e.g. --include-ext .js .html .css",
    )
    ap.add_argument(
        "--all-files", action="store_true", help="Ignore extension filtering"
    )
    ap.add_argument(
        "--max-mb",
        type=float,
        default=2.0,
        help="Skip files larger than this (default 2MB)",
    )
    ap.add_argument(
        "--no-skip-minified", action="store_true", help="Do not skip .min/.bundle/etc"
    )
    ap.add_argument(
        "--verbose", action="store_true", help="Print every file as it is processed"
    )
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        print(f"Root does not exist: {root}", file=sys.stderr)
        return 2

    allowed_exts = None
    if not args.all_files:
        if args.include_ext is not None:
            allowed_exts = {
                e.lower() if e.startswith(".") else f".{e.lower()}"
                for e in args.include_ext
            }
        else:
            allowed_exts = set(DEFAULT_EXTS)

    max_bytes = int(args.max_mb * 1024 * 1024)

    skip_name_res: list[re.Pattern] = []
    if not args.no_skip_minified:
        skip_name_res = [
            re.compile(pat, re.IGNORECASE) for pat in DEFAULT_SKIP_NAME_PATTERNS
        ]

    stats = Stats()
    for p in iter_files(root, DEFAULT_SKIP_DIRS, allowed_exts, args.include_hidden):
        process_file(p, args.dry_run, max_bytes, skip_name_res, stats, args.verbose)

    print("\nSummary")
    print(f"Scanned files:     {stats.scanned}")
    print(f"Changed files:     {stats.changed_files}")
    print(f"Total replacements:{stats.replacements}")
    print(f"Skipped large:     {stats.skipped_large}")
    print(f"Skipped binary:    {stats.skipped_binary}")
    print(f"Skipped minified:  {stats.skipped_name}")
    if args.dry_run:
        print("(dry-run: no files written)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
