import glob
import logging
import os
import sys
import shutil

KB = 1024
MAX_FILE_SIZE = 101376 * KB  # approx 100 MB (99 exactly)

logging.basicConfig(level=logging.INFO)

# Safe handling of args
work_dir = ""
if len(sys.argv) > 1:
    work_dir = sys.argv[1].rstrip("/") + "/"

print(sys.argv)


def move_files_from_build_to_root():
    source = f"{work_dir}Build"
    destination = f"{work_dir}"

    if not os.path.exists(source):
        logging.warning(f"{source} directory not found. skip file moving")
        return

    for f in os.listdir(source):
        src = os.path.join(source, f)
        dst = os.path.join(destination, f)
        shutil.move(src, dst)

    os.rmdir(source)


def split_file(path: str):
    with open(path, "rb") as f:
        n = 0
        while True:
            buf = f.read(MAX_FILE_SIZE)
            if not buf:
                break

            split_file_name = f"{path}.{str(n).rjust(2, '0')}"
            logging.info(f"create file {split_file_name}")

            with open(split_file_name, "wb") as sp_f:
                sp_f.write(buf)

            n += 1


def glob_local(pattern: str) -> list[str]:
    """Perform glob relative to work_dir and return filenames only."""
    full_pattern = work_dir + pattern
    results = glob.glob(full_pattern)
    return [os.path.basename(r) for r in results]


def try_split_file(file_pattern: str) -> list[str]:
    logging.info(f"split {file_pattern}")

    files = glob_local(file_pattern)

    # Original files = no numerical .xx suffix
    source_files = [f for f in files if not f.split(".")[-1].isdigit()]

    if len(source_files) > 1:
        raise Exception(
            f"expected one '{file_pattern}' file, but found {len(source_files)}: {source_files}"
        )

    # Already-split files
    split_files = [f for f in files if f.split(".")[-1].isdigit()]

    # If original exists & old split files exist → remove old split files
    if len(source_files) == 1 and split_files:
        logging.info(f"found {len(split_files)} already split files. Removing them...")
        for f in split_files:
            logging.info(f"remove {f}")
            os.remove(work_dir + f)

    if len(source_files) == 1:
        source_file = source_files[0]
        full_path = work_dir + source_file

        # No split needed
        if os.path.getsize(full_path) <= MAX_FILE_SIZE:
            logging.info("split not required")
            return [source_file]

        # Split
        logging.info(f"split {source_file}")
        split_file(full_path)

        logging.info(f"remove original {source_file}")
        os.remove(full_path)

        # Reload split files
        split_files = [
            f for f in glob_local(f"{source_file}.*") if f.split(".")[-1].isdigit()
        ]

    if not split_files:
        raise Exception(f"not found original or split files for pattern {file_pattern}")

    return split_files


def split_binary_files_if_need() -> dict[str, list[str]]:
    return {"dataUrl": try_split_file("*.data*"), "codeUrl": try_split_file("*.wasm*")}


def replace_url(
    source: str, mapped_files: dict[str, list[str]], replaced_key: str
) -> str:
    if replaced_key not in mapped_files:
        logging.warning(f"replace key `{replaced_key}` not found. Skip.")
        return source

    value = mapped_files[replaced_key]
    replacement = f'"{value[0]}"' if len(value) == 1 else str(value)

    key_pattern = f"{replaced_key}: "

    index = source.find(key_pattern)
    if index == -1:
        logging.warning(f"position not found for replace {replaced_key}")
        return source

    # find next comma after the key
    end = source.find(",", index)
    if end == -1:
        end = len(source)

    logging.info(f"replace `{replaced_key}` with `{replacement}`")

    return source[:index] + f"{replaced_key}: {replacement}" + source[end:]


def modify_index_html(mapped_files: dict[str, list[str]]):
    logging.info("modify index.html")

    index_path = work_dir + "index.html"

    with open(index_path, "r", encoding="utf-8") as f:
        source = f.read()

    # Remove Build/ prefix always
    source = source.replace("Build/", "")

    source = replace_url(source, mapped_files, "dataUrl")
    source = replace_url(source, mapped_files, "codeUrl")

    with open(index_path, "w", encoding="utf-8") as f:
        f.write(source)


move_files_from_build_to_root()
mapped_files = split_binary_files_if_need()
modify_index_html(mapped_files)

logging.info("completed")
