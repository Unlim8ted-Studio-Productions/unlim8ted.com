# Tools Folder

This directory mixes active scripts, AI training assets, generated outputs, and older experiments.

Current layout:

- `fragments/`: fragment source data used by Meatball AI workflows.
- `meatball ai/`: AI training and evaluation scripts.
- `testing+unused/`: old experiments and throwaway test assets.
- `docs/`: notes, prompts, and reference text files.
- `generated/`: generated JSON/TXT artifacts that scripts write out.
- `media/`: loose image assets that are not part of the main site.
- `samples/`: example input files for manual runs.

Conventions:

- Keep runnable utility scripts at `tools/` root unless you also update any hardcoded paths or docs that call them directly.
- Put new generated artifacts in `tools/generated/`.
- Put scratch notes, prompts, and text dumps in `tools/docs/`.
- Avoid adding more one-off assets to the root when a category folder exists.
