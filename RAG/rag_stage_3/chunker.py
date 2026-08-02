"""Stage 3 — window resized sections into chunks.json.

Thin wrapper: imports the frozen stage-2 chunker's `build_children` (same
120-word / 20-overlap windowing) and runs it on the RESIZED stage-3 sections.
No window logic is duplicated. Stage 2 is untouched.

Run:  python chunker.py     # build stage_3/chunks.json + size report
"""

import importlib.util
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
STAGE2 = HERE.parent / "rag_stage_2"
SECTIONS_PATH = HERE / "sections.json"
OUT_PATH = HERE / "chunks.json"


def _load_s2_chunker():
    spec = importlib.util.spec_from_file_location(
        "s2_chunker", STAGE2 / "chunker.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    if not SECTIONS_PATH.exists():
        raise SystemExit(f"{SECTIONS_PATH.name} missing — run pdf_extractor.py first")
    s2 = _load_s2_chunker()
    sections = json.loads(SECTIONS_PATH.read_text())
    children = s2.build_children(sections)
    OUT_PATH.write_text(json.dumps(children, indent=2, ensure_ascii=False))

    per_parent = Counter(c["parent_id"] for c in children)
    print(f"Parents (resized sections): {len(sections)}")
    print(f"Children (chunks):          {len(children)}")
    print(f"Children/parent: min {min(per_parent.values())} "
          f"max {max(per_parent.values())} avg {len(children)/len(per_parent):.1f}")
    print(f"Wrote {len(children)} children -> {OUT_PATH.name}")


if __name__ == "__main__":
    main()
