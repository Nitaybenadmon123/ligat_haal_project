"""Inject Metric 1 code cell from metric1_unified_cell.py (keeps METRIC1_RELEG_GAP_MAX in setup)."""
import json
from pathlib import Path

NB = Path(__file__).resolve().parent / "02_relegation_competitiveness_analysis.ipynb"
SRC_PATH = Path(__file__).resolve().parent / "metric1_unified_cell.py"


def to_src_lines(text: str) -> list[str]:
    text = text.rstrip("\n")
    return [ln + "\n" for ln in text.split("\n")]


def main() -> None:
    code = SRC_PATH.read_text(encoding="utf-8")
    compile(code, str(SRC_PATH), "exec")
    nb = json.loads(NB.read_text(encoding="utf-8"))
    nb["cells"][9]["source"] = to_src_lines(code)
    NB.write_text(json.dumps(nb, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Wrote cell 9 from metric1_unified_cell.py")


if __name__ == "__main__":
    main()
