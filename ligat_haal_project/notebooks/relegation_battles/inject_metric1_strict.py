"""Inject Metric 1 symmetric band + rank floor cell + markdown (METRIC1_SURVIVAL_BAND_PTS in setup)."""
import json
from pathlib import Path

NB = Path(__file__).resolve().parent / "02_relegation_competitiveness_analysis.ipynb"
SRC = Path(__file__).resolve().parent / "metric1_unified_cell.py"


def lines_from_text(text: str) -> list[str]:
    text = text.rstrip("\n")
    return [ln + "\n" for ln in text.split("\n")]


METRIC1_MD = """## Metric 1 — Relegation contenders index

At round approximately **75%** of the **regular** schedule, standings are rebuilt from completed matches (**sorted by points descending**).

This metric defines relegation contenders as teams within 6 points (**above or below**) the survival line (**last safe team**), **restricted to the bottom part of the table** (`rank_snap >= N - 6` after sorting by points), so top halves are excluded.

Season output prints **`lowest_safe_points`**, the **bottom six**, the **contenders** list with gaps, optional notes on exclusions by rank, and warnings when there are **no** contenders or **only one**.

"""


def main() -> None:
    code = SRC.read_text(encoding="utf-8")
    compile(code, str(SRC), "exec")
    nb = json.loads(NB.read_text(encoding="utf-8"))
    nb["cells"][8]["source"] = lines_from_text(METRIC1_MD)
    nb["cells"][9]["source"] = lines_from_text(code)
    NB.write_text(json.dumps(nb, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Updated cells 8 and 9")


if __name__ == "__main__":
    main()
