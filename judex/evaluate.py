"""Small offline evaluation harness for the Judex capstone prototype."""
from __future__ import annotations

import csv
from pathlib import Path
from judex.matcher import BNSMatcher


def evaluate(path: str | Path) -> tuple[int, int, float]:
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    matcher = BNSMatcher()
    correct = 0
    for row in rows:
        expected = int(row["section"])
        top = matcher.rank(row["complaint"], limit=3)
        predicted = [m.section.number for m in top]
        correct += expected == predicted[0]
    total = len(rows)
    return correct, total, (correct / total if total else 0.0)


if __name__ == "__main__":
    p = Path(__file__).parent.parent / "tests" / "data" / "evaluation_cases.csv"
    correct, total, accuracy = evaluate(p)
    print(f"Top-1: {correct}/{total} ({accuracy:.1%})")
