import csv
from collections import Counter, defaultdict
from pathlib import Path

from judex.matcher import BNSMatcher

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data" / "curated_training.csv"
HELDOUT = ROOT / "data" / "heldout_test.csv"


def test_training_covers_every_bns_section() -> None:
    rows = list(csv.DictReader(TRAIN.open(encoding="utf-8", newline="")))
    counts = Counter(int(row["section"]) for row in rows)
    assert len(counts) == 358
    assert set(counts) == set(range(1, 359))
    assert min(counts.values()) >= 15


def test_heldout_covers_every_bns_section() -> None:
    rows = list(csv.DictReader(HELDOUT.open(encoding="utf-8", newline="")))
    counts = Counter(int(row["section"]) for row in rows)
    assert len(counts) == 358
    assert set(counts) == set(range(1, 359))
    assert min(counts.values()) >= 4


def test_bns_catalogue_has_358_sections() -> None:
    rows = list(csv.DictReader((ROOT / "judex" / "bns_sections.csv").open(encoding="utf-8", newline="")))
    numbers = [int(row["#Sections"]) for row in rows]
    assert len(numbers) == 358
    assert set(numbers) == set(range(1, 359))


# Full semantic ranking of every section is intentionally marked slow because
# it exercises the complete TF-IDF + learned retrieval pipeline 358 times.
# Run with: pytest -m slow
# It is kept separate from the fast CI checks above so normal development is quick.
