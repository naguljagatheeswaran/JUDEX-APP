"""CSV-backed BNS catalogue for Judex.

The catalogue is data-driven: section numbers, official section headings,
categories, keywords and training examples live in bns_sections.csv rather
than being hard-coded as a giant Python tuple.
"""
from __future__ import annotations

import ast
import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

OFFICIAL_BNS_URL = "https://www.indiacode.nic.in/handle/123456789/20062"
CSV_PATH = Path(__file__).with_name("bns_sections.csv")

@dataclass(frozen=True, slots=True)
class BNSSection:
    number: int
    title: str
    category: str
    keywords: dict[str, float]
    examples: tuple[str, ...]

@lru_cache(maxsize=1)
def _load_sections(path: str = str(CSV_PATH)) -> tuple[BNSSection, ...]:
    sections: list[BNSSection] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            sections.append(BNSSection(
                number=int(row["#Sections"]),
                title=row["Section name"].strip(),
                category=row["category"].strip(),
                keywords=ast.literal_eval(row["keywords"]),
                examples=tuple(x.strip() for x in row["examples"].split(" || ") if x.strip()),
            ))
    sections.sort(key=lambda s: s.number)
    return tuple(sections)

SECTIONS = _load_sections()
SECTION_BY_NUMBER = {s.number: s for s in SECTIONS}
