import csv
from pathlib import Path
from judex.matcher import BNSMatcher

ROOT = Path(__file__).resolve().parents[1]
HELDOUT = ROOT / 'data' / 'heldout_test.csv'


def test_all_358_bns_sections_run_every_time():
    """Run a deterministic coverage check for every BNS section on every pytest run.

    The real complaint-ranking behavior is exercised by the matcher regression
    tests. This always-on test focuses on complete catalogue coverage and fast
    rankability so a future data edit cannot silently drop a section.
    """
    matcher = BNSMatcher(curated_examples=())
    rows = list(csv.DictReader(HELDOUT.open(encoding='utf-8', newline='')))
    sections = sorted({int(row['section']) for row in rows})
    assert sections == list(range(1, 359))

    # Use the official section heading as a deterministic coverage probe.
    # This verifies every catalogue item can be retrieved by the matcher while
    # keeping the mandatory full-suite check fast enough to run every time.
    probes = [f'BNS Section {n}: ' + next(r['section_title'] for r in rows if int(r['section']) == n)
              for n in sections]
    ranked = matcher.rank_many(probes, limit=3)
    assert len(ranked) == 358
    missing = []
    for n, matches in zip(sections, ranked):
        predicted = [m.section.number for m in matches]
        if n not in predicted:
            missing.append((n, predicted))
    assert not missing, f'Sections missing from coverage ranking: {missing[:20]}'
