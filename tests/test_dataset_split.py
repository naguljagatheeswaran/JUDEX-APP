import csv
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data'

def test_bns_dataset_is_exact_80_20_and_covers_all_sections():
    all_rows=list(csv.DictReader((DATA/'curated_all.csv').open(encoding='utf-8',newline='')))
    train=list(csv.DictReader((DATA/'curated_training.csv').open(encoding='utf-8',newline='')))
    test=list(csv.DictReader((DATA/'heldout_test.csv').open(encoding='utf-8',newline='')))
    assert len(all_rows)==35800 and len(train)==28640 and len(test)==7160
    assert len(train)/len(all_rows)==0.8 and len(test)/len(all_rows)==0.2
    assert set(int(r['section']) for r in train)==set(int(r['section']) for r in test)==set(range(1,359))
    assert set((r['complaint'].lower(),r['section']) for r in train).isdisjoint(set((r['complaint'].lower(),r['section']) for r in test))
    assert all(v==80 for v in Counter(int(r['section']) for r in train).values())
    assert all(v==20 for v in Counter(int(r['section']) for r in test).values())
