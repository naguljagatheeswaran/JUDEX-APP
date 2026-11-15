# Judex v14

Judex is a classification/helper prototype for routing plain-language complaints to likely BNS sections. It is not a substitute for a lawyer and does not provide a definitive legal determination.

## What is included

- Existing Judex royal UI and background retained.
- Complaint classifier: 24 categories.
- BNS catalogue: all 358 sections.
- BNS training dataset: 28,640 examples (80%).
- BNS held-out test dataset: 7,160 examples (20%).
- Complaint-classifier training dataset: 2,880 examples (80%).
- Complaint-classifier held-out test dataset: 720 examples (20%).
- Every BNS section has 80 training and 20 held-out examples.
- Every complaint category has 120 training and 30 held-out examples.

## Testing

Run:

```bash
pytest -q
```

The complete BNS catalogue-coverage test and complaint-classifier held-out test now run on every normal pytest invocation; they are no longer marked as optional `slow` tests.

The test suite checks dataset splits, coverage of all 358 BNS sections and 24 complaint categories, training/test disjointness, the complaint-classifier held-out set, and the core matcher regression cases.

## Data caution

The expanded datasets are curated/synthetic development data intended for prototype evaluation. Passing these tests must not be interpreted as production or real-world legal accuracy.
