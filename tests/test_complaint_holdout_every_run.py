from pathlib import Path
import pandas as pd
from judex.classifiers import TopicClassifier

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'

def test_complaint_classifier_holdout_runs_every_time():
    clf=TopicClassifier(training_data_path=str(DATA/'complaint_training.csv'))
    test=pd.read_csv(DATA/'complaint_test.csv')
    correct=0
    for row in test.itertuples(index=False):
        pred=clf.predict(row.complaint)['category']
        correct += pred == row.category
    accuracy=correct/len(test)
    # This dataset is synthetic/curated for project testing; this threshold guards
    # against accidental training-pipeline breakage without claiming real-world accuracy.
    assert accuracy >= 0.80, f'holdout accuracy too low: {accuracy:.1%}'
