from pathlib import Path
import pandas as pd
from judex.classifiers import TopicClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'


def test_complaint_training_is_not_the_heldout_test():
    train = pd.read_csv(DATA / 'complaint_training.csv')
    test = pd.read_csv(DATA / 'complaint_test.csv')
    assert set(train['complaint']).isdisjoint(set(test['complaint']))


def test_classifier_can_train_from_only_training_csv():
    clf = TopicClassifier(training_data_path=str(DATA / 'complaint_training.csv'))
    assert set(clf.model.classes_) == set(train_labels())


def train_labels():
    return pd.read_csv(DATA / 'complaint_training.csv')['category'].unique().tolist()
