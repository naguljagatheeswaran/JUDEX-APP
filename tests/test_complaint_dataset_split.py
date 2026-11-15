import pandas as pd
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data'

def test_complaint_dataset_is_exact_80_20_and_disjoint():
    all_df=pd.read_csv(DATA/'complaint_all.csv'); train=pd.read_csv(DATA/'complaint_training.csv'); test=pd.read_csv(DATA/'complaint_test.csv')
    assert len(all_df)==3600 and len(train)==2880 and len(test)==720
    assert len(train)/len(all_df)==0.8 and len(test)/len(all_df)==0.2
    assert set(train.category)==set(test.category)==set(all_df.category) and len(set(train.category))==24
    assert set(train.complaint).isdisjoint(set(test.complaint))
    assert all(train.groupby('category').size()==120)
    assert all(test.groupby('category').size()==30)
