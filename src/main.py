import pandas as pd

dataset = pd.read_csv("~/Projects/JobChange/input/raw.csv")
dataset.fillna(0, inplace=True)
print(dataset.target.value_counts())