import tensorflow as tf
import pandas as pd

# Thank you to https://stackoverflow.com/questions/37292872/how-can-i-one-hot-encode-in-python!
def one_hot_encode(data, feature):
    dummy = pd.get_dummies(data[[feature]])
    result = pd.concat([data, dummy], axis = 1)
    result = result.drop([feature], axis = 1)
    return(result)

# https://stackoverflow.com/questions/26414913/normalize-columns-of-pandas-data-frame
def normalize(data, feature):
    result = data.copy()
    max_value = data[feature].max()
    min_value = data[feature].min()
    if(max_value != min_value):
        result[feature] = (data[feature] - min_value) / (max_value - min_value)
    return(result)

dataset = pd.read_csv("~/Projects/JobChange/input/raw.csv")
dataset.fillna(0, inplace=True)

features_to_encode = ['gender', 'relevent_experience', 'enrolled_university', 'education_level',
'major_discipline', 'company_size', 'company_type', 'last_new_job']
for feature in features_to_encode:
    dataset = one_hot_encode(dataset, feature)

dataset = normalize(dataset, "training_hours")
dataset = normalize(dataset, "experience")

dataset.sample(frac=1)
train_data = dataset.sample(frac=.85,random_state=None)
test_data = dataset.drop(train_data.index)