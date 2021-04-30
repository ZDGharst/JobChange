import pandas as pd

# Thank you to https://stackoverflow.com/questions/37292872/how-can-i-one-hot-encode-in-python!
def one_hot_encode(data, feature):
    dummy = pd.get_dummies(data[[feature]])
    result = pd.concat([data, dummy], axis = 1)
    result = result.drop([feature], axis = 1)
    return(result)


dataset = pd.read_csv("~/Projects/JobChange/input/raw.csv")
dataset.fillna(0, inplace=True)

features_to_encode = ['gender', 'relevent_experience', 'enrolled_university', 'education_level',
'major_discipline', 'company_size', 'company_type', 'last_new_job']
for feature in features_to_encode:
    dataset = one_hot_encode(dataset, feature)

print(dataset.head(5))