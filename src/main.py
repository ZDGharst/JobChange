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
dataset.fillna(0, inplace = True)

features_to_encode = ['gender', 'relevent_experience', 'enrolled_university', 'education_level',
'major_discipline', 'company_size', 'company_type', 'last_new_job']
for feature in features_to_encode:
    dataset = one_hot_encode(dataset, feature)

dataset = normalize(dataset, "training_hours")
dataset = normalize(dataset, "experience")

# Shuffle rows in the dataset, then split into train and test data.
# By doing it this way, we avoid converting to a numpy with np.split()
dataset  = dataset.sample(frac = 1)
training = dataset.sample(frac = .85, random_state = None)
testing  = dataset.drop(training.index)

# Split labels off into their own dataframe
training_labels = training.pop('target')
testing_labels  = testing.pop('target')
training_labels = tf.keras.utils.to_categorical(training_labels)
testing_labels  = tf.keras.utils.to_categorical(testing_labels )

# print(training.shape)
# print(training.head(5))

model = tf.keras.Model
model = tf.keras.Sequential([
        tf.keras.layers.Dense(49,activation="relu"),
        tf.keras.layers.Dense(64,activation="relu"),
        tf.keras.layers.Dense(64,activation="relu"),
        tf.keras.layers.Dense(2,activation="softmax")
])
model.compile(optimizer='adam',loss='mean_squared_error', metrics=['accuracy'])
model.fit(
    training,
    training_labels,
    epochs = 30,
    validation_split = 0.176,
    batch_size = 2
)