import tensorflow as tf
import pandas as pd
import numpy as np

number_of_trailing_chars = 4


def convert_features(x):
    features = []
    # len(x)
    for i in range(1, number_of_trailing_chars+1):
        if i <= len(x):
            features.append(ord(x[-i]))
        else:
            features.append(0)
    return pd.Series(features)


def convert_label(x):
    if 'm' in x:
        return 0
    if 'f' in x:
        return 1
    if 'n' in x:
        return 2


def load_data():
    """Parses the csv file """

    file_path = './parsed-data.txt'

    # Parse the local CSV file.
    df = pd.read_csv(filepath_or_buffer=file_path, sep=';',
                     # usecols=[0,1],
                     header=None, #nrows=30,
                     converters={1: convert_label}
                     )

    df2 = df[0].apply(convert_features)
    df2[number_of_trailing_chars] = df[1]
    df2 = df2.sample(frac=1)

    nsplit = int(0.8*len(df.index))

    # Return four DataFrames.
    return (df2[0:nsplit], df2[nsplit:])


train_data, test_data = load_data()

my_batch_size = 50

my_feature_columns = []
for i in range(number_of_trailing_chars):
    my_feature_columns.append(tf.feature_column.numeric_column(key=str(i)))
print(my_feature_columns)

classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    hidden_units=[50, 50],
    n_classes=3)


def train_input_fn(dataframe, batch_size):
    features = {}
    for i in range(number_of_trailing_chars):
        features[str(i)] = tf.convert_to_tensor(
            dataframe[i].as_matrix(), dtype=tf.int32)

    inputs = (features,
              tf.convert_to_tensor(
                  dataframe[number_of_trailing_chars].as_matrix(), dtype=tf.int32)
              )
    dataset = tf.data.Dataset.from_tensor_slices(inputs)
    dataset = dataset.shuffle(buffer_size=1000).repeat(
        count=None).batch(batch_size)
    return dataset.make_one_shot_iterator().get_next()


print("starting training of classifier....")
classifier.train(
    input_fn=lambda: train_input_fn(train_data, my_batch_size),
    steps=10000)
print('finished')


def eval_input_fn(features, labels=None, batch_size=None):
    myfeatures = {}
    # No labels, use only features.
    for i in range(number_of_trailing_chars):
        myfeatures[str(i)] = tf.convert_to_tensor(
            features[i].as_matrix(), dtype=tf.int32)
    if labels is None:
        inputs = myfeatures
    if labels is not None:
        inputs = (myfeatures,
                  tf.convert_to_tensor(
                      labels.as_matrix(), dtype=tf.int32)
                  )
    dataset = tf.data.Dataset.from_tensor_slices(inputs)
    dataset = dataset.batch(batch_size)
    return dataset.make_one_shot_iterator().get_next()


print("evaluating classifier....")
eval_result = classifier.evaluate(
    input_fn=lambda: eval_input_fn(test_data[[i for i in range(number_of_trailing_chars)]], test_data[[number_of_trailing_chars]], my_batch_size))

print(eval_result)

predictions = classifier.predict(
    input_fn=lambda: eval_input_fn(test_data[[i for i in range(number_of_trailing_chars)]][:10], None, batch_size=my_batch_size))

for i in predictions:
    print(i)
