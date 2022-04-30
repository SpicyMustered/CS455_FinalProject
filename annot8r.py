# Import dataset
from tensorflow.keras.layers import TextVectorization
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import string
import re

model = tf.keras.models.load_model('currentModel')
#print(model.evaluate(test_ds, verbose=True))

def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, "<br />", " ")
    return tf.strings.regex_replace(
        stripped_html, f"[{re.escape(string.punctuation)}]", ""
    )

# Model constants.
max_features = 20000
embedding_dim = 128
sequence_length = 500

vectorize_layer = TextVectorization(
    standardize=custom_standardization,
    max_tokens=max_features,
    output_mode="int",
    output_sequence_length=sequence_length,
)

#input = np.array(["Although cell phones caused my mother to leave, most phones are bad", 1])
input = np.array(["They're not the number one killer of mice in the U.S. as of 2009", 1])
#input = np.array(["They're in my head they're all in my head oh god oh jesus", 1])
#input = np.array(["26% of cell phones on the moon are a negative health risk to Americans"])
#input = np.array(["In conclusion, there is no way I'm moving anywhere near a cell tower after this.", 1])

def PreProcess(input):
  numpy_train = (input,np.asarray(1).astype('int32'))

  raw_train_ds = tf.data.Dataset.from_tensors(numpy_train)


  vectorize_layer = TextVectorization(
    standardize=custom_standardization,
    max_tokens=max_features,
    output_mode="int",
    output_sequence_length=sequence_length,
  )

  text_ds = raw_train_ds.map(lambda x, y: x)
  #print(text_ds.as_numpy_iterator())
  vectorize_layer.adapt(text_ds)

  train_ds = text_ds.map(vectorize_layer)



  # Test it with `raw_test_ds`, which yields raw strings
  result = model.predict(train_ds)[0]
  print(result)

  tag_list = ["Lead","Position","Claim","Counterclaim","Rebuttal","Evidence","Concluding Statement"]

  max_value = max(result)


  print(tag_list[int(np.where(result==max_value)[0])])

PreProcess(input)