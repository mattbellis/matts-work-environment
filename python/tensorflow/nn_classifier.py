from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import urllib

import numpy as np
import tensorflow as tf

# Data sets
IRIS_TRAINING = "iris_training.csv"
IRIS_TRAINING_URL = "http://download.tensorflow.org/data/iris_training.csv"

IRIS_TEST = "iris_test.csv"
IRIS_TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

def main():
  # If the training and test sets aren't stored locally, download them.
  if not os.path.exists(IRIS_TRAINING):
    raw = urllib.request.urlopen(IRIS_TRAINING_URL).read()
    with open(IRIS_TRAINING, "w") as f:
      f.write(raw.decode())

  if not os.path.exists(IRIS_TEST):
    raw = urllib.request.urlopen(IRIS_TEST_URL).read()
    with open(IRIS_TEST, "w") as f:
      f.write(raw.decode())

  # Load datasets.
  training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
      filename=IRIS_TRAINING,
      target_dtype=np.int,
      features_dtype=np.float32)
  test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
      filename=IRIS_TEST,
      target_dtype=np.int,
      features_dtype=np.float32)

  # Specify that all features have real-value data
  feature_columns = [tf.feature_column.numeric_column("x", shape=[4])]

  # Build 3 layer DNN with 10, 20, 10 units respectively.
  classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                          hidden_units=[10, 20, 10],
                                          n_classes=3,
                                          model_dir="/tmp/iris_model")
  # Define the training inputs
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": np.array(training_set.data)},
      y=np.array(training_set.target),
      num_epochs=None,
      shuffle=True)

  # Train model.
  classifier.train(input_fn=train_input_fn, steps=2000)

  ########### DONE WITH TRAINING!!!! ######################

  ##########  NOW ON TO TESTING HOW GOOD IT DID!!!! ######

  # Define the test inputs
  test_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": np.array(test_set.data)},
      y=np.array(test_set.target),
      num_epochs=1,
      shuffle=False)

  # Evaluate accuracy.
  score = classifier.evaluate(input_fn=test_input_fn)
  #accuracy_score = classifier.evaluate(input_fn=test_input_fn)["accuracy"]

  #score = classifier.evaluate(input_fn=input_fn_test, steps=1)
  score_accuracy = score["accuracy"]
  score_loss = score["loss"]
  
  print()
  print("Score: ", score)
  print("Accuracy: ", score_accuracy)
  print("Loss: ", score_loss)
  print()
  #print()
  #print("\nTest Accuracy: {0:f}\n".format(accuracy_score))
  #print()

  ########### NOW TRY IT ON DATA THAT WE DON'T KNOW THE ANSWER TO ###############
  # Classify two new flower samples.
  new_samples = np.array(
      [[6.4, 3.2, 4.5, 1.5],
       [5.8, 3.1, 5.0, 1.7]], dtype=np.float32)
  predict_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": new_samples},
      num_epochs=1,
      shuffle=False)

  predictions = list(classifier.predict(input_fn=predict_input_fn))
  predicted_classes = [p["classes"] for p in predictions]

  # probability of being predicted as 1
  y_prob = [p["probabilities"][1] for p in predictions]
  x_prob = [p["probabilities"][2] for p in predictions]
  z_prob = [p["probabilities"][0] for p in predictions]

  print()
  print(len(predictions))
  print(predictions)
  print("probabilities")
  print(z_prob)
  print(y_prob)
  print(x_prob)
  print()
  print(
      "New Samples, Class Predictions:    {}\n"
      .format(predicted_classes))

  ########### NOW TRY IT ON THE TEST DATA 1 at a time ######################
  # 
  testdata = np.loadtxt(IRIS_TEST,skiprows=1,dtype=float,delimiter=',')
  #print(testdata)

  for d in testdata:
      new_samples = np.array( [d[0:4]] , dtype=np.float32)
      predict_input_fn = tf.estimator.inputs.numpy_input_fn(
          x={"x": new_samples},
          num_epochs=1,
          shuffle=False)

      predictions = list(classifier.predict(input_fn=predict_input_fn))
      predicted_classes = [p["classes"] for p in predictions]
      pred = predictions[0]["classes"][0].decode()
      probs = []
      for i in range(0,3):
          probs.append(predictions[0]["probabilities"][i])

      print('pred/actual: ',pred,int(d[4]),probs[0],probs[1],probs[2])

      #print()
      #print(len(predictions))
      #print(predictions)
      #print()
      #print(
          #"New Samples, Class Predictions:    {}\n"
          #.format(predicted_classes))

if __name__ == "__main__":
    main()
