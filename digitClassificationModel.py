# SIMPLE NEURAL NETWORK FOR DIGIT CLASSIFICATION 
# PLEASE RUN THE MODEL TILL THE TEST ACCURACY IS ATLEAST 98% FOR OBTAINING CORRECT RESULTS

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import cv2

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train/255.0
x_test = x_test/255.0

x_train_flatten = x_train.reshape(len(x_train),28*28)
x_test_flatten = x_test.reshape(len(x_test),28*28)

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(392, activation='relu'),
    keras.layers.Dense(196, activation='relu'),
    keras.layers.Dense(98, activation='relu'),
    keras.layers.Dense(49, activation='relu'),
    keras.layers.Dense(10, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(x_train, y_train, epochs=10)   
model.evaluate(x_test, y_test)

#cv2.waitKey(0)
model.save("model.h5")
print("Digit classification model saved to the disk successfully!")
