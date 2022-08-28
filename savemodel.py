import pickle
import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense
import numpy as np

dtst = tf.keras.datasets.mnist
(train_feature, train_target), (test_feature, test_target) = dtst.load_data()

train_feature = train_feature/255.0
test_feature = test_feature/255.0

model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28,1)))
model.add(tf.keras.layers.Dense(300, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(50, activation='relu'))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(10, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(0.0003),
              metrics=['accuracy'])

model.fit(train_feature, train_target, batch_size=32, epochs=20)
model.save("digitsnew.h5")
