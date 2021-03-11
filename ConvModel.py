"""Convolutional tensorflow model baziran na kerasu. Izrađen je i služi treniranju za sustav prepoznavanja govornih
komandi.

    Ovaj projekt je izrađen u svrhu Završnog rada na preddiplomskom studiju Računarstva.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers

x_train = np.load("NumpyPodatci/KomandeTrening.npy", allow_pickle=True)
x_train = np.asarray(x_train).reshape((23778, 20, 36, 1))
y_train = np.load("NumpyPodatci/KomandeTreningLabels.npy")

x_test = np.load("NumpyPodatci/KomandeTesting.npy", allow_pickle=True)
x_test = np.asarray(x_test).reshape((1835, 20, 36, 1))
y_test = np.load("NumpyPodatci/KomandeTestingLabels.npy")

model = tf.keras.Sequential()
model.add(layers.Conv2D(32, kernel_size=(2, 2), activation='relu', input_shape=[20, 36, 1],
                        kernel_regularizer=regularizers.l2(0.001)))
model.add(layers.Dropout(0.5))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(32, kernel_size=(2, 2), activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(layers.Dropout(0.5))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(64, kernel_size=(2, 2), activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(layers.Dropout(0.5))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(10, activation='softmax'))
model.summary()

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=optimizers.Adam(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          validation_data=(x_test, y_test),
          batch_size=100,
          epochs=100)

results = model.evaluate(x_test, y_test, batch_size=100)
print(results)

model.save("ConvolutionalModel")
