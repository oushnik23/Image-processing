# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 12:05:35 2022

@author: Administrator
"""

!pip install cv2

import os
os.getcwd()
os.chdir(r"C:\Users\Administrator\Desktop\PYTHON\Deep learning")

import numpy as np
import cv2

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

dataset=ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True,validation_split=.2)

train_data=dataset.flow_from_directory('brain_tumor_dataset 2/dataset',
                                 target_size = (64, 64),
                                 batch_size = 32,class_mode = 'binary',subset='training')

test_data=dataset.flow_from_directory('brain_tumor_dataset 2/dataset',
                                      target_size = (64, 64),
                                 batch_size = 32,class_mode = 'binary',subset='validation')


cnn=tf.keras.models.Sequential()
cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation='relu',input_shape=[64,64,3]))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))
cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))
cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(units=132,activation='relu'))
cnn.add(tf.keras.layers.Dense(units=1,activation='sigmoid'))
cnn.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
cnn.summary()
cnn.fit(x=train_data,validation_data=test_data,epochs=25)

from keras.preprocessing import image
test_image=image.load_img('brain_tumor_dataset 2/pred/pred15.jpg',target_size = (64, 64))
test_image2=image.array_to_img(test_image)
test_image2=np.expand_dims(test_image2,axis=0)
result=cnn.predict(test_image2)
train_data.class_indices

if result[0][0]==1:
    prediction='yes'
else:
    prediction='no'

print(prediction)
