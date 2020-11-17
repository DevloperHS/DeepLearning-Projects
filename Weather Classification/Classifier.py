# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 13:26:09 2020

@author: 91736
"""

import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import os
import glob
from PIL import Image
import matplotlib.pyplot as plt

# setting data path
base_dir =  r'C:/Users/91736/Documents/PROJECTS/Weather Classification/Dataset'

# 
train_datagen = ImageDataGenerator(rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2) # set validation split

train_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=(250 , 250),
    batch_size= 32,
    class_mode='categorical',
    subset='training') # set as training data

validation_generator = train_datagen.flow_from_directory(
    base_dir, # same directory as training data
    target_size=(250, 250),
    batch_size= 32,
    class_mode='categorical',
    subset='validation') # set as validation data


# Generated Labels
labels = (train_generator.class_indices)
labels = dict((v , k) for k , v in labels.items())
print(labels)
num_classes = 4

# defining model
tf.keras.backend.clear_session()
tf.random.set_seed(51)
np.random.seed(51)



model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (5,5), input_shape = (250 , 250 , 3), activation = 'relu'),
    tf.keras.layers.MaxPool2D(3,3),
    tf.keras.layers.Conv2D(64, (5,5) , activation = 'relu'),
    tf.keras.layers.MaxPool2D((3,3)),
    tf.keras.layers.Conv2D(128 , (5,5) , activation = 'relu'),
    tf.keras.layers.MaxPool2D(3,3),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.2, seed = 5),
    tf.keras.layers.Dense(128 , activation = 'relu'),
    tf.keras.layers.Dense(num_classes , activation = 'softmax')])

model.compile(loss = 'categorical_crossentropy' , optimizer = 'Adam' , metrics = 'accuracy')

model.summary()

history = model.fit(train_generator ,
          validation_data = validation_generator ,
          epochs = 25 ,
          steps_per_epoch = 901/32,
          validation_steps = 224/32)

#plotting 




# Testing
def prediction(test_path):
    img = image.load_img(test_path , target_size = (250 , 250))
    img = image.img_to_array(img, dtype=np.uint8)
    img = np.array(img)/255.0
        
    plt.title('Image')
    plt.axis('off')
    plt.imshow(img.squeeze())
        
    predict = model.predict(img[np.newaxis , ...])
    predicted_class = labels[np.argmax(predict[0] , axis = -1)]
        
    print('Prediction Value: ' , np.max(predict[0] , axis = -1))
    print("Classified:",predicted_class)


# plotting graphs
def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.title(string)
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()

plot_graphs(history, "accuracy")
plot_graphs(history, "loss")

C  = r'C:/Users/91736/Documents/PROJECTS/Weather Classification/Test Folder/Clouds.jpg'
R   = r'C:/Users/91736/Documents/PROJECTS/Weather Classification/Test Folder/Rainy.jpg'
S = r'C:/Users/91736/Documents/PROJECTS/Weather Classification/Test Folder/Sunrise.jpg'
CH = r'C:/Users/91736/Documents/PROJECTS/Weather Classification/Test Folder/Chance.jpg'
img = r'C:/Users/91736/Documents/PROJECTS/Weather Classification/Test Folder/IMG_20201116_091732.jpg'


prediction(C)
prediction(R)
prediction(S)
prediction(CH)


test_path = r'C:/Users/91736/Documents/PROJECTS/Weather Classification/Test Folder'

for files in test_path:
    print(files)

