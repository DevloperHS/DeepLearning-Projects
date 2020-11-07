# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 18:54:22 2020

@author: 91736
"""

import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"    
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Model
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import  VGG16
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing import image
import numpy as np


# defining directories
base_dir = r'C:/Users/91736/Documents/PROJECTS/Waste Classifier/Data Files/archive/DATASET'

train_dir = os.path.join(base_dir , 'TRAIN')
valid_dir = os.path.join(base_dir , 'TEST')

# defining organinc(O) or reduced(R) directories
train_org_dir = os.path.join(train_dir , 'O')
train_red_dir = os.path.join(train_dir , 'R')

test_org_dir = os.path.join(valid_dir , 'O')
test_red_dir = os.path.join(valid_dir , 'R')

# defining genrators

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   rotation_range = 40,
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2 ,
                                   zoom_range = 0.2 ,
                                   shear_range= 0.2 ,
                                   horizontal_flip= True
                                   )


train_generator = train_datagen.flow_from_directory(train_dir ,
                                                   target_size = (224 , 224),
                                                   batch_size = 20,
                                                   class_mode = 'binary')

valid_datagen = ImageDataGenerator(rescale = 1./255)

valid_generator = valid_datagen.flow_from_directory(valid_dir,
                                                    target_size = (224, 224),
                                                    batch_size = 20,
                                                    class_mode = 'binary')


# defining model
base_model = VGG16(input_shape = (224 , 224 , 3) , include_top = False , weights = 'imagenet')

# freezing traning for pre defined model layers
for layer in base_model.layers:
    layer.trainable = False

# seeding
tf.keras.backend.clear_session()
tf.random.set_seed(15)
np.random.seed(15)    

# Defining Model As Per Need
x = layers.Flatten()(base_model.output) # flatten values

x = layers.Dense(2024 , activation = 'relu')(x) # adding 1st dense layer

x = layers.Dropout(0.5)(x) # adding drop out regularisation

x = layers.Dense(1 , activation = 'sigmoid')(x) # adding last layer

# creating model from all parts
model = tf.keras.models.Model(base_model.input , x)


optimizer_r = RMSprop(lr = 0.001)

model.compile(loss = 'binary_crossentropy' , optimizer = optimizer_r , metrics = ['accuracy'])


# training_model
history = model.fit(train_generator,
                    epochs = 15,
                    steps_per_epoch = int(100),
                    )


acc = history.history['accuracy']
#val_acc = history.history['val_accuracy']

loss = history.history['loss']
#val_loss = history.history['val_loss']

epoch = range(len(acc))

plt.title('Model Performance')
plt.plot(epoch , acc , 'b' )
plt.plot(epoch , loss , 'r')
#plt.plot(epoch , val_acc  , 'r')
plt.legend(['acc', 'loss'])
plt.figure()





test_path1 = r'C:/Users/91736/Documents/PROJECTS/Waste Classifier/Data Files/archive/test_imgs/O_12774.jpg'
test_path2 = r'C:/Users/91736/Documents/PROJECTS/Waste Classifier/Data Files/archive/test_imgs/R_10000.jpg'


def prediction_num(path):
    img = image.load_img(path, target_size=(224 , 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    
    images = np.vstack([x])
    classes = model.predict(images)
    if classes[0]<0.5:
        print('Organic')
    else:
        print('Reduced')

print(prediction_num(test_path1))
print(prediction_num(test_path2))


prediction = model.evaluate(valid_generator)
print(prediction)



'''
plt.title('Model Loss')
plt.plot(epoch , loss , 'b')
plt.plot(epoch , val_loss  , 'r')
plt.legend(['loss' , 'val_loss'])
plt.figure()
'''