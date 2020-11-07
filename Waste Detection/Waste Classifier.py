# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 14:01:00 2020

@author: 91736
"""
# imports
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing import image


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

train_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory(train_dir ,
                                                   target_size = (150 , 150),
                                                   batch_size = 25,
                                                   class_mode = 'binary')

valid_datagen = ImageDataGenerator(rescale = 1./255)

valid_generator = valid_datagen.flow_from_directory(valid_dir,
                                                    target_size = (150 , 150),
                                                    batch_size = 25,
                                                    class_mode = 'binary')


# defining model 
model = tf.keras.Sequential([
    # 1st Layer
    tf.keras.layers.Conv2D(filters = 32 ,
                           kernel_size = (3,3),
                           input_shape = (150 , 150 , 3) ,
                           activation = 'relu'),
    tf.keras.layers.MaxPool2D(2,2),
    #2nd Layer
    tf.keras.layers.Conv2D(64 , (3,3) , activation = 'relu'),
    tf.keras.layers.MaxPool2D(2,2),
    #3rd Layer
    tf.keras.layers.Conv2D(128 , (3,3) , activation = 'relu'),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    # FCC
    tf.keras.layers.Dense(128 , activation = 'relu'),
    tf.keras.layers.Dense(1 , activation = 'sigmoid')
    ])


model.summary()

# compiler
optimizer_r = RMSprop(lr = 0.001)

model.compile(loss = 'binary_crossentropy' , optimizer = optimizer_r , metrics = ['accuracy'])


# training_model
history = model.fit(train_generator , 
                    validation_data= valid_generator ,
                    epochs = 150,
                    steps_per_epoch = 100,
                    validation_steps= 11)



acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epoch = range(len(acc))

plt.title('Model Accuracy')
plt.plot(epoch , acc , 'b' )
plt.plot(epoch , val_acc  , 'r')
plt.legend(['acc', 'val_acc'])
plt.figure()

plt.title('Model Loss')
plt.plot(epoch , loss , 'b')
plt.plot(epoch , val_loss  , 'r')
plt.legend(['loss' , 'val_loss'])
plt.figure()



test_path1 = r'C:/Users/91736/Documents/PROJECTS/Waste Classifier/Data Files/archive/test_imgs/R_10000.jpg'
test_path2 = r'C:/Users/91736/Documents/PROJECTS/Waste Classifier/Data Files/archive/test_imgs/O_12774.jpg'



def prediction(path):
    img = tf.keras.preprocessing.image.load_img(
        path, target_size=(150, 150))
    
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    
    prediction = model.predict(img_array)
    return prediction[0]

predict_r = prediction(test_path1)
print('R' , predict_r)

predict_o = prediction(test_path2)
print('O' , predict_o)

def prediction_num(path):
    img = image.load_img(path, target_size=(150 , 150))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    
    images = np.vstack([x])
    classes = model.predict(images, batch_size=10)
    if classes[0]>0.5:
        print('R')
    else:
        print('O')

print(prediction_num(test_path1))
print(prediction_num(test_path2))


print (train_generator.class_indices)

labels = '\n'.join(sorted(train_generator.class_indices.keys()))

with open('labels.txt', 'w') as f:
  f.write(labels)
  
  
model.save('Waste_Classifier.h5' , overwrite = True , include_optimizer = True)

model.save_weights('Weights' , overwrite = True , save_format= 'h5')  