# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 09:04:33 2020

@author: 91736
"""
# dependencies
import tensorflow as tf
from webptools import dwebp
import numpy as np
import random , os , glob
import matplotlib.pyplot as plt
from tensorflow .keras.models import Sequential
from tensorflow.keras.preprocessing import image
from tensorflow.keras.callbacks import ModelCheckpoint , EarlyStopping 
from tensorflow.keras.layers import Conv2D , Flatten , MaxPooling2D , Dense , Dropout , SpatialDropout2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator , img_to_array , array_to_img

#paths
base_path = r'C:/Users/91736/Documents/PROJECTS/Waste Classifier/Data Files/dataset-resized/dataset-resized' # file path
img_list = glob.glob(os.path.join(base_path , '*/*.jpg'))
print(len((img_list)))

# Image Augmentation
train_datagen = ImageDataGenerator(rescale = 1./255 ,
                                   rotation_range= 40,
                                   width_shift_range= 0.2,
                                   height_shift_range= 0.2,
                                   zoom_range = 0.2,
                                   shear_range = 0.2,
                                   horizontal_flip= True,
                                   vertical_flip= True,
                                   fill_mode= 'nearest'
                                   )
valid_datagen = ImageDataGenerator(rescale = 1./255 , validation_split = 0.1)

train_gen = train_datagen.flow_from_directory(base_path ,
                                              batch_size = 32,
                                              target_size = (300 , 300),
                                              subset = 'training',
                                              class_mode = 'categorical')

valid_gen = valid_datagen.flow_from_directory(base_path,
                                              target_size = (300 ,300),
                                              batch_size = 32 ,
                                              class_mode = 'categorical' ,
                                              subset = 'validation')
# printing out labels genrated
labels = (train_gen.class_indices)
labels = dict((v , k) for k , v in labels.items())
print(labels)

# printing shape of test and validation data
for img_batch , lb_batch in train_gen:
    break
print(img_batch.shape , lb_batch.shape)

# writing labels to a file
Labels = '\n'.join(sorted(train_gen.class_indices))
print(Labels)
with open('labels.txt' , 'w') as f:
    f.write(Labels)

tf.keras.backend.clear_session()
tf.random.set_seed(51)
np.random.seed(51)
    
# Model
model = Sequential()

#1st layer
model.add(Conv2D(32 , (3,3) , input_shape = (300 , 300 ,3) , activation = 'relu'))
model.add(MaxPooling2D(2,2))

#2nd layer
model.add(Conv2D(64 , (3,3) , activation = 'relu'))
model.add(MaxPooling2D(2,2))

#3rd layer
model.add(Conv2D(128 , (3,3) , activation = 'relu'))
model.add(MaxPooling2D(2,2))


# Classification layers
model.add(Flatten())
model.add(Dropout(0.2))
#model.add(Dense(128 , activation = 'relu'))
#model.add(Dropout(0.2))
model.add(Dense(64 , activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(32 , activation = 'relu'))
model.add(Dropout(0.2))
#5th last layer
model.add(Dense(6 , activation = 'softmax'))


# saving best model
path = 'classifier_model.h5'
checkpoint = ModelCheckpoint(path , monitor = 'val_accuracy' , save_best_only=True , mode= 'max' , verbose = 1 )
callback_ls = [checkpoint]

#model.summary()

# compiling model 
model.compile(loss = 'categorical_crossentropy' , 
              optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001 ),
              metrics = ['accuracy'])




# training model
history = model.fit(train_gen ,
                    validation_data = valid_gen ,
                    epochs = 50 ,
                    steps_per_epoch = 2276//32  ,
                    validation_steps = 251 // 32,
                    workers = 4 ,
                    callbacks = callback_ls)

#es=EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50) # early stopping

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

# predicting results
def predict_class(path):
    img = image.load_img(path , target_size = (300 , 300))
    img = image.img_to_array(img, dtype=np.uint8)
    img = np.array(img)/255.0
    
    plt.title('Image')
    plt.axis('off')
    plt.imshow(img.squeeze())
    
    predict = model.predict(img[np.newaxis , ...])
    predicted_class = labels[np.argmax(predict[0] , axis = -1)]
    
    print('Prediction Value: ' , np.max(predict[0] , axis = -1))
    print("Classified:",predicted_class)
  
  
# test path change as per need
'''
predict_class(r'C:/Users/91736/Documents/PROJECTS/Waste Classifier/Data Files/test_imgs/plastic_bottle.jpg')

# 73% accuracy reached - 50 epochs 

'''
