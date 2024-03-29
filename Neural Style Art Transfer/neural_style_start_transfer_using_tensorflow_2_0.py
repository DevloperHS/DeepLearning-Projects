# -*- coding: utf-8 -*-
"""Neural Style Start Transfer Using Tensorflow 2.0
"""

import tensorflow as tf
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import numpy as np
 
print(tf.__version__)

# Setting Seeds For Produciblity

np.random.seed(7)

# Loading Contents

Just load files at path 


```
Files
/tmp : -> Upload 2 images with name content and style , good to rename before hand
```


"""

content_img = plt.imread('content.jpg')
style_img = plt.imread('style.jpg')

"""# Displaying images"""

# display setup
fig,(ax1 , ax2) = plt.subplots(1,2 , figsize=(10 ,20))

# Content And Style Of image Side By Side
ax1.set_title('Main/ Content Image')
ax1.imshow(content_img)

ax2.set_title('Style Image')
ax2.imshow(style_img)

plt.show()

"""# Loading And Reshaping Images

Helper function 

> 1. load the two images in arrays of number
> 2. reshape them for making them compatible with the model.
> 3. Expand dimension to match of input_shape to model (batch_size, h, w, d)
"""

def load_image(image):
    image = plt.imread(image)
    img = tf.image.convert_image_dtype(image , tf.float32)
    img = tf.image.resize(img , [400 , 400]) 
    img = img[tf.newaxis , :] # Shape -> (batch_size, h, w, d)
    return img

# use fn load_image

'''Content'''
content_img = load_image('content.jpg')

'''Style'''
style_img = load_image('style.jpg')

# verify function load_image by lokking at shape
content_img.shape , style_img.shape

"""# Loading Model  (VGG19)"""

vgg = tf.keras.applications.VGG19(include_top = False , weights = 'imagenet')
vgg.trainable = False

#vgg.summary()

"""##Getting Style Features , One Need To Extract Features From These Layers:
>'block1_conv1'

>'block2_conv1'

>'block3_conv1'

>'block4_conv1'

>'block5_conv1'

##And To Get Content Features One Need:
>'block4_conv2' 
"""

# Viewing Layers by calling out 'layers' attribute over 'VGG19 model' and iterating 
for layer in vgg.layers:
    print(layer.name)

"""### Storing All Variable In Respected Lists Variables Accordingly

Exploring Content And  Total Items In Each Layers List
"""

print('Exploring Content And Total Items In Each Layers List')

content_layers = ['block4_conv2']
print('''\nContent Feature Extracting Layer List :''' , content_layers)

style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']
print('''Style Feature Extracting Layer List :''') 
for layer in style_layers:
    print([layer])
    

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

print('\nNo Of Item In Content Layers List: ' , num_content_layers)
print('No Of Item In Style Layers List: ' , num_content_layers)

"""# Defining Heper Functions

Contents:

>Custom VGG model having Specified layer in layers list

Gram_matrix:
>The gram matrices help you determine how similar are the features across the   feature maps within a single convolutional layer 
 
>matrix_mulptiplication of (gm * gm^-1) where gm = gram matrix

Helps In
>Run forward passes on the images and extract the necessary features along the way.
"""

def mini_model(layer_names , model):
    vgg =  tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    outputs = [vgg.get_layer(name).output for name in layer_names]
    model = tf.keras.Model([vgg.input], outputs)
    return model

def gram_matrix(tensor):
    '''
    takes a tensor in var temp
    squuezes it 
    reshape it 
    apply a * a^-1
    expand_dimesnion / add new dimesions
    
    '''
    temp = tensor
    temp = tf.squeeze(temp)
    reshape = tf.reshape(temp , [temp.shape[2] , temp.shape[0]*temp.shape[1]])
    result = tf.matmul(temp , temp , transpose_b = True )
    gram = tf.expand_dims(result , axis = 0)
    return gram

"""### Creating a custom model using mini_model func by defining a Custome_Style_Model Class"""

class Custom_Style_Model(tf.keras.models.Model):
  def __init__(self, style_layers, content_layers):
    super(Custom_Style_Model, self).__init__()
    self.vgg =  mini_model(style_layers + content_layers, vgg)
    self.style_layers = style_layers
    self.content_layers = content_layers
    self.num_style_layers = len(style_layers)
    self.vgg.trainable = False

  def call(self, inputs):
    # Scale back the pixel values
    inputs = inputs*255.0
    # Preprocess them with respect to VGG19 stats
    preprocessed_input = preprocess_input(inputs)
    # Pass through the mini network
    outputs = self.vgg(preprocessed_input)
    # Segregate the style and content representations
    style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                      outputs[self.num_style_layers:])

    # Calculate the gram matrix for each layer
    style_outputs = [gram_matrix(style_output)
                     for style_output in style_outputs]

    # Assign the content representation and gram matrix in
    # a layer by layer fashion in dicts
    content_dict = {content_name:value
                    for content_name, value
                    in zip(self.content_layers, content_outputs)}

    style_dict = {style_name:value
                  for style_name, value
                  in zip(self.style_layers, style_outputs)}

    return {'content':content_dict, 'style':style_dict}

"""###Use Custom Model On Images To Get Content And Style Features Accordingly (Using Above Made Class)"""

extractor = Custom_Style_Model(style_layers, content_layers)
style_extracted = extractor(style_img)['style']
content_extracted = extractor(content_img)['content']

"""# Setting Up Optimizers"""

opt = tf.optimizers.Adam(learning_rate=0.020)

"""# Defining Overall Weight

Define the overall content and style weights and also the weights for each of the style representations
"""

style_weight = 1e-2  # play
content_weight = 1e3 # play

# setting up random weight for different layers for model intialization
style_weights = {'block1_conv1': 1.,
                'block2_conv1': 0.8,
                'block3_conv1': 0.5,
                'block4_conv1': 0.3,
                'block5_conv1': 0.1}

"""# Defining Loss Fn

![Screen_Shot_2019-06-20_at_1.16.56_PM.png](data:image/png;base64,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)![Screen_Shot_2019-06-20_at_1.16.44_PM.png](data:image/png;base64,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)![Screen_Shot_2019-06-20_at_1.16.36_PM.png](data:image/png;base64,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)

where, α denotes content weight, and β denotes style weight\



& 

 Tc refers to the target image and Cc refers to the content image.



T(s,i) is the gram matrix of the target image calculated at block i and S(s,i) is the gram matrix of the style image calculated at block i. With wi  provide custom weights to the different convolution blocks to attain a detailed representation of the style. Finally, a is a constant that accounts for the values in each layer within the blocks. 


"""

# The loss function to optimize
def total_loss(outputs):
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    style_loss = tf.add_n([style_weights[name]*tf.reduce_mean((style_outputs[name]-style_extracted[name])**2)
                           for name in style_outputs.keys()])
    # Normalize
    style_loss *= style_weight / num_style_layers

    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_extracted[name])**2)
                             for name in content_outputs.keys()])
    # Normalize
    content_loss *= content_weight / num_content_layers
    loss = style_loss + content_loss
    return loss

"""###Defining Function For:

> Calculating the gradients of the loss function  just defined.


>Use these gradients to update the    target image

Note : Old School Way!!!

**Key Components**
>`GradientTape`, to take advantage of automatic differentiation, which can calculate the gradients of a function based on its composition.  

>Also use of  `the tf.function` decorator to speed up the operations.
"""

total_variation_weight=30


@tf.function()
def train_step(image):
  with tf.GradientTape() as tape:
    # Extract the features
    outputs = extractor(image)
    # Calculate the loss
    loss = total_loss(outputs)
    loss += total_variation_weight*tf.image.total_variation(image)

  # Determine the gradients of the loss function w.r.t the image pixels
  grad = tape.gradient(loss, image)
  # Update the pixels
  opt.apply_gradients([(grad, image)])
  # Clip the pixel values that fall outside the range of [0,1]
  image.assign(tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0))

"""###Defining target Image As Content Image"""

target_image = tf.Variable(content_img)

"""# Train  The Network"""

epochs = 10
steps_per_epoch = 550
 
step_count = 0
for n in range(epochs):
    for m in range(steps_per_epoch):
        step_count += 1
        train_step(target_image)
    plt.imshow(np.squeeze(target_image.read_value() , 0))
    plt.title('Training At Each Steps: {}'.format(step_count))
    print('Steps: {}'.format(step_count))

"""##Play with Below Parameters To See Variations In Created Image:


"""

list = ['style_weight',
        'content_weight', 
        'steps_per_epoch',
        'epochs']

print('Play with Below Parameters To See Variations In Created Image:')
for items in list:
    print(items)

"""# **Thanks For Viewing**

"""
