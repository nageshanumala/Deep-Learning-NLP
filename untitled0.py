#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 19:53:55 2019

@author: nagesh
"""

# Last amended: 25th Nov, 2018
# Ref:  https://github.com/keras-team/keras/issues/4301
#       https://keras.io/applications/#vgg16
#       https://keras.io/applications/
#
# Objectives:
#	     i)  Experimenting with Very Deep ConvNets: VGG16
#        ii) Peeping into layers and plotting extracted-features
#
#  BEFORE THIS EXPT DO THIS:
#   sudo apt-get install imagemagick   DONE
#  On Windows 10, works without any extra installation

#  Make tensorflow as backend
#  =========================

#    cp /home/ashok/.keras/keras_tensorflow.json  /home/ashok/.keras/keras.json
#    cat /home/ashok/.keras/keras.json
#    source activate tensorflow
#    ipython
#    OR, on Windows
#    > conda activate tensorflow_env
#    > atom

## Expt 1
# =======
===

# 1.0 Imp
#ort libraries
%reset -f

from keras.applications.vgg16 import VGG16
# With every deep-learning network, keras has a function
#  to automatically process image to required dimensions
from keras.applications.vgg16 import preprocess_input

# 1.1 Keras image preprocessing:
#      https://github.com/keras-team/keras/blob/master/keras/preprocessing/image.py
from keras.preprocessing import image


# 1.2
import numpy as np
import pylab

# For PIL image manipulation
from PIL import Image as im

###################### AA. Image processing ##########################

# 3.1 Where is my image?
#img_path = "C:\\Users\\ashok\\Desktop\\chmod\\Documents\\deeplearning\\data\\cat.jpg"
img_path = '/Users/nagesh/BIGDATA/cat.png'

# 3.2 Read image in Python Image Library (PIL) format
img = image.load_img(img_path)
type(img)                  # PIL.Image.Image

# 3.3 Some examples of image manipulation using PILLOW library
#     Ref: http://pillow.readthedocs.io/en/3.1.x/handbook/tutorial.html
img.size
img.show()
img.save("/home/ashok/abc.png")
img.rotate(45).show()
img.transpose(im.FLIP_LEFT_RIGHT).show()
img.transpose(im.FLIP_TOP_BOTTOM).show()


# 3.4 Transform PIL image to numpy array
x = image.img_to_array(img)
x.shape                              # 320 X 400 X 3
                                     # Last index of 3 is depth

# 4. For processing an image in VGG16, shape of image should be:
#         [samples, height, width, no_of_channels ]
#    So we need to transfrom the img-dimensions. We can use
#    np.newaxis() as follows:
#    https://stackoverflow.com/a/25755697

x[np.newaxis, :, :, :].shape
x = x[np.newaxis, :, :, :]


# 4.1.1 OR do it this way
#       reshape to array rank 4
#  x = x.reshape((1,) + x.shape)


# 4.2 About preprocess_input, pl see
#     https://stackoverflow.com/questions/47555829/preprocess-input-method-in-keras
#     Some models use images with values ranging from 0 to 1.
#     Others from -1 to +1. Others use the "caffe" style,
#     that is not normalized, but is centered.
#     The preprocess_input function is meant to adjust your image
#     to the format the model requires.

x
x[0,80,90,1]                # 136 (pixel intensity)
x = preprocess_input(x, mode = 'tf')    # 'tf' : tensorflow
x[0,80,90,1]                # 0.06666 (normalized pixel intensity)
x[0, :2, :2 , :2]           # Have a look at few data-points
x.shape                     # (1, 320, 400,3) shape remains same

###################### BB. Model Building ##########################
# 5.0 Create VGG16 model
#     Use the same weights as in 'imagenet' experiment
#     include_top: F means do not include the 3 fully-connected layers
#      at the top of the network.
#     Model weights are in folder ~/.keras/models
#     OR on Windows on: C:\Users\ashok\.keras\models\
model = VGG16(
	          weights='imagenet',
	          include_top=False
	          )

# 5.1 Get features output from all filters
#      We make predictions considering all layers, ie till model end
#      We have jumped 'model.compile' and 'model.fit' steps
#      Why? Read below.
"""
Why no model compilation and fitting?
=====================================
	A model is compiled for setting its configuration such
	as type of optimizer to use and loss-function to use.
	Given these, it is 'fitted' or 'trained' on the dataset
	to learn weights.
	But, if weights are already fully learnt, as is the case in
	this example then there is no need to compile and 'fit'
	the model.
	We straightaway move to 'prediction' using our data so to say
	as 'test' data.
"""
features = model.predict(x)

# 5.2 So how many features?
features.shape                   # (1,10,12,512)
                                 # 1: sample ;
								 # (10,12): feature ht and width;
								 # 512: Depth/number of features
								 # See below, model summary

# 4.3 Number '512' matches with the last layer of model (block5_pool):
model.summary()

				 #  1      =    One batch input,
				 # (10,12) =    filter size
				 #  512    =    No of filters


###################### CC. Display a feature ##########################

# 5 Display output of a specific feature.  Try 10, 115, 150, 500
pic=features[0,:,:,150]         # (1,10,12,512) => initial index can only be 0
pylab.imshow(pic)               # Image of 10 X 12 total no of squares/pixels
pylab.gray()                    # Gray image
pylab.show()


######################  ################################################  ##########################
#           *************  Features at some intermediate layer *****************
######################  ################################################  ##########################

## Expt 2
#==========
# 	Objective:
#             Extract and display features from any arbitrary
#             intermediate convolution layer in VGG16
#
# For layer names of VGG16, pl see:
#     https://github.com/keras-team/keras/blob/master/keras/applications/vgg16.py

# 1. Call libraries
%reset -f
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input

# 1.1 For image processing
from keras.preprocessing import image
import matplotlib.pyplot as plt

# 1.2 Help create a model uptil some intermediate layer
from keras.models import Model

#from keras.models import Sequential
# from keras.layers import *


# 2. Create base model
#    include_top = False, abandons the last three FC layers
base_model = VGG16(
	               weights='imagenet',
	               include_top=False
	               )


# 2.1 What are layer names?
base_model.summary()

# 2.2 To see complete model, including FC layers, try following:
#     WARNING: Be warned, there will be fresh download of model
#              weights.
#             Generally FC layers have a very large number of weights
#             Do it only you have sufficient RAM allocated to your VM
# full_model = VGG16(weights='imagenet', include_top=True)
# full_model.summary()


# 3.  Input layer, next. NAME MAY BE DIFFERENT IN EACH CASE.
#     CHECK NAME OF Ist INPUT LAYER in base_model.summary()
#     a = base_model.get_layer('input_1')
a = base_model.layers[0]
# 3.1
a.input                   # <tf.Tensor 'input_1:0' shape=(?, ?, ?, 3) dtype=float32>
                          # Does not give much information except shape
                          #  Note that input 'batch-size' and 'input-shape' are free
                          #  But the tensor does require all four parameters
# 3.2
a.output                  # Both inputs and output shapes have same shape
# 3.3
a_in = a.input            # Get input interface. That will make us correct


# 4. Looking at the first convolution layer: 'block1_conv1'
# 4.1 Retrieve a layer based on its name (unique)
b = base_model.get_layer('block1_conv1')

# 4.2
b.input                   # What is input to this layer like?
                          # <tf.Tensor 'input_4:0' shape=(?, ?, ?, 3) dtype=float32>
# 4.3
b.output                  # We are interested in the output of this layer
                          # <tf.Tensor 'block1_conv1_3/Relu:0' shape=(?, ?, ?, 64) dtype=float32>
# 4.4
b_out = b.output



# 5    Instantiate model uptill required layer
#      Model begins with 'input' of one layer to 'output'
#      of the IInd layer
model = Model(inputs=a_in , outputs= b_out)

# 5.1
model.summary()

# 4.0 Image processing
img_path = '/home/ashok/Images/cat/cat.png'
#img_path = "C:\\Users\\ashok\\Desktop\\chmod\\Documents\\deeplearning\\data\\cat.jpg"

# 4.1
img = image.load_img(img_path)
# 4.2
x = image.img_to_array(img)
# 4.3
x = np.expand_dims(x, axis=0)          # Another way to expand dimensions
x.shape                                # (1, 320, 400, 3)
# 4.4
x = preprocess_input(x, mode = 'tf')
x
# 5 Feed 'x' to input and predict 'output'
#   This is an intermediate (first) layer of vgg16
# 5.1
block1_conv1_features = model.predict(x)
block1_conv1_features.shape           #  (1, 320, 400, 64)


# 5.2 See nine features in various filters
for i in range(12):
	plt.subplot(3,3, 1 + i)
	im = block1_conv1_features [0,:,:,i+20]
	plt.imshow(im, cmap=plt.get_cmap('gray'))
plt.show()
##########################
