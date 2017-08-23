# Convolutional Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
# Convolution - input image, applying feature detectors => feature map
# 3D Array because colored images
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
# Feature Map - Take Max -> Pooled Feature Map, reduced size, reduce complexity
# without losing performance, don't lose spatial structure
classifier.add(MaxPooling2D(pool_size = (2, 2)))


# Step 3 - Flattening
# Pooled Feature Maps apply flattening maps to a huge vector 
# for a future ANN that is fully-conntected
# Why don't we lose spatial structure by flattening?
# We don't because the high numbers from convolution feature from the feature detector
# Max Pooling keeps them these high numbers, and flattening keeps these high numbers
# Why didn't we take all the pixels and flatten into a huge vector?
# Only pixels of itself, but not how they're spatially structured around it
# But if we apply convolution and pooling, since feature map corresponds to each feature 
# of an image, specific image unique pixels, we keep the spatial structure of the picture.
classifier.add(Flatten())


# Step 4 - Full Connection
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))