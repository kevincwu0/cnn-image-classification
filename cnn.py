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

# Adding second convolution layer
# don't need to include input_shape since we're done it
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
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

# Compile - SGD, Loss Function, Performance Metric
# Logarithmic loss - binary cross entropy, more than two outcomes, categorical cross entropy
# Metrics is the accuracy metric
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# part 2 - Fitting the CNN to the images 
# Keras preprocessing images to prevent overfitting, image augmentation, 
# great accuracy on training poor results on test sets
# Need lots of images to find correlations, patterns in pixels
# Find patterns in pixels, 10000 images, 8000 training, not much exactly or use a trick
# Image augmentation will create batches and each batch will create random transformation
# leading to more diverse images and more training
# Image augmentation allows us to enrich our dataset to prevent overfitting

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

classifier.fit_generator(training_set,
                        samples_per_epoch=8000,
                        nb_epoch=25,
                        validation_data=test_set,
                        nb_val_samples=2000)