import os
import csv
import cv2
import sklearn
import numpy as np
from random import shuffle

samples = []
# Loads all the samples 
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

# Removes the first element from samples because it contains header information
# 0:center, 1:left, 2:right, 3:steering, 4:throttle, 5:brake, 6:speed
header = samples.pop(0)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# Takes in the source path from driving_log.csv, assumes that the data is
# stored in './data/IMG/'
def process_image(source_path):
    filename = source_path.split('/')[-1] 
    path = './data/IMG/' + filename
    image = cv2.imread(path)
    return image

# A generator so that data can be loaded and preprocessed on the fly
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []

            for batch_sample in batch_samples:
                # In order to find an optimal correction factor trigonometry
                # and physics could be used, but experimentation works too
                correction = 0.2
                steering_center = float(batch_sample[3]) 
                # Adjusted steering measurements for left and right
                steering_left = steering_center + correction
                steering_right = steering_center - correction
                # Angles stored in a list for easy access with iteration
                batch_sample_angles = [steering_center, steering_left, steering_right]

                # Loop to reach center, left and right
                for i in range(3):
                    source_path = batch_sample[i]
                    image = process_image(source_path)
                    images.append(image)
                    angles.append(batch_sample_angles[i])
            
            images_augmented = []
            angles_augmented = []
            
            # Image flipping section, adds more data, helps with generalization
            for image, angle in zip(images, angles):
                # Regular unmodified image and angle
                images_augmented.append(image)
                angles_augmented.append(angle)

                # Flipped image and angle vertically
                images_augmented.append(cv2.flip(image,1))
                angles_augmented.append(angle * -1.0)

            X_train = np.array(images_augmented)
            y_train = np.array(angles_augmented)
            yield sklearn.utils.shuffle(X_train, y_train)

# Compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, SpatialDropout2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D


# Model starts here
# The Network Architecture is inspired by NVIDIA's
# https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/ 

model = Sequential()

# Normalize and mean center image
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))

# Crop unwanted pixels: the hood of the car and unwanted background noise
model.add(Cropping2D(cropping=((70,25),(0,0))))

model.add(Conv2D(24,(5,5), strides=(2,2), activation='relu'))
model.add(SpatialDropout2D(0.25))

model.add(Conv2D(36,(5,5), strides=(2,2), activation='relu'))
model.add(SpatialDropout2D(0.25))

model.add(Conv2D(48,(5,5), strides=(2,2), activation='relu'))
model.add(SpatialDropout2D(0.25))

model.add(Conv2D(64,(3,3), activation='relu'))
model.add(SpatialDropout2D(0.25))

model.add(Conv2D(64,(3,3), activation='relu'))
model.add(SpatialDropout2D(0.25))

model.add(Flatten())

model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, steps_per_epoch=len(train_samples), \
                    validation_data=validation_generator, \
                    validation_steps=len(validation_samples), epochs=5)

model.save('model.h5')
exit()

