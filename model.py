import os
import csv
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import cv2
import numpy as np
import sklearn
import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Activation
from keras.layers.convolutional import Conv2D, Cropping2D
import math

samples = []
with open('../../../opt/carnd_p3/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None)
    for line in reader:
        samples.append(line)
    
train_samples, validation_samples = train_test_split(samples, test_size = 0.2)

def generator(samples, batch_size = 32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            measurements = []
            for batch_sample in batch_samples:
                for i in range(3):
                    source_path = batch_sample[i]
                    tokens = source_path.split('/')
                    filename = tokens[-1]
                    local_path = "../../../opt/carnd_p3/data/IMG/" + filename
                    image = cv2.imread(local_path)
                    images.append(image)
                measurement = float(line[3])
                correction = 0.35
                measurements.append(measurement)
                measurements.append(measurement + correction)
                measurements.append(measurement - correction)

            augmented_images = []
            augmented_measurements = []
            for image, measurement in zip(images, measurements):
                augmented_images.append(image)
                augmented_measurements.append(measurement)
                flipped_image = cv2.flip(image, 1)
                flipped_measurement = float(measurement) * -1.0
                augmented_images.append(flipped_image)
                augmented_measurements.append(flipped_measurement)
                
            X_train = np.array(augmented_images)
            y_train = np.array(augmented_measurements)
            yield sklearn.utils.shuffle(X_train, y_train)

batch_size = 32 
    
# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

#ch, row, col = 3, 80, 320  # Trimmed image format 

model = Sequential()
model.add(Lambda(lambda x: (x / 127.5) - 1., input_shape = (160, 320, 3)))
model.add(Cropping2D(cropping = ((70, 25), (0, 0))))
model.add(Conv2D(24, (5, 5), strides = (2,2), activation = 'relu'))
model.add(Conv2D(36, (5, 5), strides = (2,2), activation = 'relu'))
model.add(Conv2D(48, (5, 5), strides = (2,2), activation = 'relu'))
model.add(Conv2D(64, (3, 3), activation = 'relu'))
model.add(Conv2D(64, (3, 3), activation = 'relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('relu'))
model.add(Dense(1))

model.compile(optimizer = 'adam', loss = 'mse')
model.fit_generator(train_generator, steps_per_epoch=math.ceil(len(train_samples)/batch_size*6), validation_data=validation_generator, validation_steps=math.ceil(len(validation_samples)/batch_size*6), epochs=3, verbose=1)
#model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, nb_epoch = 3)

model.save('model.h5')