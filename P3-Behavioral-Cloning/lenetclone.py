import csv
import cv2
import numpy as np

#Loading CSV File
lines = []
with open('../data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

#Reading images and steering angles from CSV file and storing them in list
images = []
measurements = []
for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = '../data/IMG' + filename
    image = cv2.imread(current_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)

#Converting the list into numpy arrays
#This constitutes Features and Labels
X_train = np.array(images)
y_train = np.array(measurements)

#Model Architecture starts from here
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Convolution2D

model = Sequential()
#Preprocessing the images
#Normalization and Mean Centre
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))

model.add(Convolution2D(6,5,5,border_mode='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(16,5,5,border_mode='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(120))
model.add(Activation('relu'))

model.add(Dense(84))
model.add(Activation('relu'))

model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=7)

model.save('model.h5')
