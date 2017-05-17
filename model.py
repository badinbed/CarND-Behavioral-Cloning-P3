import os
import csv
import cv2
import argparse
import numpy as np

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def load_samples(folder):
	samples = []
	with open('rec/driving_log.csv') as csvfile:
		reader = csv.reader(csvfile)
		for line in reader:
			samples.append(line)
	return samples

def generator(samples, batch_size=32):
	num_samples = len(samples)
	while 1: # Loop forever so the generator never terminates
		shuffle(samples)
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset:offset+batch_size]

			images = []
			angles = []
			for batch_sample in batch_samples:
				name = 'rec/IMG/'+os.path.split(batch_sample[0])[-1]
				center_image = cv2.imread(name)
				center_angle = float(batch_sample[3])
				images.append(center_image)
				angles.append(center_angle)

			# trim image to only see section with road
			X_train = np.array(images)
			y_train = np.array(angles)
			
			yield shuffle(X_train, y_train)


from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Conv2D, Activation, AveragePooling2D
import keras

def create_nvidia_model():
'''
Generates a model based on nvidias approach using:
    - a normalization layer to shift values from [0, 255] to [-1, 1]
    - a cropping layer to cut of car hood and sky/trees
    - 3 convolutional layers with kernel=5x5 and stride=2x2
    - 2 convolutional layers with kernel=3x3 and stride=1x1
    - a flattening layer
    - 4 fully connected layers with 100, 50, 10, 1 outbound connections
    
    - Activation functions throughout are relu
    - loss function is mean squared error
    - Optimizer is adam
'''
    model = Sequential()

    # attempt to resize model in a simple manner but didn't like the results
    #model.add(AveragePooling2D(input_shape=(160, 320, 3)))
    
    # Preprocess incoming data, centered around zero with small standard deviation 
    model.add(Lambda(lambda x: x/127.5 - 1.0, input_shape=(160, 320, 3)))

    # crop out top 70 rows (sky/trees) and bottom 25 (car hood)
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))

    model.add(Conv2D(24, (5, 5), strides=(2, 2)))
    model.add(Activation('relu'))

    model.add(Conv2D(36, (5, 5), strides=(2, 2)))
    model.add(Activation('relu'))

    model.add(Conv2D(48, (5, 5), strides=(2, 2)))
    model.add(Activation('relu'))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model
    

### print the keys contained in the history object
#print(history_object)

### plot the training and validation loss for each epoch
#plt.plot(history_object.history['loss'])
#plt.plot(history_object.history['val_loss'])
#plt.title('model mean squared error loss')
#plt.ylabel('mean squared error loss')
#plt.xlabel('epoch')
#plt.legend(['training set', 'validation set'], loc='upper right')
#plt.show()

def main():
    # command line arguments
    parser= argparse.ArgumentParser(prog='model', description='')
    parser.add_Argument('--in')
    parser.add_argument('--epochs', default=32)
    
    args =  parser.parse_args()
    
    
    samples = load_samples(sample_folder)
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)
			
			
    # compile and train the model using the generator function
    train_generator = generator(train_samples, batch_size=32)
    validation_generator = generator(validation_samples, batch_size=32)

    
    model = keras.models.load_model('model.h5')
    history_object = model.fit_generator(train_generator, steps_per_epoch= len(train_samples)//32, validation_data=validation_generator, validation_steps=len(validation_samples)//32, epochs=5, verbose=1)

    model.save('model.h5')

if __name__ == "__main__":
    main()