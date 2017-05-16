import os
import csv
import cv2
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

train_samples, validation_samples = train_test_split(samples, test_size=0.2)
			
			
# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Conv2D, Activation, AveragePooling2D
import keras
print(keras.__version__)
model = Sequential()

#model.add(AveragePooling2D(input_shape=(160, 320, 3)))
# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x: x/127.5 - 1.0, input_shape=(160, 320, 3)))
# crop out top 70 rows (sky/trees) and bottom 25 (car hood)
model.add(Cropping2D(cropping=((70, 25), (0, 0))))

#model.add(Conv2D(24, (5, 5), strides=(2, 2)))
#model.add(Activation('relu'))

# model.add(Conv2D(36, (5, 5), strides=(2, 2)))
# model.add(Activation('relu'))

# model.add(Conv2D(48, (5, 5), strides=(2, 2)))
# model.add(Activation('relu'))

# model.add(Conv2D(64, (3, 3)))
# model.add(Activation('relu'))

# model.add(Conv2D(64, (3, 3)))
# model.add(Activation('relu'))

model.add(Flatten())
# model.add(Dense(100))
# model.add(Dense(50))
# model.add(Dense(10))
model.add(Dense(1))
model.summary()

model.compile(loss='mse', optimizer='adam')
model = keras.models.load_model('model.h5')
history_object = model.fit_generator(train_generator, steps_per_epoch= len(train_samples)//32, validation_data=validation_generator, validation_steps=len(validation_samples)//32, epochs=5, verbose=1)

model.save('model.h5')
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
    # my code here

if __name__ == "__main__":
    main()