import csv
import cv2
import numpy as np

lines = []
with open('data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)
		
images = []
measurements = []
for line in lines[1:]:
	source_path = line[0]
	file_name = source_path.split('/')[-1]
	current_path = 'data/IMG/' + file_name
	image = cv2.imread(current_path)
	images.append(image)
	measurements.append(float(line[3]))
	
X_train = np.array(images)
y_train = np.array(measurements)
print('min:', y_train.min(), 'max:', y_train.max())


from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda

model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Flatten())
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, nb_epoch=2, shuffle=True)

model.save('model.h5')
