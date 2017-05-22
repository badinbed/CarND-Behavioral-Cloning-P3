import os
import csv
import cv2
import math
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
	
			
class SampleGenerator:
	def __init__(self, samples, batch_size=32, use_side_cams = False, angle_correction = 0.2, flip_horizontal=False, shift_vertically = 0):
		n_cam_imgs = 3
		self.samples = samples
		self.use_side_cams = use_side_cams
		self.angle_correction = angle_correction
		self.flip_horizontal = flip_horizontal
		self.shift_vertically = shift_vertically
		
		self.num_samples = len(samples)
		self.batch_size = batch_size
		if use_side_cams:
			self.num_samples = self.num_samples * n_cam_imgs
			self.batch_size = (self.batch_size // n_cam_imgs) * n_cam_imgs
		
		self.batch_size = min(self.batch_size, self.num_samples)
		self.num_steps = math.ceil(self.num_samples / self.batch_size)
	
	def flip(self, x):
		x = x.swapaxes(1, 0)
		x = x[::-1, ...]
		x = x.swapaxes(0, 1)
		return x
		
	def shift(self, x, rows):
		return np.roll(x, rows, 0)
			
	def augment(self, img, angle):
		img = np.asarray(img)
		if self.flip_horizontal and np.random.random() < 0.5:
			img = self.flip(img)
			angle = -angle
			
		if self.shift_vertically:
			rows = np.random.uniform(-self.shift_vertically, self.shift_vertically)
			img = self.shift(img, rows)
			
		return img, angle
		
			
	def generate(self):
		while 1: # Loop forever so the generator never terminates
			shuffle(self.samples)
			for offset in range(0, len(self.samples), self.batch_size):
				batch_samples = self.samples[offset:offset + self.batch_size]

				images = []
				angles = []
				for batch_sample in batch_samples:
					center_angle = float(batch_sample[3])
					center_image = cv2.imread(os.path.relpath(batch_sample[0]))
					aug_image, aug_angle = self.augment(center_image, center_angle)
					images.append(aug_image)
					angles.append(aug_angle)
					
					if self.use_side_cams:
						left_image, left_angle = self.augment(cv2.imread(os.path.relpath(batch_sample[1])), center_angle + self.angle_correction)
						images.append(left_image)
						angles.append(left_angle)
						right_image, right_angle = self.augment(cv2.imread(os.path.relpath(batch_sample[2])), center_angle - self.angle_correction)
						images.append(right_image)
						angles.append(right_angle)
						

				# trim image to only see section with road
				X_train = np.array(images)
				y_train = np.array(angles)
				
				yield X_train, y_train
				
				
	def test_augmentation(self, img_file):
		result = []
		print('Testing augmentation on file:', img_file)
		img = np.asarray(cv2.imread(img_file))
		result.append(img)
		result.append(self.flip(img))

		for rows in range(-self.shift_vertically, self.shift_vertically + 1):
			result.append(self.shift(img, rows))
			
		print('augmentation result: ', len(result), 'images')
		return np.array(result)
		
			
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
	parser= argparse.ArgumentParser()
	parser.add_argument('--sample_folder')
	parser.add_argument('--use_side_cams', default=True, action='store_true')
	parser.add_argument('--angle_correction', default=0.2, type=float)
	parser.add_argument('--flip', default=False, action='store_true')
	parser.add_argument('--shift', default=0, type=int)
	parser.add_argument('--epochs', default=5, type=int)
	parser.add_argument('--batch_size', default=30, type=int)
	parser.add_argument('--sample_size', default=None, type=int)
	parser.add_argument('--validation_size', default=0.2, type=float)
	parser.add_argument('--load_model', default=None)
	parser.add_argument('--save_model', default='model.h5')
	parser.add_argument('cmd', choices=['train', 'eval', 'test_augment'])

	args =  parser.parse_args()
    
	with keras.backend.get_session():
		# load samples for training or evaluation
		samples = shuffle(load_samples(args.sample_folder), n_samples=args.sample_size)
		if len(samples) == 0:
			print('No samples found in:', args.sample_folder)
			return 0
			
		print("Samples loaded from {}: {}".format(args.sample_folder, len(samples)))

		# load model or create a new one if none specified
		if not args.load_model:
			model = create_nvidia_model()
			print('New model based on nvidia compiled using mse loss and adam optimizer')
		else:
			model = keras.models.load_model(args.load_model)		
			print('Loaded model:', args.load_model)
			model.summary()

			
		# if we trained the model we want to save the result
		if args.cmd == 'train':	
			train_samples, validation_samples = train_test_split(samples, test_size=args.validation_size)
			train_gen = SampleGenerator(train_samples, 
									batch_size=args.batch_size, 
									use_side_cams = args.use_side_cams, 
									angle_correction = args.angle_correction,
									flip_horizontal = args.flip,
									shift_vertically = args.shift)
		
			valid_gen = SampleGenerator(validation_samples, 
									batch_size=args.batch_size, 
									use_side_cams = args.use_side_cams, 
									angle_correction=args.angle_correction)	
											
		
			print("Training model: epochs={}, batch_size={} steps_per_epoch={}, validation_steps={}".format(args.epochs, train_gen.batch_size, train_gen.num_steps, valid_gen.num_steps))			
			history_object = model.fit_generator(train_gen.generate(), 
												steps_per_epoch = train_gen.num_steps, 
												validation_data = valid_gen.generate(), 
												validation_steps = valid_gen.num_steps, 
												epochs = args.epochs, 
												verbose=1)
			
			print('Saving model to:', args.save_model)
			model.save(args.save_model)
		elif args.cmd == 'eval':
			eval_gen = SampleGenerator(samples, 
									batch_size = args.batch_size, 
									use_side_cams = args.use_side_cams, 
									angle_correction=args.angle_correction)
			print("Evaluating model: batch_size={} steps={}".format(eval_gen.batch_size, eval_gen.num_steps))			
			
			loss = model.evaluate_generator(eval_gen.generate(), steps = eval_gen.num_steps)
			print('Average loss:', loss / len(samples)) # sucks that evaluate_generator doesnt have verbosity
		elif args.cmd == 'test_augment':
			gen = SampleGenerator(samples,
						flip_horizontal = args.flip,
						shift_vertically = args.shift)
			images = gen.test_augmentation(samples[0][0])
			
			for i in range(len(images)):
				print("test_augment-" + str(i) + ".jpg")
				cv2.imwrite("test_augment-" + str(i) + ".jpg", images[i])
	
if __name__ == "__main__":
    main()