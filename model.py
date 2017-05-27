import os
import csv
import cv2
import math
import argparse
import pickle
import numpy as np

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def load_samples(folders):
	samples = []
	for folder in folders:
		with open(os.path.join(folder, 'driving_log.csv')) as csvfile:
			reader = csv.reader(csvfile)
			for line in reader:
				for indx in range(3):
					line[indx] = os.path.join(folder, 'IMG', os.path.basename(line[indx]))
				line[3] = float(line[3])
				samples.append(line[:4])
	return samples
	
			
class SampleGenerator:
	def __init__(self, samples, n_iterations = 3, batch_size=256, use_side_cams = False, angle_correction = 0.25, translate=False):
		n_cam_imgs = 3
		self.samples = samples
		self.use_side_cams = use_side_cams
		self.angle_correction = (0, angle_correction, -angle_correction)
		self.num_samples = len(samples)
		self.batch_size = batch_size

		self.num_steps = math.ceil(self.num_samples * n_iterations / self.batch_size)
	
	def aug_brightness(self, img, rmin=0.5, rmax = 1.5):
		hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
		hsv_arr = np.array(hsv, dtype = np.float64)
		
		# apply random factor to value (brightness) between [rmin, rmax]
		hsv_arr[:,:,2] = hsv_arr[:,:,2] * np.random.uniform(rmin, rmax)
		hsv_arr[:,:,2][hsv_arr[:,:,2]>255]  = 255 #clamp
		hsv_arr = np.array(hsv_arr, dtype = np.uint8)
		gbr = cv2.cvtColor(hsv_arr,cv2.COLOR_HSV2BGR)
		return gbr
		
	def aug_translation(elf, img, angle, range_x = 50, range_y = 20, angle_per_pixel = 0.004):
		tx = np.random.uniform(-range_x, range_x)
		ty = np.random.uniform(-range_y, range_y)

		#adjust steering angle according the horizontal shift
		angle += tx * angle_per_pixel
		
		# apply translation matrix
		M = np.float32([[1,0,tx],[0,1,ty]])
		rows, cols = img.shape[:2]
		translated = cv2.warpAffine(img, M , (cols, rows))
		return translated, angle	
	
			
	def augment(self, img, angle):
	
		img, angle = self.aug_translation(img, angle, 50, 20, 0.004)
		img = self.aug_brightness(img, 0.5, 1.5)
		
		if np.random.uniform() < 0.5:
			img = cv2.flip(img, 1)
			angle = -angle
			
		return np.array(img), angle
		
			
	def generate(self):
		epoch = 0
		while 1: # Loop forever so the generator never terminates
			shuffle(self.samples)
			epoch += 1
			discard = 1
			for i in range(self.num_steps):
				images = []
				angles = []
				while len(images) < self.batch_size:

				
					sample = self.samples[np.random.randint(self.num_samples)]
					while abs(sample[3]) < 0.1 and np.random.uniform() > discard / epoch:
						sample = self.samples[np.random.randint(self.num_samples)]
						
					img_idx = np.random.randint(3) if self.use_side_cams else 0
					image = cv2.imread(sample[img_idx])
					angle = sample[3] + self.angle_correction[img_idx]
					
					aug_image, aug_angle = self.augment(image, angle)
					images.append(aug_image)
					angles.append(aug_angle)
				

					# trim image to only see section with road
				X_train = np.asarray(images)
				y_train = np.asarray(angles)
				
				yield X_train, y_train
				
				
	def test_augmentation(self, img_file, angle):
		result = []
		print('Testing augmentation on file:', img_file)
		img = cv2.imread(img_file)
		result.append((np.array(img), angle))
		result.append((np.array(cv2.flip(img, 1)), -angle))

		for rows in range(12):
			result.append((np.array(self.aug_brightness(img)), angle))
		for rows in range(12):
			i, a = self.aug_translation(img, angle)
			result.append((np.array(i), a))	
			
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
 

def main():
	# command line arguments
	parser= argparse.ArgumentParser()
	parser.add_argument('-sf', '--sample_folder', nargs='+')
	parser.add_argument('-cam', '--use_side_cams', default=False, action='store_true')
	parser.add_argument('-ac', '--angle_correction', default=0.25, type=float)
	parser.add_argument('-e', '--epochs', default=5, type=int)
	parser.add_argument('-bs', '--batch_size', default=256, type=int)
	parser.add_argument('-ss', '--sample_size', default=None, type=int)
	parser.add_argument('-vs', '--validation_size', default=0.2, type=float)
	parser.add_argument('-lm', '--load_model', default=None)
	parser.add_argument('-sm', '--save_model', default='model.h5')
	parser.add_argument('-a', '--action', default='train', choices=['train', 'eval', 'test_augment', 'plot_history'])

	args =  parser.parse_args()
    
	with keras.backend.get_session():
		# load samples for training or evaluation
		if args.sample_folder:
			samples = load_samples(args.sample_folder)
			if len(samples) > 0:
				samples = shuffle(samples, n_samples=args.sample_size)
				print("Samples loaded from {}: {}".format(args.sample_folder, len(samples)))

		
		# load model or create a new one if none specified
		if not args.load_model:
			model = create_nvidia_model()
			hist = {'loss':[], 'val_loss':[] , 'epochs':[], 'sources':[]}
			print('New model based on nvidia compiled using mse loss and adam optimizer')
			model.summary()
		else:
			model = keras.models.load_model(args.load_model)
			filename, extension = os.path.splitext(args.load_model)
			hist = pickle.load(open(filename + '_hist.p', 'rb'))
			print('Loaded model:', args.load_model)
			model.summary()

			
		# if we trained the model we want to save the result
		if args.action == 'train':	
			train_samples, validation_samples = train_test_split(samples, test_size=args.validation_size)
			train_gen = SampleGenerator(train_samples, 
									batch_size=args.batch_size, 
									use_side_cams = args.use_side_cams, 
									angle_correction = args.angle_correction,)
		
			valid_gen = SampleGenerator(validation_samples, 
									batch_size=args.batch_size, 
									use_side_cams = args.use_side_cams, 
									angle_correction=args.angle_correction)	
											
		
			print("Training model: epochs={}, batch_size={} steps_per_epoch={}, validation_steps={}".format(args.epochs, train_gen.batch_size, train_gen.num_steps, valid_gen.num_steps))			
			hist_object = model.fit_generator(train_gen.generate(), 
												steps_per_epoch = train_gen.num_steps, 
												validation_data = valid_gen.generate(), 
												validation_steps = valid_gen.num_steps, 
												epochs = args.epochs, 
												verbose=1)

			hist['loss'] += hist_object.history['loss']
			hist['val_loss'] += hist_object.history['val_loss']
			hist['epochs'].append(args.epochs)
			hist['sources'].append(args.sample_folder)
			
			print('Saving model to:', args.save_model)
			model.save(args.save_model)
			hist_file, extension = os.path.splitext(args.save_model)
			hist_file += '_hist.p'
			print('Saving history to:', hist_file)
			pickle.dump(hist, open(hist_file, 'wb'))
				
		elif args.action == 'eval':
			eval_gen = SampleGenerator(samples, 
									batch_size = args.batch_size, 
									use_side_cams = args.use_side_cams, 
									angle_correction=args.angle_correction)
			print("Evaluating model: batch_size={} steps={}".format(eval_gen.batch_size, eval_gen.num_steps))			
			
			loss = model.evaluate_generator(eval_gen.generate(), steps = eval_gen.num_steps)
			print('Average loss:', loss / len(samples)) # sucks that evaluate_generator doesnt have verbosity
		elif args.action == 'test_augment':
			gen = SampleGenerator(samples)
			images = gen.test_augmentation(samples[0][0], samples[0][3])
			
			for i in range(len(images)):
				file_name = 'test_augment-{:04}-{:4.3f}.jpg'.format(i, images[i][1])
				print(file_name)
				cv2.imwrite(os.path.join('augmentation', file_name), images[i][0])
		elif args.action == 'plot_history':
			x = list(range(1, len(hist['loss']) + 1))
			plt.plot(x, hist['loss'], '#0072bd')
			plt.plot(x, hist['val_loss'], '#d95319')
			vl = 0.5
			for v in hist['epochs'][:-1]:
				vl += v
				plt.axvline(vl, linestyle='--', lw=0.5, color='#edb120')
				
			plt.xticks(x)
			plt.title('model mean squared error loss')
			plt.ylabel('mean squared error loss')
			plt.xlabel('epoch')
			plt.legend(['training set', 'validation set', 'sessions'], loc='upper right')
			plt.show()
	
if __name__ == "__main__":
    main()