import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, ZeroPadding2D,BatchNormalization, Activation, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.utils import shuffle
import cv2
import imutils
import numpy as np
import matplotlib.pyplot as plt
import time
from os import listdir

def crop_brain_contour(image, plot=False):
    
	# Convert the image to grayscale, and blur it slightly
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (5, 5), 0)

	# Threshold the image, then perform a series of erosions +
	# dilations to remove any small regions of noise
	thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
	thresh = cv2.erode(thresh, None, iterations=2)
	thresh = cv2.dilate(thresh, None, iterations=2)

	# Find contours in thresholded image, then grab the largest one
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	c = max(cnts, key=cv2.contourArea)
    

	# Find the extreme points
	extLeft = tuple(c[c[:, :, 0].argmin()][0])
	extRight = tuple(c[c[:, :, 0].argmax()][0])
	extTop = tuple(c[c[:, :, 1].argmin()][0])
	extBot = tuple(c[c[:, :, 1].argmax()][0])
    
	# crop new image out of the original image using the four extreme points (left, right, top, bottom)
	new_image = image[extTop[1]:extBot[1], extLeft[0]:extRight[0]]            

	if plot:
		plt.figure()

		plt.subplot(1, 2, 1)
		plt.imshow(image)
		plt.tick_params(axis='both',which='both',top=False,bottom=False,left=False,right=False,labelbottom=False,labeltop=False,labelleft=False,labelright=False)
        
		plt.title('Original Image')
            
		plt.subplot(1, 2, 2)
		plt.imshow(new_image)

		plt.tick_params(axis='both', which='both',top=False,bottom=False,left=False,right=False,labelbottom=False,labeltop=False,labelleft=False,labelright=False)

		plt.title('Cropped Image')
        
		plt.show()
    
	return new_image

def load_data(dir_list, image_size):
	"""
	Read images, resize and normalize them. 
	Arguments:
		dir_list: list of strings representing file directories.
	Returns:
		X: A numpy array with shape = (#_examples, image_width, image_height, #_channels)
		y: A numpy array with shape = (#_examples, 1)
	"""

	# load all images in a directory
	X = []
	y = []
	image_width, image_height = image_size
    
	for directory in dir_list:
		for filename in listdir(directory):
			# load the image
			image = cv2.imread(directory + '/' + filename)
			# crop the brain and ignore the unnecessary rest part of the image
			image = crop_brain_contour(image, plot=False)
			# resize image
			image = cv2.resize(image, dsize=(image_width,image_height), interpolation=cv2.INTER_CUBIC)
			# normalize values
			image = image / 255.
			# convert image to numpy array and append it to X
			X.append(image)
			# append a value of 1 to the target array if the image
			# is in the folder named 'yes', otherwise append 0.
			if directory[-3:] == 'yes':
				y.append([1])
			else:
				y.append([0])
                
	X = np.array(X)
	y = np.array(y)
    
	# Shuffle the data
	X, y = shuffle(X, y)
    
	print(f'Number of examples is: {len(X)}')
	print(f'X shape is: {X.shape}')
	print(f'y shape is: {y.shape}')
    
	return X, y

def split_data(X, y, test_size=0.2):
       
	"""
	Splits data into training, development and test sets.
	Arguments:
		X: A numpy array with shape = (#_examples, image_width, image_height, #_channels)
		y: A numpy array with shape = (#_examples, 1)
	Returns:
		X_train: A numpy array with shape = (#_train_examples, image_width, image_height, #_channels)
		y_train: A numpy array with shape = (#_train_examples, 1)
		X_val: A numpy array with shape = (#_val_examples, image_width, image_height, #_channels)
		y_val: A numpy array with shape = (#_val_examples, 1)
		X_test: A numpy array with shape = (#_test_examples, image_width, image_height, #_channels)
		y_test: A numpy array with shape = (#_test_examples, 1)
	"""
    
	X_train, X_test_val, y_train, y_test_val = train_test_split(X, y, test_size=test_size)
	X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val, test_size=0.5)
    
	return X_train, y_train, X_val, y_val, X_test, y_test

def compute_f1_score(y_true, prob):
	# convert the vector of probabilities to a target vector
	y_pred = np.where(prob > 0.5, 1, 0)
    
	score = f1_score(y_true, y_pred)
    
	return score

def data_percentage(y):
    
	m=len(y)
	n_positive = np.sum(y)
	n_negative = m - n_positive
    
	pos_prec = (n_positive* 100.0)/ m
	neg_prec = (n_negative* 100.0)/ m
    
	print(f"Number of examples: {m}")
	print(f"Percentage of positive examples: {pos_prec}%, number of pos examples: {n_positive}") 
	print(f"Percentage of negative examples: {neg_prec}%, number of neg examples: {n_negative}")

best_model = load_model(filepath='models/cnn-parameters-improvement-24-0.86.model')
best_model.metrics_names

augmented_path = 'augmented_data/'
augmented_yes = augmented_path + 'yes' 
augmented_no = augmented_path + 'no'

IMG_WIDTH, IMG_HEIGHT = (240, 240)
X, y = load_data([augmented_yes, augmented_no], (IMG_WIDTH, IMG_HEIGHT))
X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y, test_size=0.3)

loss, acc = best_model.evaluate(x=X_test, y=y_test)
print (f"Test Loss = {loss}")
print (f"Test Accuracy = {acc}")

y_test_prob = best_model.predict(X_test)

f1score = compute_f1_score(y_test, y_test_prob)
print(f"F1 score: {f1score}")

y_val_prob = best_model.predict(X_val)
f1score_val = compute_f1_score(y_val, y_val_prob)
print(f"F1 score: {f1score_val}")

data_percentage(y)
print("Training Data:")
data_percentage(y_train)
print("Validation Data:")
data_percentage(y_val)
print("Testing Data:")
data_percentage(y_test)

