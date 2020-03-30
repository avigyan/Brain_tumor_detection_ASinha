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

def plot_sample_images(X, y, n=50):
	"""
	Plots n sample images for both values of y (labels).
	Arguments:
		X: A numpy array with shape = (#_examples, image_width, image_height, #_channels)
		y: A numpy array with shape = (#_examples, 1)
	"""
    
	for label in [0,1]:
		# grab the first n images with the corresponding y values equal to label
		images = X[np.argwhere(y == label)]
		n_images = images[:n]
        
		columns_n = 10
		rows_n = int(n/ columns_n)

		plt.figure(figsize=(20, 10))
        
		i = 1 # current plot        
		for image in n_images:
			plt.subplot(rows_n, columns_n, i)
			plt.imshow(image[0])
            
			# remove ticks
			plt.tick_params(axis='both', which='both',top=False, bottom=False,left=False,right=False,labelbottom=False,labeltop=False, labelleft=False,labelright=False)
            
			i += 1
        
		label_to_str = lambda label: "Yes" if label == 1 else "No"
		plt.suptitle(f"Brain Tumor: {label_to_str(label)}")
		plt.show()

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

# Nicely formatted time string
def hms_string(sec_elapsed):
	h = int(sec_elapsed / (60 * 60))
	m = int((sec_elapsed % (60 * 60)) / 60)
	s = sec_elapsed % 60
	return f"{h}:{m}:{round(s,1)}"

def compute_f1_score(y_true, prob):
	# convert the vector of probabilities to a target vector
	y_pred = np.where(prob > 0.5, 1, 0)
    
	score = f1_score(y_true, y_pred)
    
	return score

def build_model(input_shape):
	"""
	Arugments:
		input_shape: A tuple representing the shape of the input of the model. shape=(image_width, image_height, #_channels)
	Returns:
		model: A Model object.
	"""
	# Define the input placeholder as a tensor with shape input_shape. 
	X_input = Input(input_shape)
    
	# Zero-Padding: pads the border of X_input with zeroes
	X = ZeroPadding2D((2, 2))(X_input)
    
	# CONV -> BN -> RELU Block applied to X
	X = Conv2D(32, (7, 7), strides = (1, 1), name = 'conv0')(X)
	X = BatchNormalization(axis = 3, name = 'bn0')(X)
	X = Activation('relu')(X)
    
	# MAXPOOL
	X = MaxPooling2D((4, 4), name='max_pool0')(X)
    
	# MAXPOOL
	X = MaxPooling2D((4, 4), name='max_pool1')(X)
    
 	# FLATTEN X 
	X = Flatten()(X)
	# FULLYCONNECTED
	X = Dense(1, activation='sigmoid', name='fc')(X)
    
	# Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
	model = Model(inputs = X_input, outputs = X, name='BrainDetectionModel')
    
	return model

def plot_metrics(history):
    
	train_loss = history['loss']
	val_loss = history['val_loss']
	train_acc = history['acc']
	val_acc = history['val_acc']
    
	# Loss
	plt.figure()
	plt.plot(train_loss, label='Training Loss')
	plt.plot(val_loss, label='Validation Loss')
	plt.title('Loss')
	plt.legend()
	plt.show()
    
	# Accuracy
	plt.figure()
	plt.plot(train_acc, label='Training Accuracy')
	plt.plot(val_acc, label='Validation Accuracy')
	plt.title('Accuracy')
	plt.legend()
	plt.show()

def data_percentage(y):
    
	m=len(y)
	n_positive = np.sum(y)
	n_negative = m - n_positive
    
	pos_prec = (n_positive* 100.0)/ m
	neg_prec = (n_negative* 100.0)/ m
    
	print(f"Number of examples: {m}")
	print(f"Percentage of positive examples: {pos_prec}%, number of pos examples: {n_positive}") 
	print(f"Percentage of negative examples: {neg_prec}%, number of neg examples: {n_negative}")


augmented_path = 'augmented_data/'

# augmented data (yes and no) contains both the original and the new generated examples
augmented_yes = augmented_path + 'yes' 
augmented_no = augmented_path + 'no'

IMG_WIDTH, IMG_HEIGHT = (240, 240)

X, y = load_data([augmented_yes, augmented_no], (IMG_WIDTH, IMG_HEIGHT))

plot_sample_images(X, y)

X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y, test_size=0.3)

print ("number of training examples = " + str(X_train.shape[0]))
print ("number of development examples = " + str(X_val.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(y_train.shape))
print ("X_val (dev) shape: " + str(X_val.shape))
print ("Y_val (dev) shape: " + str(y_val.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(y_test.shape))

IMG_SHAPE = (IMG_WIDTH, IMG_HEIGHT, 3)

model = build_model(IMG_SHAPE)

model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# tensorboard
log_file_name = f'brain_tumor_detection_cnn_{int(time.time())}'
tensorboard = TensorBoard(log_dir=f'logs/{log_file_name}')

# checkpoint
# unique file name that will include the epoch and the validation (development) accuracy
filepath="cnn-parameters-improvement-{epoch:02d}-{val_acc:.2f}"
# save the model with the best validation (development) accuracy till now
checkpoint = ModelCheckpoint("models/{}.model".format(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max'))

start_time = time.time()

model.fit(x=X_train, y=y_train, batch_size=32, epochs=24, validation_data=(X_val, y_val), callbacks=[tensorboard, checkpoint])

end_time = time.time()
execution_time = (end_time - start_time)
print(f"Elapsed time: {hms_string(execution_time)}")

history = model.history.history
for key in history.keys():
	print(key)

plot_metrics(history)



