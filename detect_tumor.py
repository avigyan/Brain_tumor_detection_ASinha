from tensorflow.keras.layers import Conv2D, Input, ZeroPadding2D,BatchNormalization, Activation, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model, load_model
import imutils
import cv2

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

best_model = load_model(filepath='models/cnn-parameters-improvement-24-0.86.model')
best_model.metrics_names
IMG_WIDTH, IMG_HEIGHT = (240, 240)
image_width, image_height = (IMG_WIDTH, IMG_HEIGHT)

image = cv2.imread('no 923.jpg') # This is a sample image for testing
image = crop_brain_contour(image, plot=False)
image = cv2.resize(image, dsize=(image_width,image_height), interpolation=cv2.INTER_CUBIC)
image = image / 255.

tumor_prob = best_model.predict(image.reshape(1,240,240,3))
output = image.copy()

if tumor_prob > 0.60:
	cv2.putText(output, "Brain Tumor Detected", (10, 25),	cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
else:
	cv2.putText(output, "Brain Tumor Not Detected", (10, 25),	cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2)


cv2.imshow("Text", output)
cv2.waitKey(0)

