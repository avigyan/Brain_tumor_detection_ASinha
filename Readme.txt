AUTHOR: AVIGYAN SINHA

This project determines whether or not Brain Tumor is present in a given Brain MRI image using a CNN model:

1) data_aug.py - significantly increases the diversity of data available for training models using data augmentation

2) train.py - trains a CNN model on the augmented data and saves the model

3) test.py - evaluates the saved trained CNN model on some sample images and shows us some performance metrics

4) detect_tumor.py - determines whether Brain Tumor is present  in an input MRI image and diplays it on the image

USAGE:
python3 detect_tumor.py