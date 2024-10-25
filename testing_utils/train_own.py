# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 14:08:37 2018

@author: vahid
"""

import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from smallervggnet import SmallerVGGNet
import matplotlib.pyplot as plt
from imutils import paths
import numpy as np
import argparse
import random
import pickle
import cv2
import os
from keras import callbacks


from keras import backend as t
t.set_image_data_format('channels_last')
#%%%%

data = []
labels = []
data_dir_list=[('black_206'),('white_206'),('black_cadillac'),('white_cadillac')]
imagePaths="/home/vahid/car-color-recognition/keras-multi-label /dataset2/"
for dataset in data_dir_list:
    img_list=os.listdir(imagePaths+'/'+dataset)
    print('load the dataset'+'{}\n'.format(dataset))

    EPOCHS = 75
    INIT_LR = 1e-3
    BS = 32
    IMAGE_DIMS = (96, 96, 3)

    print("[INFO] loading images...")
    
    # loop over the input images
    for imagePath in img_list:
        image = cv2.imread(imagePaths+'/'+dataset+'/'+imagePath)
        print(image)
        image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
        image = img_to_array(image)
        data.append(image)
        l = label = dataset.split(os.path.sep)[-1].split("_")
        labels.append(l)
            
            
# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
print("[INFO] data matrix: {} images ({:.2f}MB)".format(
	len(imagePaths), data.nbytes / (1024 * 1000.0)))

# binarizer implementation
print("[INFO] class labels:")
mlb = MultiLabelBinarizer()
labels = mlb.fit_transform(labels)

# loop over each of the possible class labels and show them
for (i, label) in enumerate(mlb.classes_):
	print("{}. {}".format(i + 1, label))

# the data for training and the remaining 20% for testing
trainX, testX, trainY, testY = train_test_split(data,
	labels, test_size=0.2, random_state=3)

# construct the image generator 
aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")

# in the network so we can perform multi-label classification
print("[INFO] compiling model...")
model = SmallerVGGNet.build(
	width=IMAGE_DIMS[1], height=IMAGE_DIMS[0],
	depth=IMAGE_DIMS[2], classes=len(mlb.classes_),
	finalAct="sigmoid")

# initialize the optimizer
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])


H = model.fit_generator(
	aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY),
	steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS, verbose=1)

# save the model to disk
print("[INFO] serializing network...")
model.save('/home/vahid/car-color-recognition/keras-multi-label /results/weight' )


# save the multi-label binarizer to disk
print("[INFO] serializing label binarizer...")
f = open('/home/vahid/car-color-recognition/keras-multi-label /results/lb', "wb")
f.write(pickle.dumps(mlb))
f.close()







