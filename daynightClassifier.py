# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 20:25:10 2020

@author: Diksha
"""


#importing Libraries
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.image as mimg
import numpy as np
import glob            

#function to load dataset

def load_dataset(image_dir):
    
    # Populate this empty image list
    im_list = []
    image_types = ["day", "night"]
    
    
    # Iterate through each color folder
    for im_type in image_types:
        
        # Iterate through each image file in each image_type folder
        # glob reads in any image with the extension "image_dir/im_type/*"
        for file in glob.glob(os.path.join(image_dir, im_type, "*")):
            
            # Read in the image
            im = mimg.imread(file)
            
            # Check if the image exists/if it's been correctly read-in
            if not im is None:
                # Append the image, and it's type (day,night) to the image list
                im_list.append((im, im_type))
    
    return im_list

#getting the path to  current directory
image_dir_train = os.getcwd()
training = load_dataset(image_dir_train)


#resizing all images to a particular dimension
def standarize_img(image):
    
    stimg = cv2.resize(image,(1100,600))
    
    return stimg

#label encoding
def encode(label):
    num_val = 0
    if(label == 'day'):
        num_val = 1
    return(num_val)    
    
def standarize(image_list):
    stdlist =[]
    for i in image_list:
        img = i[0]
        label = i[1]
        
        
        stdlist.append([standarize_img(img),encode(label)])
        
    return(stdlist)

stdl = standarize(training)

#sample pre-processed image
plt.imshow(stdl[1][0])



#feature extraction(value changes drastically between day and night images)

img = stdl[2][0]
img1 = np.copy(img)
img2 = cv2.cvtColor(img1,cv2.COLOR_BGR2HSV)

#creating hsv channels
h = img2[:,:,0]
s = img2[:,:,1]
v = img2[:,:,2]

plt.imshow(v,cmap = 'gray')

#average brightness

def avg_bright(image):
    
    b = np.sum(image[:,:,2])
    area = 600*1100.0
    bright = b/area
    return bright 
    
#print(avg_bright(stdl[160][0]))

def classify(image):
    threshold = 100
    predicted = 0
    if(avg_bright(image) > threshold):
        predicted = 1
    return(predicted)

# applying on testing dataset and predicting the accuracy

import random

img_dir_test = os.getcwd()
test = load_dataset(img_dir_test)

std2 = standarize(test)
random.shuffle(std2)

def getmisclassified_image(lst):
    misclassified = []
    for i in lst:
        im = i[0]
        true = i[1]
        predict = classify(im)
        if(true != predict):
            misclassified.append([im,predict,true])
            
            
    return(misclassified)
    
misclass =  getmisclassified_image(std2)
total = len(std2)
mis = len(misclass)
print("No of misclassified images:", mis)
print("total testing samples", total)

accuracy = ((total -mis)/total) * 100


print(accuracy)