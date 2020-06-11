# Day-Night-Classifier
A machine Learning model to classify between day and night images.

About the Dataset Used
The dataset used for the project is a smaller version of the huge AMOS dataset (Archive of Many Outdoor Scenes).400 total images are separated into training and testing datasets.60% of these images are training images, used it to create a classifier. 40% are test images, used to test the accuracy of the classifier.

Aim: To build a classifier that can accurately label these images as day or night, by applying simple yet effecient computer vision techniques.

These are some variables to keep track of where our image data is stored:
1. image_dir_training: the directory where our training image data is stored
2. image_dir_test: the directory where our test image data is stored
3. IMAGE_LIST: list of training image-label pairs
4. STANDARDIZED_LIST: list of preprocessed image-label pairs
