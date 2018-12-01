# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image-center]: ./IMG/center_2018_11_30_18_29_51_073.jpg "Center Image"
[image-left]: ./IMG/left_2018_11_30_18_29_51_073.jpg "Left Image"
[image-right]: ./IMG/right_2018_11_30_18_29_51_073.jpg "Right Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results
* recorded-video.mp4 a recorded video for apprlximately 2 laps recorded in autonomus mode.

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model implememnts NVIDIA architecture.
My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 256 (model.py lines 68-90) 

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 107).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road as shown in the following images:

![alt text][image-center]
![alt text][image-left]
![alt text][image-right]


For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to trying to simulate my driving behavior on the model by recording 3 images for each frame. These 3 images are corresponding to the center, right, and left cameras on the car.
For each frame we have the driving wheel angle as the training is a supervised one.

My first step was to use a convolution neural network model similar to the network implementation in NVIDIA architecture. I thought this model might be appropriate because it deals also with a massive amount of images like the NVIDIA architecture did.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To combat overfitting, I implemented the early termination approach so that it trains the model for up to 50 epochs, and the model stops as soon as the validation accuracy does not improve for two consecutive epochs.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, and to improve the driving behavior in these cases, I added more training data for the areas where the car does not perform well in.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 68-90) consisted of a convolution neural network with the following layers and layer sizes  
 * A convolutional layer with filter = 24, and kernel-size = 5.
 * A 2x2 max pooling layer.
 * A convolutional layer with filter = 36, and kernel-size = 5.
 * A convolutional layer with filter = 48, and kernel-size = 5.
 * A 2x2 max pooling layer.
 * A convolutional layer with filter = 64, and kernel-size = 3.
 * A convolutional layer with filter = 64, and kernel-size = 3.
 * A 2x2 max pooling layer.
 * A fully connected layer with 1164 neurons.
 * A RELU layer.
 * A fully connected layer with 100 neurons.
 * A RELU layer.
 * A fully connected layer with 50 neurons.
 * A RELU layer.
 * A fully connected layer with 10 neurons.
 * A RELU layer.
 * A fully connected layer with 1 neurons.

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image-center]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover when it goes of the road.

To augment the data sat, I also flipped images and angles thinking that this would helpful to help generalizing the model. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]


After the collection process, I had 4800x3 of data points.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by the early termination approach when I run it on the network. I used an adam optimizer so that manually training the learning rate wasn't necessary.
