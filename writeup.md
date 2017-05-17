**Behavioral Cloning Project Writeup**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[image2]: ./examples/center-driving.jpg "Center Driving"
[image3]: ./examples/recover1.jpg "Recovery Image"
[image4]: ./examples/recover2.jpg "Recovery Image"
[image5]: ./examples/recover3.jpg "Recovery Image"
[image6]: ./examples/flip1.jpg "Normal Image"
[image7]: ./examples/flip2.jpg "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup.md summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network with filter sizes 3x3 and 5x5.The depths for the model are between 24 and 64 (model.py lines 100-104) 

The model includes RELU layers to introduce nonlinearity (they start on code line 100), and the data is normalized in the model using a Keras lambda layer (code line 95). 

####2. Attempts to reduce overfitting in the model

The model contains a dropout layer in order to reduce overfitting (model.py line 106). 

The model was trained and validated on different data sets to ensure that the model was not overfitting. The data was split into training samples and validation samples in model.py line 20. 

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 112).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving and recovering from the left and right sides of the road. 

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The core strategy for deriving a model architecture mainly consisted of research.

My first step was to use a convolution neural network model similar to the one that NVDIA uses. I thought this model might be appropriate because it is a model that has actually been used before by self driving car engineers.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I add a dropout layer on model.py line 106.

The final step was to run the simulator to see how well the car was driving around track one. There was one spot where the vehicle fell off the track, the second curve in track 1. To improve the driving behavior in these cases, I collected more data for curves.

Also worth mentioning, one approach that I think improved results consisted in me modifying the drive.py file. I exaggerated steering angles by a little bit. I think that this helped because my model was very good at helping the car recover when it is heading out of bounds. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 95-110) consisted of image normalization with a lambda layer, a cropping layer, three convolutional layers with filter sizes of 5x5, two convolutional layers with filter sizes of 3x3, a dropout layer (keep chance 0.5) and 4 fully connected layers.

A rough visualization of the architecture can be found by following the link in the comments of model.py.

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I started by using the dataset that udacity provided and then kept adding more data.  

I first recorded several laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover. These images show what a recovery looks like starting from the vehicle being close to the left to the vehicle being centered again:

![alt text][image3]
![alt text][image4]
![alt text][image5]

To augment the data, I also flipped images and angles thinking that this would help with model generalization. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

After the collection process, I had about 40 thousand number of data points (assumes inclusion of flipped images). I then preprocessed this data later with the cropping functionality that keras provides. 

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5, because after 5 epochs the mean squared error did not change that much. 

I used an adam optimizer so that manually training the learning rate wasn't necessary.

Overall, I'm pleased with my results. Some things I would like to try in the near future include:
* Trying out different models
* Modifying drive.py to be able to use the left and right cameras
* Improve image preprocessing
* Make the car work on track 2

