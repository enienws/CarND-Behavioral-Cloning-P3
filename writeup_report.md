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
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"
[image8]: ./images/cropped.jpg "Cropped Image"
[image9]: ./images/mse.png "MSE Formula"
[image10]: ./images/model1.png "Training Model 1"
[image11]: ./images/model2.png "Training Model 2"
[image12]: ./images/model3.png "Training Model 3"
[image13]: ./images/model4.png "Training Model 4"
[image14]: ./images/model5.png "Training Model 5"
[image15]: ./images/model6.png "Training Model 6"

[image16]: ./images/center_driving1.jpg "Center Driving 1"
[image17]: ./images/center_driving2.jpg "Center Driving 2"

[image18]: ./images/1.png "Flip 1"
[image19]: ./images/2.png "Flip 2"
[image20]: ./images/3.png "Flip 3"
[image21]: ./images/4.png "Flip 4"
[image22]: ./images/5.png "Flip 5"
[image23]: ./images/6.png "Flip 6"

[image24]: ./images/architecture.png "Architecture"




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

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```
I seperated both the implementation of reading from a CSV file and the implementation of generator from model.py, related code can be found in readdataset.py. 


#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I have chosen to implement network architecture that is defined in [paper](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) named **End to End Learning for Self-Driving Cars** . 

This model arcihtecture consists of 5 convolutional layers and 3 fully connected layers. Additionally 1 normalization layer and 1 cropping layer are also implemented in model architecture in order to take advantage of GPU in these processes. 

The input image is normalized by using a lambda layer (line 13 in model.py). Normalization is performed by using the following formula: x/255.0 - 0.5. This way every pixel value is in the range [0-1] and data mean is at 0.0. 

The input image size of the model is 320x160. Cropping layer (line 16 in model.py) crops 70 pixels from top and 25 pixels from down. Cropped region looks like to following image: 

![alt text][image8]

The model consists of 3 convolutional layers with 5x5 kernel size and (2,2) strides. First layer consists of 24 filters, second layer consists of 36 filters, third layer consists of 48 layers (line 19 - 28 in model.py). 

Following two convolutional layers consists of 3x3 kernels without strides. There are 64 filters in both layers (line 31 - 36). 

In convolutional layers ReLU is used as nonlinearity source. Every convolutional layer has a ReLU nonlinearity (lines 19, 23, 27, 31, 35 in model.py)

#### 2. Attempts to reduce overfitting in the model

In the first training attempt of project, the model simply overfits data. This is cleanly due to lack of enough training data for learner to generalize the problem. So I have added more training data in order to reduce overfitting. The first attempt contains 6180 samples. Overfit is avoided when number of training samples is raised to nearly 140k samples. 

The model contains dropout layers at the end of convolutional layers in order to reduce overfitting (lines 20, 24, 28, 32). A probability of 0.3 is used to turn off nodes in the network. 

Additionally in order to let the model to generalize more I have also used data from second track. 

The model was trained and validated on different data sets to ensure that the model was not overfitting (line 61 in model.py). Different data sets are formed by splitting the data as train and validation datasets. The percentage is .8 for train and .2 for validation. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track. A video is captured for a lap on the track. 

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (line 25 on model.py).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, clock-wise and counter-clock-wise running on track

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I begin the project by reading the [paper](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) and tried to understand design principles used in architecture. 

The only way for the car to understand the curvature of the road is to look the road borders. So it is clearly reasonable to use convolutional kernels in order to extract the curvature from the road. 

Network needs to relate the road curvature to steering angle. The problem is a regression problem since car needs continues floating point values for steering angle in order to make smooth turns. Loss function is a mean squared error which is simply evaluated by the following formula:

![alt text][image9]

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. Here is an example output graph:

![alt text][image11]

The intersting thing is that when I tested this model in simulator it drives very well for a lap in track. However driving well for a lap is not meaning that model works well beacuse we do not know whether it can handle all driving situations on the road. Overfitting is a signature for a model that cannot generalize the model so that there will be situations where the model cannot handle some driving situations. 

To combat overfitting I continued to add more training data. I used center lane driving strategy. Additionally I turn the track in clock-wise direction in order to generalize the model well. I trained the model with 6180, 41157, and 90051 samples however overfitting problem is continued.

Then I started to augment data by flipping the frames and multiplying steering angles with -1. This time I used 180102 samples and it seems overfitting problem is avoided. Following graph is related to first trained model that avoids overfitting: 

![alt text][image13]

After 4th model training I have used transfer learning in order to keep previous learnt experience. This way I was able to decrease number of epochs to 3 (previously I used 10 and 25) hence I decreased the training time.

The final step was to run the simulator to see how well the car was driving around track one. For nearly all models vehicle drove well without leaving the track. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 9-54) consisted of a convolution neural network with the following layers and layer sizes:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 225x160x3 RGB image   							| 
| Convolution 5x5     	| 2x2 stride, valid padding, outputs 110x78x24 	|
| RELU					|												|
| Convolution 5x5     	| 2x2 stride, valid padding, outputs 53x37x36 	|
| RELU					|												|
| Convolution 5x5     	| 2x2 stride, valid padding, outputs 24x24x48 	|
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 22x22x64 	|
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 20x20x64 	|
| RELU					|												|
| Fully connected 1		| Input: 25600, Output 100        									|
| Fully connected 2		| Input: 100, Output 50        									|
| Fully connected 3		| Input: 50, Output 10        									|
| Output: Steering Control		| Input: 10, Output 1        									|

Here is a visualization of the architecture [(source)](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)

![alt text][image24]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I started with center lane driving. Here are two samples from this kind of driving:

![alt text][image16]
![alt text][image17]

For all of my training attemps I used left and right cameras too. I simply modified the steering angle by increasing or decreasing by .2. 

I always used the mouse in order to control the car. I believed that generating smooth steering angles is more appropriate since this is a regression problem. I realized that using keyboard generates bad noisy steering angles. So for consecutive training images (ie. n th and n+1 th images) steering angle may differ too much. 

Then I repeated this process on track two in order to get more data points. I obtained the following graph for the training: 

![alt text][image15]

It can be seen that this model's accuracy is not better than previous model however I believe that this model generalizes the problem more than the previous model does. 

To augment the data sat, I also flipped images and angles. As I have mentioned in Section 1 overfitting problem end by doubling the number of samples from 900051 to 180102. Below images are examples for flipping for right center and left cameras

![alt text][image18]
![alt text][image19]
--
![alt text][image20]
![alt text][image21]
--
![alt text][image22]
![alt text][image23]


After the collection process, I had 180102 number of data points.

I always randomly shuffled the dataset in the python generator (line 35 in model.py)

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. 
