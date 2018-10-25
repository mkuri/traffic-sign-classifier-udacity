# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[histogram]: ./figures/histogram.png "the data histogram"
[samples]: ./figures/explore_imgs.png "sample images"
[averaging]: ./figures/clahe_compare.png "histogram averaging"
[gray]: ./figures/grayscale.jpg "gray"
[loss]: ./figures/reduction_of_loss.png "loss"
[web]: ./figures/web_images.png "web"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/mkuri/traffic-sign-classifier-udacity)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed.

![The histogram of the data][histogram]

Here is the sample of the data set.

![The sample of the dataset][samples]


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I performed a histogram averaging process on the luminance of the images because it turned out that dark images was included when visualizing the train dataset.

Here is an example of traffic sign images before and after histogram averaging. (Left: before, Right: after)

![The histogram averaging][averaging]

Then, I decided to convert the images to grayscale.

Here is an example of a traffic sign image before and after grayscaling.

![grayscale][gray]

As a last step, I normalized the image data because I want to equalize the influence of each feature. Since the feature is only the luminance value (0 - 255) at this time, the effect is small.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 gray image   							| 
| Convolution 5x5     	| 1x1 stride, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	    | 1x1 stride, outputs 10x10x16      									|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Fully connected		| outputs 84        									|
| RELU					|												|
| Dropout					|	probability 0.5											|
| Fully connected		| outputs 43        									|
| RELU					|												|
| Dropout					|	probability 0.5											|
| Softmax				|         									|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam optimizer to reduce the cross entropy loss.
* The batch size is 4
* The number of epoches is 10
* learning rate is 0.0005

Here is a transition of the loss.

![Transition of the loss][loss]

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.988
* validation set accuracy of 0.958
* test set accuracy of 0.939

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
  * My first architecture is the Lenet architecture without histogram averaging process because I implemented the Lenet architecture in the last lessen.
* What were some problems with the initial architecture?
  * The accuracy was low due to dark or unclear images.
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
  * I didn't change the model architecture from the original Lenet model. I only changed preprocess function.
* Which parameters were tuned? How were they adjusted and why?
  * I tried to change the learning rate in one digit increments. Finally, I choose 0.0005 as the learning rate from within [0.001, 0.0001].
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
  * It is important to choose filters that can extract enough features to separate each label. Because road signs can be identified by patterns, we can identificate them by extracting the feature of the shape by the convolution layer. Including the dropout layer is close to ensemble learning. Therefore, excessive learning can be prevented.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![Web images][web]

The second image might be difficult to classify because another sign is included in the image.
The background of the third image is cluttered.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Yield      		| Yield| 
| Bumpy road |No entry 										|
| No vehicles					| No vehicles|
| Speed limit 70km/h	      		| Speed limit 70km/h|
| Keep right			| Keep right      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%.

I wondered bumpy road and no entry signs are not resemble.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in traffic_sign_classifier.py.

1. Correct: Yield

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .9113         			| Yield   									| 
| .0550     				| Go straight or right 										|
| .0100					| Speed limit 60km/h|
| .0088	      			| Road work 				 				|
| .0079				    | Speed limit 50km/h      							|


2. Correct: Bumpy road

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .6846         			| No entry   									| 
| .1404     				| Speed limit 20km/h|
| .0603					| cycles crossing|
| .0247	      			| Speed limit 30km/h|
| .0188				    | Bumpy road      							|

3. Correct: No vehicles

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .9095         			| No vehicles   									| 
| .0652     				| Yield 										|
| .0066					| Speed limit 70km/h											|
| .0064	      			| Speed limit 120km/h					 				|
| .0063				    | Stop      						|

4. Correct: Speed limit 70km/h

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .9664         			| Speed limit 70km/h   									| 
| .0355     				| Speed limit 100km/h 										|
| .0000					| Speed limit 120km/h											|
| .0000	      			| Roundabout mandatory					 				|
| .0000				    | Speed limit 20km/h      							|

5. Correct: Keep right

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0000         			| Keep right   									| 
| .0000     				| Road work 										|
| .0000					| Yield											|
| .0000	      			| Go straight or right					 				|
| .0000				    | Turn left ahead      							|


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

Not yet.
