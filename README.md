#**Traffic Sign Recognition** 

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

[image1]: ./examples/train_examples.png "Visualization"
[image2]: ./examples/train_examples_gray.png "Grayscaling"
[image3]: ./Images/30.jpg "30"
[image4]: ./Images/priority.jpg "Priority"
[image5]: ./Images/road_work.jpg "Road work"
[image6]: ./Images/slippery_road.jpg "Slippery road"
[image7]: ./Images/stop.jpg "Stop"
[image8]: ./examples/distribution-train.png "Distribution train"
[image9]: ./examples/distribution-validation.png "Distribution validation"
[image10]: ./examples/distribution-test.png "Distribution test"


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/debuggio/Udacity-SDC-TrafficSignClassifier/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Distribution of the labels is presented:
![Distribution train][image8]
![Distribution validation][image9]
![EDistribution test][image10]


###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because there is no need to process colors, plus in different countries colors may deffer

Here is an example of a traffic sign image before and after grayscaling.

![Example grayscale images][image2]

####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| convolution 	     	| 1x1 stride, valid padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 					|
| Convolution	  		| 1x1 stride, valid padding, outputs 10x10x16	|
| RELU 					|												|
| Max pooling	      	| 2x2 stride, valid padding, outputs 5x5x6		|
| Fully connected		| 120      										|
| RELU 					| 	        									|
| Dropout				| 0.5 keep probability for training set 		|
| Fully connected		| 84      										|
| RELU 					| 	        									|
| Dropout				| 0.5 keep probability for training set 		|
| Fully connected		| 43       										|
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I started these parameters:
* EPOCHS = 10
* KEEP_PROBABILITY_TRAIN = .5
* BATCH_SIZE = 128
* RATE = 0.001

RESULTS
* train_accuracy: 0.92
* Validaition_accuracy: 0.85

after some tuning I ended up with:
* EPOCHS = 9
* KEEP_PROBABILITY_TRAIN = .5
* BATCH_SIZE = 50
* RATE = 0.002

RESULTS
* train_accuracy: 0.987
* Validaition_accuracy: 0.963

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

But I wasn't happy, because it only recognized 3 out of 5 images from web. Plus starting from 7 ehoch accuracy didn't raise at all (even for 20+ epoches)
As a result I picked lower learning rate with more epoches

* EPOCHS = 20
* KEEP_PROBABILITY_TRAIN = .5
* BATCH_SIZE = 100
* RATE = 0.0009

My final model results were:
* training set accuracy of 0.994
* validation set accuracy of 0.960 
* test set accuracy of 0.916

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen? 
** At first I tried LeNet from 1 of previous labs and it gave me around 85% on validation set, that is too low
* What were some problems with the initial architecture? 
** There were many problems started with not the best values for parameters ending with too simple model
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
** I added 2 dropout step, so network wont rely on any given activation to be present. So it learns redundant representation for everything
* Which parameters were tuned? How were they adjusted and why?
** Tuned all parameters. 
*** Epoches - because model gave a pretty small accuracy for chosen learn for other 3 parameters
*** keep probability - to have only 50% of data passed through, to prevent overfitting (50% was moetioned in video)
*** batch size - less than 128 gave better results, this value was chosen kind of empirically
*** learning rate - I started from 0.02 and reduced it until learning accuracy started growing more or less smoothly and gave better results
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
** Adding 2 droupout layers improved my model accuracy for validation set for around 5%, because there was a problem with overfitting. 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

I tried to find images that we can meet in real life. Images with different size (scalling should produce some troubles for trained model), with different backgrounds (trees, buildings). Also I'm using 1 image with watermarks on it to make things more complicated. Also I'm using image with number 30 on it. It might be missclassified for 50 or 80, because numbers after resizing may look almost the identically

Here are five German traffic signs that I found on the web:

![speed limit 30][image3] ![priority][image4] ![road_work][image5] 
![slippery_road][image6] ![stop][image7]

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| Priprity road 		| Priprity road									|
| Road work				| Road work										|
| 30 km/h	      		| 50 km/h					 					|
| Slippery Road			| Slippery Road      							|

The first image (speed limit 30) might be difficult to classify because it has watermarks, some angle and 30 looks kind of like 50. As a result I got 99% that it's Speed limit 50km/h. I think if I spend more time on processing initial images, this problem should gone

The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. I'm not happy with this results, because 80% is less than 91 from test set validation. I think that images requires more pre-processing. 1 thing that I used some images to challenge my model (like: 30 with watermarks and trees behing and stop sign with a blur). I'm thinking about adding additional filters for grayscaled image, but I'll play with it later

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 57th cell of the Ipython notebook.


| Probability         	| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------:|:---------------------------------------------:| 
| .99 (wrong)  			| 30km/h      			| 50km/h (0.998690) 30km/h (0.001254)			| 
| 1     				| Priprity road 		| Priprity road									|
| .99					| Road work				| Road work										|
| .88	      			| Slippery road			| Slippery road									|
| 1					    | Stop      			| Stop											|


As I described above, image is a bit challenging, unfortunatelly for now, my model is not able to handle it
