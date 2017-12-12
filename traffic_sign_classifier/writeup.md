# **Traffic Sign Recognition** 

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[taining_image]: ./figures/training_data.png "Training Image"
[training_data_statics]: ./figures/training_data_statics.png "Training Data Statistics"
[testing_images]: ./figures/testing_images.png "Testin Image"
[test_top5]: ./figures/top5_result.png "Top 5 result"

---
### Writeup / README

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing the histogram of different classes in the data. We can see the dataset are highly imbalanced.

![training_data_statics]

Here are some examples of training data.

![taining_image]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)


As a first step, I decided to convert the images to grayscale because it's more robust to color distortion.
Since the number of data for each class are unbalanced, I sampled images in classes with replacement up to the same number.

I also normalized the image data because neural networks does better on normalized inputs which not deviate from zero too much.

Since traffic signs usually lies in centers of images, I crop images at center and rescale to original size.

In order to make model more robust to distorted or poor-quality images, I decided to generate additional data by the following data augmentation techniques:
* Random roate by -20 to 20 degree
* Random brightness
* Random contrast



#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.
Based on LeNet, I add few layers according to VGG16 to make the model more complex.
To make training faster, I use Self-Normalized Linear Unit (SELU) activation to make inputs of layers normalized. 
Normalized inputs make the optimization faster since the gradient updates in different coordinates are more uniform.
My final model consisted of the following layers:

Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 32x32x6 	|
| SELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x6 				|
| Convolution 5x5     	| 1x1 stride, same padding, outputs 16x16x16 	|
| SELU					|												|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 16x16x32 	|
| SELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 8x8x32 				|
| Dropout            | keep probability 0.8  |
| Convolution 3x3     	| 1x1 stride, same padding, outputs 16x16x32 	|
| SELU					|												|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 16x16x64 	|
| SELU					|												|
| Dropout            | keep probability 0.8  |
| Fully Connected 			|	outputs 120											|
| SELU					|												|
| Dropout            | keep probability 0.8  |
| Fully Connected 			|	outputs 120											|
| SELU					|												|
| Softmax       |number of classes 43   |

 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.
The model was trained with Adam optimizer with the initial learning rate on 0.001. During training I dropped the learning rate by the factor of ten one time (1e-3). The batch size was 80. The number of epochs was 89.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy ~ 97%
* validation set accuracy 93.5%
* test set accuracy of 93.6%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?

  * I first chose LeNet since it's a simple model and easier to train. I also used data augmentation including flip up-down, flip left-right and random crop.
* What were some problems with the initial architecture?

  * The accuracy was really bad. I later realized that flipping images changes the meaning of signs. I also found the training accuracy of LeNet didn't exceed 90%, I thought the model architecture was not complex enough to learn to classify traffic signs.
* How was the architecture adjusted and why was it adjusted? 

  * I use the following data augmentation instead: random brightness, random contrast, random rotate for small angle. 
  * To make the training faster I also use SELU. 
  * I also crop center 64% of the image and resize, since most of traffic signs lies in the center.
* Which parameters were tuned? How were they adjusted and why?

  * I changed the dropout for regularization since the model was overfittings. However too much dropout damages the performance. I use keep probability of 0.8. 
  * During training I found optimization didn't make progress and jumping at certain value. It's possible the optimizer bounce back-and-forth when approaching the optimal due to too large learning rate. I drop the learning rate by the factor of 10 during the middle of the trainig and the model stated to make progress.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

  * Convolution are used since we want the model to capture spatial information. 
  * Pooling are used to do locally summarization of image patches. 
  * Fully connected layers are used to do summarization of the spatial features.
  * I use SELU as the activation function for faster and stable training.

If a well known architecture was chosen:
* What architecture was chosen?
  * LeNet was used as the base. But I added a few conconvolution layers.
* Why did you believe it would be relevant to the traffic sign application?
  * LeNet was initially used for MNist data. Since traffic signs are simple geometry figures, I believe it's useful.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
  * training set accuracy ~ 97%
  * validation set accuracy 93.5%
  * test set accuracy of 93.6%
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![testing_images]

The fourth image must harder to be classified since it's reflecting lights. Random brightness included in data augmentation should help.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Right-of-way at the next intersection    		| Right-of-way at the next intersection   									| 
| Speed limit (30km/h)     			| Speed limit (30km/h)										|
| Roundabout mandatory				| Roundabout mandatory											|
| Road work    		| Road work				 				|
| Keep right			| Keep right      							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 93%.


#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)
* training set accuracy ~ 97%
* validation set accuracy 93.5%
* test set accuracy of 93.6%

Below are top5 probability and it's label for each testing image:

![test_top5]

We can see the top 5 probability for each testing image often looks similiar ( e.g. The model was confused Speed limit (30km/h) with other speed limit signs).


