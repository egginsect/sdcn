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
[test_image1]: ./examples/placeholder.png "Traffic Sign 1"
[test_image2]: ./examples/placeholder.png "Traffic Sign 2"
[test_image3]: ./examples/placeholder.png "Traffic Sign 3"
[test_image4]: ./examples/placeholder.png "Traffic Sign 4"
[test_image5]: ./examples/placeholder.png "Traffic Sign 5"

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
Since the number of data for each class are unbalanced, I sample images in classes with replacement up to the same number.

I also normalized the image data because neural networks does better on normalized inputs which not deviate from zero too much.

Since traffic signs usually lies in centers of images, I crop images at center and rescale to original size.

In order to make model more robust to distorted or poor-quality images, I decided to generate additional data by the following data augmentation techniques:
* Random roate by -20 to 20 degree
* Random brightness
* Random contrast



#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.
Based on LeNet, I add few layers according to VGG16 to make the model more complex.
To make training faster, I use SELU activation to make inputs of layers normalized. 
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
The model is trained with Adam optimizer with the initial learning rate on 0.001. During training I drop the learning rate by the factor of ten one time (1e-3). The batch size is 80. The number of epochs was 89.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy ~ 97%
* validation set accuracy 93.5%
* test set accuracy of 93.6%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
I first chose LeNet since it's a simple model and easier to train. I also use data augmentation including flip up-down, flip left-right and random crop.
* What were some problems with the initial architecture?
The accuracy was really bad. I later realized that flipping images changes the meaning of signs. I also found the training accuracy of LeNet didn't exceed 90%, I think the model architecture was not complex enough to learn information.
* How was the architecture adjusted and why was it adjusted? 
I use the following data augmentation instead: random brightness, random contrast, random rotate for small angle. To make the training faster I also use Self-Normalized Linear Unit. I also add a f
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

