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

[image1]: ./writeup_images/image1.png "Dataset type distribution"
[image2]: ./writeup_images/image2.png "Data per class distribution"
[image3]: ./writeup_images/image3.png "Data augmentation result 1"
[image4]: ./writeup_images/image4.png "Data augmentation result 2"
[image5]: ./writeup_images/image5.png "Data augmentation result 3"
[image6]: ./writeup_images/image6.png "Training and validation accuracy"
[image7]: ./writeup_images/image7.png "Training and validation loss"
[image8]: ./writeup_images/image8.png "Sample traffic sign image 1"
[image9]: ./writeup_images/image8.png "Sample traffic sign image 2"
[image10]: ./writeup_images/image8.png "Sample traffic sign image 3"
[image11]: ./writeup_images/image8.png "Sample traffic sign image 4"
[image12]: ./writeup_images/image8.png "Sample traffic sign image 5"
[image13]: ./writeup_images/image9.png "Prediction result"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.
In this project, a traffic sign classifier using LeNet neural network architecture with TensorFlow and German Traffice Sign Dataset.
The project code is included in the workspace directory,
and the file name is 'Traffic_Sign_Classifier.ipynb'.


### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.
The pandas library was adopted to calculate summary statistics of the traffic
signs data set.

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

The visualized summary on distribution of training/validation/test dataset is depicted as a bar chart in [image1].

![alt text][image1]


#### 2. Include an exploratory visualization of the dataset.
An exploratory visualization of the data set is shown in [image2] as a bar chart showing the number of each class out of 43 classes.

![alt text][image2]


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale, using the function <rgb2gray(img_org)>.

Since the number of images per class is not uniform, where some classes have relatively more data over 1000, while others have less than 500, to prevent overfitting, data augmentation using affine transform was adopted to equalize the number of data per class. This procedure was done by a customized function <affineTransImg(img, range_angle, range_shear, range_trans)>.

Some results of this data augmentation is shown at [image3], [image4], and [image5].

![alt text][image3]
![alt text][image4]
![alt text][image5]


Data preprocessing procedure, which is operated by <preProcData(imgs, labs)> is designed as well as data augmentation. The data preprocessing procedure consist of normalization, done by <normalize(img_org)> function, and <claheHist(img_org)> function for adaptive histogram equalization. The data preprocessing procedure is applied to training set, validation set, and test set.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

For the model architecture, LeNet architecture is adopted. The architecture consists of the following layers.

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image						| 
|						|												|
| Conv1 5x5x1x6       	| 1x1 stride, valid padding, outputs 28x28x6	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6   				|
|						|												|
| Conv2 5x5x6x16       	| 1x1 stride, valid padding, outputs 10x10x6	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x6     				|
|						|												|
| FC3               	| Flatten,     outputs 400                  	|
| FC4               	| Flatten,     outputs 120                  	|
| FC5               	| Flatten,     outputs 84                   	|
| FC6               	| Flatten,     outputs 43                   	|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

At the beginning, to train the model, <tf.truncated_normal(shape, mu, sigma)> was adopted, but the training accuracy was congested around 96%. Therefore, initiliazation using Xavier initializer, <tf.contrib.layers.xavier_initializer()> was finally adopted, resulting in the training accuracy around 98%.

AdamOptimizer is adopted for optimization, and the number of epochs, the number of batches, and the learning rate are set to 150, 128, and 0.0005, respectively.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The final model results were:
* training set accuracy of 97.222
* validation set accuracy of 99.832
* test set accuracy of 99.8

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
LeNet. The application is kind of simple, and a simple and light architecture was concerned as enough.
* What were some problems with the initial architecture?
The initilization method of whether using Xavier initializer or Gaussian initialization was concerned.
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
The architecture was adjusted by adopting dropout, normalization, and equalization as well as data augmentation by generating additional data per class.
* Which parameters were tuned? How were they adjusted and why?
The number of epochs, batch sizes, and learning rate as typical hyper parameters were considered.

If a well known architecture was chosen:
* What architecture was chosen?
Accurate but not-too-heavy architecture such as VGG-16 can be considered.
* Why did you believe it would be relevant to the traffic sign application?
The dataset can be increased if needed because there are a lot of traffic sign images found on web, and their shape and aspect is somewhat standardized.

The overall progress is shown in [image6] and [image7], in terms of accuracy and loss variation with respect to epochs.

![alt text][image6]
![alt text][image7]

 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Following images are five German traffic sign images found on the web and preprocessed.

![alt text][image8] ![alt text][image9] ![alt text][image10] 
![alt text][image11] ![alt text][image12]

Some images with unclear background can be difficult to classify.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Turn right ahead 		| Turn right ahead								| 
| Roundabuot mandatory  | Roundabout mandatory							|
| Stop					| Stop											|
| Road work      		| Road work					 			     	|
| Yield     			| Yield             							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The prediction for image classification is performed by the function <predictImgSoftmax(list_imgs, list_label, list_probs, fig_size=(20,10)>.

The prediction result for the 5 test images from the web is summarized at [image13] below.

![alt text][image13]


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


