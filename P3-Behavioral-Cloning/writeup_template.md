# **Behavioral Cloning** 

## Writeup Report

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/01-nonflip.jpg "Normal Image"
[image2]: ./examples/02-flip.jpg "Flipped Image"
[image3]: ./examples/03-left.jpg "Left Camera Image"
[image4]: ./examples/04-center.jpg "Center Camera Image"
[image5]: ./examples/05-right.jpg "Right Camera Image"
[image6]: ./examples/Img01.JPG "Training and Validation Loss"
[image7]: ./examples/Img02.JPG "MSE vs Epoch"
[image8]: ./examples/Img03.JPG "Nvidia Model Architecture"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py, Model.ipynb containing the script to create and train the model. I worked in Jupyter Notebook as I am more used to with it due to earlier projects but project requirement was to submit model.py so I submitted both the formats. Also, there is Model.html as a snapshot of notebook.
* drive.py for driving the car in autonomous mode. I did not modified anything in file and submitting it as it is.
* model.h5 and model.h52 containing a trained convolution neural network. The first file model.h5 was trained for 3 epochs and I observed that during my 3rd epoch, the training and validation loss was little more compared to 2nd epoch. So I retrained the model only for 2 epochs and stored it in model.h52. I am submitting both the models here and they both work fine on my system.
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

Alternatively, for 2nd model, one can use following command:

```sh
python drive.py model.h52
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

As mentioned in the project rubric as well as during the classroom video 17, I defined a generator function (line 18 to 55 of model.py) yeilding batch data for each batch of training. From what I learned, it reduces the overhead on system memory by generating data on the fly instead of storing it in the memory. Then I used model.fit_generator() (line 94 of model.py) of keras to leverage defined python generator function.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I followed the classroom lectures and wrote code for each and every model discussed in the classroom. For final training, I used Nvidia model which was again discussed in the classroom.(lines 81-90 of model.py). As explained in the classroom, I leveraged Lambda layers to do normalization as well as Mean Centring of data at line 75 of model.py.

Relu Layers were provided in terms of activation of Convolutional Layers to introduce non-linearity in the model.

Keras class Cropping2D was also used to remove top 70 and bottom 25 pixels from each image data. This was again covered in the classroom lecture which was really efficient and beneficial technique available in Keras. 

#### 2. Attempts to reduce overfitting in the model

In order to avoid overfitting, I augmented the data as per the recommendations provided in the classroom videos. I used images from all the 3 cameras i.e. Left, Center and Right (Line no 29 to 39 in model.py).  I used a correction factor of 0.2 (line 20) for Left and Right Image and their steering angle measurement. And then all of this data was flipped horizontally in order to augment data (Line 43 to 48). As a result my total data size becomes len(train_samples)*2*3 = 38568 images. Following images illustrates five different types of images:

![alt text][image1]

![alt text][image2]

![alt text][image3]

![alt text][image4]

![alt text][image5]

I used sklearn.utils.shuffle() for randomization of training data, and sklearn.model_selection.train_test_split() to create Validation data from training data with the test_size of 0.2 (line 16). These methods were already used in P2 and were also discussed in classroom of P3 in accordance with generator function which became very helpful.

All of these techniques were implemented to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track. A video recording of autonomous mode testing of the model is also provided as per the requirements.

#### 3. Model parameter tuning

My model actually used adam optimizer so I did not had to apply momentum decay or other methods of moderating learning rate and the model was tuned automatically due to the optimizer chosen. (line 92 - model.py).

#### 4. Appropriate training data

I tried to record few training labs of my own by manually driving in the simulator but since I have a i3 4GB machine with no on-chip graphics card, my system tend to crash while trying to generate training data. So I used the training data provided by Udacity itself and implemented augmentation techniques which were discussed in the classroom to increase the data size and generalize the model. My model happen to work fine with the provided dataset on Nvidia model after implementing augmentation techniques and data shuffling. Although my Lenet model did not performed well at all with Udacity's data but since I am not including it in the submission, I am not highlighting it here much. I will try to find the root cause and methods which can help Lenet to perform well in the simulator in future when the time permits.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I followed the guidelines provided in the classroom lectures. I tried to implement Lenet as a model and trained initially only for center camera images for around 15 epochs. The model clearly did not performed well and I was getting better training and validation accuracy but the car was not driving well at all. So then I preprocessed the data and applied shuffling to train the model better. This time I trained it for 25 epochs and my model was overfitting again. Besides the loss tend to oscillate after 18th epoch. So I retrained the model only for 18 epochs and tested it on the simulator but no luck getting the desired results.

I then kept the all the pre-processing techniques as it is but only changed the model to Nvidia one and the model performed drastically well. I heard from my mentor as well as other students on slack that they were able to perform well using Lenet itself but I don't know why I was not able to do so. I will try to figure out the root cause behind this whenever I get chance in future but since I am way behind my project submission deadline, I stuck with Nvidia model to finish my project.

#### 2. Final Model Architecture

As I mentioned earlier, my final model uses Nvidia Model Architecture. Original Nvidia Model Architecture consists of 5 Convolutional layers in the following order: 24@31x98, 36@14x47, 48@5x22, 64@3x20, 64@1x18. Then it has a flatten layer followed by 3 fully connected layers outputting 100 neurons, 50 neurons and 10 neurons. Below I have provided an image of this model from their paper: "End to End Learning for Self-Driving Cars"

![alt text][image8]

I had to alter this model in a way that I had to remove the last convolution layer of filter size = 64 and kernal size = 3. I had to remove this layer because my model was throwing an exception of negative dimensions at this layer. I believe this could have occurred because of valid padding and the dimensions of the input to this layer would have reduced so significantly that it was not receiving expected image dimensions. 

A better way to resolve this would be to analyse the root cause and if it is what I expected, then either we need to add zero padding to maintain image dimension or we need to alter earlier layers such that this layers receives image in proper dimensions. For now, I provided a work around by removing this layer and my model happened to work fine in simulator even without this layer. Although I highly consider that removal of this layer could lead to unexpected effects in real world where the tracks are not as good as simulator one. Since this was the last convolution layer, it was supposed to identify high level features which are not extracted any more in my model. But there are 4 convolutional layers apart from this one which did a great job for me.

#### 3. Creation of the Training Set & Training Process

![alt text][image6]

![alt text][image7]

Like I mentioned above, I tried to record the training lap but due to limitation of hardware resources at my end, I could not do it so I used the data provided by Udacity. I trained mainly two models with that data: Lenet and Nvidia model. Just for fun, I followed all the ways explained in the classroom sessions and then picked the best out of all those.

My final model which is Nvidia's Model was trained two times. First time, I trained it for 3 epochs and tested the model in simulator. It worked fine to my surprise in the first go itself when I brought Nvidia model into the picture. It was not working well with lenet I don't know why. But I observed that in my 3rd epoch, my training and validation loss increased compared to the 2nd epoch which indicated overfitting. So I cleared the notebook kernal and trained the model again, but this time only for 2 epochs. I saved this model as well and tested it on simulator and it worked same as the previous one which was great. These two models are provided in model.h5 and model.h52 files respectively. And as per rubric requirements, I created a video file for around 2 laps using the 2nd model's data which is also submitted as a part of project submission. The video file is recorded for 48 FPS as 60 FPS appeared to be very fast to me.