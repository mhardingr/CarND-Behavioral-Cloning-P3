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

TODO

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
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

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py final-training/final-model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

The model.py file has the following usage details:
```sh
usage: model.py [-h] --training-directory TRAINING_DIR --output-summary
                OUT_SUMMARY --training-config TRAINING_CONFIG
                --output-checkpoint OUTPUT_CHECKPOINT --training-round
                TRAINING_ROUND [--input-checkpoint INPUT_CHECKPOINT]

Configurable training pipeline

optional arguments:
  -h, --help            show this help message and exit
  --training-directory TRAINING_DIR
                        Path to training data folder (containing *.csv files)
  --output-summary OUT_SUMMARY
                        Output filename for summary of training
  --training-config TRAINING_CONFIG
                        Hyperparamter configuration file
  --output-checkpoint OUTPUT_CHECKPOINT
                        Output checkpoint filename for trained model
  --training-round TRAINING_ROUND
                        Training round number (selects CSVs to use as training
                        data)
  --input-checkpoint INPUT_CHECKPOINT
                        (Optional) Starting checkpoint filename for training
```

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model's architecture is outined by the global variable `ARCH_LAYERS` defined in model.py:30-48. This model is a slight modification of the NVIDIA paper "End to End Learning for Self-driving Cars" ([link](https://arxiv.org/pdf/1604.07316v1.pdf)).
In particular, the input layer expects images of size (160,320,3) converted to the YUV colorspace. This is followed by a standard normalization layer in which pixels are scaled to the range [-.5,.5].
Next, a cropping layer precedes the convolutional layers to remove the top 50 and bottom 20 pixels, thereby removing the influence of the features of the sky and the hood of the car on the steering commands of the model.
The convolutional layers are exactly the same as in the NVIDIA paper's CNN, except there is now an additional final layer of of 64 3x3 kernels. Theese feature-extraction layers are connected by ReLU activation functions.
After the feature-extraction layers, the output is flattened and concatenated with an additional input, the current speed of the car as measured by the simulator. 
This allows the steering commands to be influenced by the current speed of the car.
Lastly, the final 3 layers are fully connected layers such that 2 RELU-activated layers transform the output to a vector of size 64, which is converted to a steering signal between [-1., 1] by a final Dense layer with a `tanh` activation.

#### 2. Attempts to reduce overfitting in the model

To reduce overfitting, the model was trained using a configurable split of the input dataset into training data and validation data (model.py:162). Additionally, several methods of data augmentation (namely, horizontal flipping all input images and utilizing the left and right camera images) were introduced at training time to cancel a potential bias for left turns and instability when driving off the center of the track. 
The model was also trained in "rounds" which consisted of meaningful slices of a larger dataset: round 1 included the sample dataset exemplifying normal driving around the track, round 2 consisted of a much larger dataset including a number of forward and backward laps of the track, and round 3 purely contained images emphasizing difficult turns and examples of recoveries to the center of the track lane.
This incremental approach allowed for intermediate evaluation and targeted learning of the model, which allowed for a more coherent ability to navigate track 1 and track 2.

#### 3. Model parameter tuning

As shown in the usage for model.py, a `training-config` parameter must be defined which refers to a file containing values for several hyperparameters. These
hyperparameters are maintained at runtime within the global variable `HP_DICT` (defined model.py:20) and referenced throughout the file.
The hyperparameters include the number of training epochs, the training's patience for plateaued performance, the batch-size, the initial learning rate,
the name of the optimizer (i.e. "Adam"), the adjustment value for the steering value of left and right images, the proportion of the validation set,
and whether or not to freeze the layers before the fully-connected layers during training.

The Adam optimizer was heavily used to simplify the learning rate tuning, however the `start_lr` parameter was used to limit the influence of training on later rounds of training to allow for incremental and consistent model training and improvements.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, and driving the track in the forwards and backwards positions. Most of the training data was collected at or just below the top-speed of 30mph.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
