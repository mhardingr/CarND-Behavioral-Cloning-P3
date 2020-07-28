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

[image0]: ./examples/round1/left_2016_12_01_13_31_14_803.jpg "Round 1 left"
[image1]: ./examples/round1/center_2016_12_01_13_31_14_803.jpg "Round 1 center"
[image2]: ./examples/round1/right_2016_12_01_13_31_14_803.jpg "Round 1 right"
[image3]: ./examples/round2/forward_center_2020_02_04_20_32_18_752.jpg "Round 2 example forward"
[image4]: ./examples/round2/backward_center_2020_02_04_20_41_20_477.jpg "Round 2 example reverse"
[image5]: ./examples/round2/random_center_2020_02_06_06_21_17_458.jpg "Round 2 example random"
[image6]: ./examples/round3/dirt_center_2020_02_07_16_23_50_927.jpg "Round 3 example dirt"
[image7]: ./examples/round3/sharp_left_center_2020_02_07_16_20_28_655.jpg "Round 3 example sharp"
[image8]: ./examples/flipped/center_2016_12_01_13_31_14_803.jpg "Normal image"
[image9]: ./examples/flipped/flipped_center_2016_12_01_13_31_14_803.jpg "Flipped Image"

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

### Architecture and Training Documentation

#### 1. Solution Design Approach

I chose to begin my architecture search with the NVIDIA paper's proven but simple CNN model, as described above. To account for the network simpler input space (due to the simulator's relative simplicity), I made modifications
that would speed up training as well as combat overfitting, such as (statically, within the network) cropping the sky and the ego-car's hood from the image input (as discussed above).
One other change I made from the original architecture was to add a 6th convolutional layer, which mostly served the purpose of halving the size of the feature-extraction output and improved training times.
The final fully-connected layers are the most computationally and memory expensive to compute, so halving their input size was significant.

#### 2. Final Model Architecture

The final model architecture (model.py:30-48) is as described under Model Architecture and Training Strategy, section 1. The verbose model architecture is outline below in Tensorflow output format:

```
Layer (type)                    Output Shape         Param #     Connected to
==================================================================================================
input_1 (InputLayer)            (None, 160, 320, 3)  0
__________________________________________________________________________________________________
lambda_1 (Lambda)               (None, 160, 320, 3)  0           input_1[0][0]
__________________________________________________________________________________________________
cropping2d_1 (Cropping2D)       (None, 90, 320, 3)   0           lambda_1[0][0]
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 43, 158, 24)  1824        cropping2d_1[0][0]
__________________________________________________________________________________________________
conv2d_2 (Conv2D)               (None, 20, 77, 36)   21636       conv2d_1[0][0]
__________________________________________________________________________________________________
conv2d_3 (Conv2D)               (None, 8, 37, 48)    43248       conv2d_2[0][0]
__________________________________________________________________________________________________
conv2d_4 (Conv2D)               (None, 6, 35, 64)    27712       conv2d_3[0][0]
__________________________________________________________________________________________________
conv2d_5 (Conv2D)               (None, 4, 33, 64)    36928       conv2d_4[0][0]
__________________________________________________________________________________________________
conv2d_6 (Conv2D)               (None, 2, 31, 64)    36928       conv2d_5[0][0]
__________________________________________________________________________________________________
flatten_1 (Flatten)             (None, 3968)         0           conv2d_6[0][0]
__________________________________________________________________________________________________
input_2 (InputLayer)            (None, 1)            0
__________________________________________________________________________________________________
concatenate_1 (Concatenate)     (None, 3969)         0           flatten_1[0][0]
                                                                 input_2[0][0]
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 200)          794000      concatenate_1[0][0]
__________________________________________________________________________________________________
dense_2 (Dense)                 (None, 64)           12864       dense_1[0][0]
__________________________________________________________________________________________________
dense_3 (Dense)                 (None, 1)            65          dense_2[0][0]
==================================================================================================
Total params: 975,205
Trainable params: 975,205
Non-trainable params: 0
__________________________________________________________________________________________________   
```

#### 3. Creation of the Training Set & Training Process

For the creation of the training data, I followed the overarching principles of: 1) have a lot of good quality data, 2) balance that data as much as possible, and 3) include examples of recoveries for model robustness.
I used the 8037 images from the sample training set, and added ~25k images of my own. ~21k of my images were recorded from ~5 laps of good driving (roughly center-lane only) for both forward and reverse laps, and a small focused subset of ~3k images were recorded purely focused on sharp turns, edge-cases like turns with a dirt border, and some recovery maneuvers from the sides of the lane back to the center.

Some notes on the training method itself:
To train my model, I leveraged Python generators to minimize memory pressure and CPU computing bottlenecks - i.e. the batches of images and target steering angles were prepared on-the-fly from the CSV files.
In preparing the final model, a training round number selected the CSV files (and therefore the subset of training set) to learn from. The final model was 
trained using the same batch size of 16x3x2 (96 images) - 16 lines from the CSV, which included 3 images for left/right/center that were each flipped once horizontally.
The left and right images were leveraged to introduce robustness to the model because these images were associated with a statically adjusted steering angle - the left image added .15 to the center image's steering value, and right image subtracted this amount.
Concerning the dataset for each round of training, the training data was first shuffled then split into 80% training data, 20% validation. Shuffling occurred between epochs and within training batches produced by the generator.
Lastly, training rounds were only successful if the validation error decreased appreciably from the start of training, and so long as the model at least maintained its
prior rounds' level of driving performance. Note: the evaluation of the model in simulation after rounds of training was done at a speed almost half that of the training set because this seemed to greatly dampen steering oscillations likely due to a relatively small overall training dataset.

Overall training story:
As mentioned, training occurred in rounds in order to incrementally introduce improved driving behavior into the model. Round 1 subset of the training data included on the sample training data in order to exemplify both image features and steering angles of good driving for a lap or two.
Round 2's subset was a much larger dataset consisting of the 5+ laps of my own driving on track 1 for both forward and reverse directions (i.e. ~10 laps of good driving). 
Lastly, round 3's subset was a focused small dataset focused on edge-cases and recoveries.

Sample images from each round of training data:

Round 1 (sample dataset):

![alt text][image0]
![alt text][image1]
![alt text][image2]

Round 2 (bulk "good" dataset, balanced with forward and backward laps):

![alt text][image3]
![alt text][image4]
![alt text][image5]

Round 3 (recovery dataset):

![alt text][image6]
![alt text][image7]

Below is an example of the data augmentation done within the generator: horizontal flipping of any of the left/right/center images.

![alt text][image8]
![alt text][image9]


The training story was therefore: after training 10 epochs of round 1 training data with a large starting learning rate (1e-4), the model effectively learned how to successfully drive to the sharp left turn after the bridge.
This model was capable of reaching the other end of the bridge 4/5 times, and would exhibit large steering oscillations while still staying within the driving surface of the road.
Therefore, the purpose of the second round of training would be to greatly dampen the steering oscillations and bolster the robustness of this starting model. It would be the focus of the third round to teach the model to navigate the dirt-bordered and other sharp turns without taking away from the non-edge case performance of the model learned from rounds 1 and 2.
This was done successfully by training the model in round 2 for 10 epochs using a smaller starting learning rate (5e-6) and frozen feature-extraction layers (effectively "transfer learning" - only the final fully-connected layers were tunable) followed by 5 epochs of round 3 training data with the same small learning rate (5e-6) and thawed feature-extraction layers (thereby enabling the model to learn the features of dirt-bordered turns and sharp turns and to recognize when recovery maneuvers were necessary).
A key element of this form of incremental training was to decrease the starting learning rates in order to prevent "overwriting" the improvements of the previous model, and this is also a form of combatting overfitting.
It can be said that a mix of "transfer learning" and "fine-tuning" was used by the second and third rounds of training in order to implant certain behaviors into a good base driving model.
