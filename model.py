from keras.layers import Input, Dense, Conv2D, Flatten, Lambda, Cropping2D, Concatenate, AveragePooling2D
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, Callback
from sklearn.model_selection import train_test_split
from scipy import ndimage

import argparse
import math
import os.path
import sys
import tensorflow as tf
import numpy as np
import cv2
import json
import glob
import signal
import os

HP_DICT = {
        'epochs': -1,
        'stop_patience': -1,
        'batchsize': -1,
        'start_lr': -1.0,
        'optimizer': None,
        'static_cam_angle_adjust': -1.0,
        'validation_split': -1.0,
        'freeze_feature_layers':False
        }
ARCH_LAYERS = [
        # Input shape: (160,320)
        # Normalize YUV
        Lambda(lambda x: x/255 - 0.5),
        # Crop top 50 pixels, bottom 20 pixels
        Cropping2D(cropping=((50,20),(0,0))),
        # Input shape: (90, 320)
        Conv2D(24, 5, strides=(2,2), padding="valid", activation='relu'), # SAME
        Conv2D(36, 5, strides=(2,2), padding="valid", activation='relu'), # SAME
        Conv2D(48, 5, strides=(2,2), padding="valid", activation='relu'), # SAME
        Conv2D(64, 3, padding="valid", activation='relu'),
        Conv2D(64, 3, padding="valid", activation='relu'),
        Conv2D(64, 3, padding="valid", activation='relu'),
        Flatten(),
	Concatenate(),
        Dense(200, activation='relu'),
        Dense(64, activation='relu'),
        Dense(1, activation='tanh')
        ]

IMAGE_SHAPE = (160,320, 3)

def init_model(from_checkpoint=None):
    freeze_feature_layers = HP_DICT['freeze_feature_layers']

    inp = Input(shape=IMAGE_SHAPE)
    speed = Input(shape=(1,))
    _inp = inp
    out = None
    for layer in ARCH_LAYERS:
        if isinstance(layer, Concatenate):
            out = layer([_inp, speed])
        else:
            out = layer(_inp)
        _inp = out
    steering = out
    model = Model(inputs=[inp,speed], outputs=steering)
    # Due to TF bug, we can only freeze layers after Model init
    # which is useful for certain rounds in which we want to tune
    # only the network head (the Dense/FC layers)
    if freeze_feature_layers:
        for layer in model.layers:
            if not isinstance(layer, Dense):
                layer.trainable=False

    opti = None
    if HP_DICT['optimizer'] == "Adam":
        opti = Adam(lr=HP_DICT['start_lr'])
    else:
        print("Bad optimizer!")
        sys.exit(-1)

    if from_checkpoint:
        print("Loading weights from input checkpoint (%s) ..."%from_checkpoint)
        model.load_weights(from_checkpoint)

    # Compile the deep regression model with the chosen optimizer
    model.compile(opti, loss="mean_squared_error", metrics=['mse'])
    return model


def augmented_data_from_csv_gen(csv_lines):
    n_samples = len(csv_lines)
    cam_adjust = HP_DICT['static_cam_angle_adjust']
    while True:
        # Shuffle at the beginning of each epoch
        np.random.shuffle(csv_lines)

        batchsize = HP_DICT['batchsize']
        for i in range(0, n_samples, batchsize):
            lines_batch = csv_lines[i:i+batchsize]

            # Augment minibatch by extracting images referred to
            # by X_batch entries and transforming them and the
            # associated steering angle
            images = []
            speeds = []
            steering = []
            for line in lines_batch:
                csv_entries = line.split(',')
                paths, y, speed = csv_entries[:3], float(csv_entries[3]), float(csv_entries[-1].strip())
                for i, path in list(enumerate(paths)):
                    # Preprocess the path (in case paths are from Windows with "\\")
                    path = path.replace("\\",'/')
                    path = args.training_dir + "IMG/" + path.split('/')[-1].strip()
                    # Open image using RGB, convert to YUV, save a LR-flipped copy
                    img = cv2.cvtColor(ndimage.imread(path), cv2.COLOR_RGB2YUV)
                    flipped_img = np.fliplr(img)
                    images.extend([img, flipped_img])
                    speeds.extend([speed, speed])
                    if i == 0:
                        # Center image needs no steering adustment
                         steering.extend([y, -y])
                    elif i == 1:
                        # Left image needs positive adjustment
                        steering.extend([y+cam_adjust, -(y+cam_adjust)])
                    elif i == 2:
                        # Right image needs negative adjustment
                        steering.extend([y-cam_adjust, -(y-cam_adjust)])
                    else:
                        # Shouldn't be here
                        raise Exception("More images per sample than expected!")
            # Shuffle augmented arrays because don't want to force ordering of batch
            same_state = np.random.get_state()
            np.random.shuffle(images)
            np.random.set_state(same_state)
            np.random.shuffle(speeds)
            np.random.set_state(same_state)
            np.random.shuffle(steering)
            # Convert to numpy arrays
            images = np.array(images)
            speeds = np.array(speeds)
            steering = np.array(steering)
            yield [images, speeds], steering

def handle_ctrl_c(signum, frame):
    global stop_training
    print("Caught CTRL+C, marking this as the last epoch ...")
    stop_training=True

signal.signal(signal.SIGINT, handle_ctrl_c)
stop_training = False

# Create custom Keras callback for stopping training when Ctrl-C sets glbl flag
class FlagStopCallback(Callback):
    def on_epoch_end(self, end, logs):
        global stop_training
        if stop_training:
            self.model.stop_training = True

def train_model(model, training_lines):
    # Shuffle then split input dataset into training and validation samples
    train_samples, validation_samples = train_test_split(training_lines,
            test_size=HP_DICT['validation_split'], shuffle=True)
    n_tr_samples = len(train_samples)
    n_val_samples = len(validation_samples)

    # Initialize generators and fit_generator args
    tr_gen = augmented_data_from_csv_gen(train_samples)
    valid_gen = augmented_data_from_csv_gen(validation_samples)
    steps_per_epoch = math.ceil(n_tr_samples / HP_DICT['batchsize'])
    validation_steps = math.ceil(n_val_samples / HP_DICT['batchsize'])

    # Initialize the callbacks for training. TODO: ModelCheckpointing?
    earlystop = EarlyStopping(patience=HP_DICT['stop_patience'],
            verbose=1)
    flagstop = FlagStopCallback()

    # Train the model using training and validation generators
    tr_history = model.fit_generator(tr_gen, steps_per_epoch=steps_per_epoch,
            epochs=HP_DICT['epochs'], verbose=1,
            validation_data=valid_gen, validation_steps=validation_steps,
            callbacks=[earlystop, flagstop])
    return tr_history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="Configurable training pipeline")
    parser.add_argument('--training-directory', dest="training_dir",
            required=True,
            help="Path to training data folder (containing *.csv files)")
    parser.add_argument('--output-summary', dest="out_summary",
            required=True,
            help="Output filename for summary of training")
    parser.add_argument("--training-config", dest="training_config",
            required=True,
            help="Hyperparamter configuration file")
    parser.add_argument("--output-checkpoint", dest="output_checkpoint",
            required=True,
            help="Output checkpoint filename for trained model")
    parser.add_argument("--training-round", dest="training_round",
            required=True, type=int,
            help="Training round number (selects CSVs to use as training data")
    parser.add_argument("--input-checkpoint", dest="input_checkpoint",
            required=False, default=None,
            help="(Optional) Starting checkpoint filename for training")
    args = parser.parse_args()

    # Validate arguments
    if os.path.isfile(args.out_summary):
        print("Cannot overwrite existing output summary file!")
        sys.exit(-1)
    if os.path.isfile(args.output_checkpoint):
        print("Cannot overwrite existing output checkpoint file!")
        sys.exit(-1)
    if not os.path.isfile(args.training_config):
        print("Cannot open training configuration file!")
        sys.exit(-1)
    if (args.input_checkpoint and not os.path.isfile(args.input_checkpoint)):
        print("Cannot open input checkpoint file to initialize model!")
        sys.exit(-1)
    if (args.training_round < 1 or args.training_round > 4):
        print("Invalid training round number %d" % args.training_round)
        sys.exit(-1)

    # Read in and validate hyperparameters
    with open(args.training_config) as config_f:
        for line in config_f.readlines():
            hypname, hypval = line.strip().split('=')
            if hypname not in HP_DICT:
                print("Invalid hyperparameter found in training config: %s" % hypname)
                sys.exit(-1)
            else:
                hyptype = type(HP_DICT[hypname])
                try:
                    if not hyptype == type(None):
                        hypval = hyptype(hypval)
                    HP_DICT[hypname] = hypval
                except Exception as e:
                    print("Invalid hyperparameter value found in training config: %s must be %s." \
                            % (hypname, str(hyptype)))
                    print(e)
    # Print out hyperparameters
    print("Using following hyperparameter values:\n%s" % HP_DICT)

    # Validate and read into memory image pathnames and steering angles
    # Select training_csvs using training round number
    training_csvs = glob.glob(args.training_dir + "tr_round%d/*.csv" % args.training_round)
    print("Using training_csvs:", training_csvs)
    training_lines = []
    for tr_fn in training_csvs:
        if not os.path.isfile(tr_fn):
            print("Cannot open training data file: %s. Quitting ..." % tr_fn)
            sys.exit(-1)
        with open(tr_fn) as tr_f:
            skipped_first = False
            for line in tr_f.readlines():
                if not skipped_first:
                    skipped_first = True
                    continue
                training_lines.append(line.strip())

    # Initialize the model
    model = init_model(from_checkpoint=args.input_checkpoint)
    model.summary()

    # Train the model
    try:
        history = train_model(model, training_lines)
    except KeyboardInterrupt as e:
        print("Caught Ctrl+C while training: %s\
\nCleaning up, checkpointing, and outputing summary..." % str(e))

    # Save the model
    model.save(args.output_checkpoint)

    # Output the training summary
    # Print commandline
    # Hyperparameter values
    # model summary
    # Number of epochs actually trained before training stopped
    # Training and validation losses
    tr_loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_trained = len(tr_loss)
    output_summary = "Training summary:\n"
    output_summary += "cmdline: " + " ".join(sys.argv) + "\n"
    output_summary += "Training data: " + ", ".join(training_csvs) + "\n"
    output_summary += "Hyperparameters:\n\t" \
        + "\t".join(map(lambda k: "{}={}\n".format(k, str(HP_DICT[k])), HP_DICT.keys()))\
        + "\n"
    output_summary += "Total epochs trained: %d\n" % epochs_trained
    output_summary += "Training loss: %4.6f; Validation loss: %4.6f\n" % (tr_loss[-1], val_loss[-1])
    print("+"*60+"\n"+output_summary)
    model.summary()
    with open(args.out_summary, 'w') as out_f:
        out_f.write(output_summary)
        model.summary(print_fn=lambda x: out_f.write(x + "\n"))

    # Save the history object for later visualization
    hist_out = "hist_" + args.out_summary.split('.')[0] + ".json"
    with open(hist_out, 'w') as hist_f:
        json.dump(history.history, hist_f)
        print("Done dumping history to file: %s" % hist_out)
