from keras.layers import Input, Dense, Conv2D, Flatten, Lambda, Cropping2D
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from scipy import ndimage

import argparse
import math
import os.path
import sys
import tensorflow as tf
import numpy as np

HP_DICT = {
        'epochs': -1,
        'stop_patience': -1,
        'batchsize': -1,
        'start_lr': -1.0,
        'optimizer': None,
        'static_cam_angle_adjust': -1.0,
        'validation_split': -1.0
        }
ARCH_LAYERS = [
        # Input shape: (160,320)
        Lambda(lambda x: tf.image.rgb_to_yuv(x)),
        # Normalize YUV
        Lambda(lambda x: x/255 - 0.5),
        # Crop top 50 pixels, bottom 20 pixels
        Cropping2D(cropping=((50,20),(0,0))),
        # Input shape: (90, 320)
        Conv2D(24, 5, strides=(2,2), padding="valid", activation='relu'),
        # Input shape: (24, 38, 158)
        Conv2D(36, 5, strides=(2,2), padding="valid", activation='relu'),
        # Input shape: (36, 17, 77)
        Conv2D(48, 5, strides=(2,2), padding="valid", activation='relu'),
        # Input shape: (48, 7, 32)
        Conv2D(64, 3, padding="valid", activation='relu'),
        # Input shape: (64, 5, 30)
        Conv2D(64, 3, padding="valid", activation='relu'),
        # Input shape: (64, 3, 28)
        Conv2D(64, 3, padding="valid", activation='relu'),
        # Input shape: (64, 1, 26)
        Flatten(),
        # Input shape: (1,1664)
        Dense(200),
        Dense(64),
        Dense(1)
        ]
        
IMAGE_SHAPE = (160,320)

def init_model(from_checkpoint=input_checkpoint):
    inp = Input(shape=IMAGE_SHAPE)
    _inp = inp
    for layer in ARCH_LAYERS:
        out = layer(_inp)
        _inp = out
    steering = out
    model = Model(inputs=inp, outputs=steering)

    opti = None
    if HP_DICT['optimizer'] == "Adam":
        opti = Adam(learning_rate=HP_DICT['start_lr'])
    else:
        print("Bad optimizer!")
        sys.exit(-1)

    if from_checkpoint:
        print("Loading weights from input checkpoint (%s) ..."%input_checkpoint)
        model.load_weights(input_checkpoint)

    # Compile the deep regression model with the chosen optimizer
    model.compile(opti, loss="mean_squared_error", metrics=['mse'])
    return model

def augmented_data_from_csv_gen(csv_lines):
    n_samples = len(csv_lines)
    cam_adjust = HP_DICT['static_cam_angle_adjust']
    while True:
        # Shuffle at the beginning of each epoch
        numpy.random.shuffle(csv_lines)

        batchsize = HP_DICT['batchsize']
        for i in range(0, n_samples, batchsize):
            lines_batch = csv_lines[i:i+batchsize]

            # Augment minibatch by extracting images referred to
            # by X_batch entries and transforming them and the 
            # associated steering angle
            images = []
            steering = []
            for line in lines_batch:
                csv_entries = line.split(',')
                paths, y = csv_entries[:-1], float(csv_entries[-1])
                for i, path in list(enumerate(paths)):
                    # Open image using RGB, save a LR-flipped copy
                    img = ndimage(path)
                    flipped_img = np.fliplr(img)
                    images.extend([img, flipped_img])
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
            # Convert to numpy arrays
            images = np.array(images)
            steering = np.array(steering)
            yield (images, steering)

def train_model(model, training_lines):
    # Split input dataset into training and validation samples
    train_samples, validation_samples = train_test_split(training_lines,
            test_size=HP_DICT['validation_split'])
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

    # Train the model using training and validation generators
    tr_history = model.fit_generator(tr_gen, steps_per_epoch=steps_per_epoch,
            epochs=HP_DICT['epochs'], verbose=1,
            validation_data=valid_gen, validation_steps=validation_steps,
            callbacks=[earlystop])
    return tr_history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="Configurable training pipeline")
    parser.add_argument('--training-files', dest="training_files",
            required=True, 
            help="Comma-separated list of CSV files comprising training data")
    parser.add_argument('--output-summary', dest="out_summary",
            required=True,
            help="Output filename for summary of training")
    parser.add_argument("--training-config", dest="training_config",
            required=True,
            help="Hyperparamter configuration file")
    parser.add_argument("--output-checkpoint", dest="output_checkpoint",
            required=True,
            help="Output checkpoint filename for trained model")
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
                    if hyptype:
                        hypval = hyptype(hypval)
                    HP_DICT[hypname] = hypval
                except Exception as e:
                    print("Invalid hyperparameter value found in training config: %s must be %s." \
                            % (hypname, str(hyptype)))
                    print(e)
    # Print out hyperparameters
    print("Using following hyperparameter values:\n%s" % HP_DICT)

    # Validate and read into memory image pathnames and steering angles
    training_csvs = [fn.strip() for fn in args.training_files.split(',')]
    training_lines = []
    for tr_fn in training_csvs:
        if not os.path.isfile(tr_fn):
            print("Cannot open training data file: %s. Quitting ..." % tr_fn)
            sys.exit(-1)
        with open(tr_fn) as tr_f:
            for line in tr_f.readlines():
                training_lines.append(line.strip()) 

    # Initialize the model
    model = init_model(from_checkpoint=args.input_checkpoint)

    # Train the model
    history = train_model(model, training_lines)

    # Save the model
    model.save(args.output_checkpoint)

    # Output the training summary
    # Print commandline
    # Hyperparameter values
    # model summary
    # Number of epochs actually trained before training stopped
    # Training and validation losses
    epochs_trained = len(history.history['loss'])
    output_summary = "Training summary:\n"
    output_summary += "cmdline: " + " ".join(sys.argv) + "\n"
    output_summary += "Hyperparameters:\n\t" \
        + "\t".join(map(lambda k: "%s=%s\n".format(k, str(HYP_DICT[k])), HYP_DICT))\
        + "\n"
    output_summary += "Total epochs trained: %d\n" % epochs_trained
    print("+"*60+"\n"+output_summary)
    model.summary()
    with open(args.out_summary, 'w') as out_f:
        out_f.write(output_summary)
        model_summary(print_fn=lambda x: out_f.write(x + "\n"))
