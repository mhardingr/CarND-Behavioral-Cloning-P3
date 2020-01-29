import argparse
import os.path
import sys

HP_DICT = {
        'epochs': -1,
        'start_lr': -1.0,
        'optimizer': None,
        'static_cam_angle_adjust': -1.0,
        'drop_prob': -1.0
        }

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
            required=False,
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
    training_images = []
    training_steering = []  # Keep training data mapped properly
    for tr_fn in training_csvs:
        if not os.path.isfile(tr_fn):
            print("Cannot open training data file: %s. Quitting ..." % tr_fn)
            sys.exit(-1)
        # Read into memory training data (TODO: apply steering adjustment)
        # TODO: Need to create new images that flip
        with open(tr_fn) as tr_f:
            for line in tr_f.readlines():
                entries = line.strip().split(',')
                
