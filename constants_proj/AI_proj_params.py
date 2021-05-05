from enum import Enum

class PreprocParams(Enum):
    input_folder_raw = 3  # Input folder where the observations output is
    output_folder = 4  # Where to output the data
    imgs_output_folder = 40  # Where to output the imgs
    zero_one = 13
    mean_var = 14


class ParallelParams(Enum):
    NUM_PROC = 1

class ProjTrainingParams(Enum):
    input_folder_preproc = 1
    output_folder = 2  # Where to output the data
    fields_names = 3  # Array with the names of the fields to be analyzed
    output_fields = 4  # String containing the name of the output field
    output_folder_summary_models = 5  # Where to output the data
    locations = 6
    tot_depths = 7
    stats_file = 8
    normalize = 9

class PredictionParams(Enum):
    input_folder = 1  # Where the images are stored
    output_folder = 2  # Where to store the segmented contours
    output_imgs_folder = 3  # Where to store intermediate images
    output_file_name = 4  # Name of the file with the final statistics
    show_imgs = 5  # If we want to display the images while are being generated (for PyCharm)
    model_weights_file = 8  # Which model weights file are we going to use
    # Indicates that we need to resample everything to the original resolution. If that is the case
    metrics = 10
    compute_metrics = 12  # This means we have the GT ctrs
    save_imgs = 16  # Indicates if we want to save images from the segmented contours

