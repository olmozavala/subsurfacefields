from tensorflow.keras.optimizers import Adam, SGD
import tensorflow.keras.metrics as metrics
import tensorflow.keras.losses as losses
from os.path import join
import os
import tensorflow as tf
import numpy as np

from constants_proj.AI_proj_params import *
from constants.AI_params import TrainingParams, ModelParams, AiModels
from img_viz.constants import PlotMode

# ----------------------------- UM -----------------------------------
_preproc_folder = "/data/SubsurfaceFields/PreprocGoM"
_output_folder = "/data/SubsurfaceFields/Output"  # Where to save everything

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Decide which GPU to use to execute the code
# tf.config.experimental.VirtualDeviceConfiguration(memory_limit=12288)
tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2000)

NORMALIZE = False
RAND_LOC = 3 # How many random locations to use
DEPTH_SIZE = 78
# How big is the hidden layers are limitted by around ~1170 for the GPU
HID_LAY_SIZE = int(DEPTH_SIZE*min(RAND_LOC,15))
HID_LAYERS = 3  # Number of hidden layers
YEARS = 43

np.random.seed(SEED)  # THIS IS VERY IMPORTANT BECAUSE WE NEED IT SO THAT THE NETWORKS ARE TRAINED AND TESTED WITH THE SAME LOCATIONS
_run_name = F"GoMLoc_{RAND_LOC}_hidcells_{HID_LAY_SIZE}_hidlay_{HID_LAYERS}_NORM_{str(NORMALIZE)}_SEED_{str(SEED)}"

if RAND_LOC == MAX_LOCATION: # Here we select the locations we want to use
    LOCATIONS = range(RAND_LOC)
else:
    LOCATIONS = np.random.randint(0, MAX_LOCATION, RAND_LOC)

def get_preproc_config():
    model_config = {
        PreprocParams.input_folder_raw: "/data/COAPS_nexsan/people/xbxu/hycom/GLBb0.08/profile",
        # PreprocParams.input_folder_raw: "/data/SubsurfaceFields/Input",
        PreprocParams.imgs_output_folder: join(_preproc_folder, "imgs"),
        PreprocParams.output_folder: _preproc_folder,
        ProjTrainingParams.locations: MAX_LOCATION,
        ProjTrainingParams.tot_depths: DEPTH_SIZE,
        # ProjTrainingParams.bbox: [18.09165, 31.9267, -98, -76.40002]  # Min max lat and min max lon
        ProjTrainingParams.bbox: [18.09165, 31.9267, 360-98, 360-76.40002]  # Min max lat and min max lon
        # ProjTrainingParams.bbox: [24.0, 35.0, 360-102, 360-94]  # Min max lat and min max lon
    }
    return model_config

def append_model_params(cur_config):
    output_size = RAND_LOC*DEPTH_SIZE*2  # We want to output all the profiles depths for temperature and salinity
    hid_lay_size = HID_LAY_SIZE
    hid_layers = HID_LAYERS
    model_config = {
        ModelParams.MODEL: AiModels.ML_PERCEPTRON,
        ModelParams.DROPOUT: False,
        ModelParams.BATCH_NORMALIZATION: True,
        # One value for each location for SSH and SST. Also one for the day of the year
        ModelParams.INPUT_SIZE: RAND_LOC*2 + 1,
        ModelParams.HIDDEN_LAYERS: hid_layers,
        ModelParams.CELLS_PER_HIDDEN_LAYER: [hid_lay_size for x in range(hid_layers) ], # All depth levels for T and S
        ModelParams.NUMBER_OF_OUTPUT_CLASSES: output_size, # All depth levels for T and S
        ModelParams.ACTIVATION_HIDDEN_LAYERS: 'relu',
        ModelParams.ACTIVATION_OUTPUT_LAYERS: None,
    }
    return {**cur_config, **model_config}

def get_training_2d():
    cur_config = {
        TrainingParams.output_folder: F"{join(_output_folder,'Training')}",
        TrainingParams.validation_percentage: .10,
        TrainingParams.test_percentage: .10,
        TrainingParams.file_name: "RESULTS.csv",

        TrainingParams.evaluation_metrics: [metrics.mse],  # Metrics to show in tensor flow in the training
        TrainingParams.loss_function: metrics.mse,  # (OVERRIDED IN Training file) Loss function to use for the learning

        TrainingParams.optimizer: Adam(lr=0.001),  # Default values lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None,
        # TrainingParams.optimizer: SGD(),  # Default values lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None,
        TrainingParams.batch_size: 20,
        TrainingParams.epochs: 5000,
        TrainingParams.config_name: _run_name,
        TrainingParams.data_augmentation: True,
        ProjTrainingParams.normalize: NORMALIZE,
        ProjTrainingParams.tot_depths: DEPTH_SIZE,
        ProjTrainingParams.input_folder_preproc: _preproc_folder,
        ProjTrainingParams.output_folder: join(_output_folder, "images"),
        ProjTrainingParams.output_folder_summary_models:  F"{join(_output_folder,'SUMMARY')}",
        ProjTrainingParams.locations: LOCATIONS,
        ProjTrainingParams.years: YEARS,
        ProjTrainingParams.stats_file: join(_preproc_folder, "MEAN_STD_by_loc.csv")
    }
    return append_model_params(cur_config)


def get_prediction_params():
    weights_folder = join(_output_folder,"Training", _run_name, "models")
    cur_config = {
        TrainingParams.config_name: _run_name,
        PredictionParams.input_folder: _preproc_folder,
        PredictionParams.output_folder: F"{join(_output_folder,'Prediction')}",
        PredictionParams.output_imgs_folder: F"{join(_output_folder,'Prediction','imgs')}",
        PredictionParams.show_imgs: False,
        PredictionParams.model_weights_file: join(weights_folder, "Simple_CNNVeryLarge_Input_All_with_Obs_No_SSH_NO_LATLON_Output_ALL_80x80_UpSampling_NoLand_Mean_Var_2020_10_29_17_10-01-0.45879732.hdf5"),
        PredictionParams.metrics: metrics.mse,
    }

    return {**append_model_params(cur_config), **get_training_2d()}


