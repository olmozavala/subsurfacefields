# External
from tensorflow.keras.optimizers import Adam, SGD
import tensorflow.keras.metrics as metrics
import tensorflow.keras.losses as losses
from os.path import join
import os
import tensorflow as tf
import numpy as np
# Common
from ai_common.constants.AI_params import TrainingParams, ModelParams, AiModels
# This project
from constants_proj.AI_proj_params import ProjTrainingParams, PredictionParams, PreprocParams

# ----------------------------- UM -----------------------------------
_preproc_folder = "/data/SubsurfaceFields/PreprocGoM3D"
_output_folder = "/data/SubsurfaceFields/Output"  # Where to save everything

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Decide which GPU to use to execute the code
# tf.config.experimental.VirtualDeviceConfiguration(memory_limit=12288)
tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2000)

NORMALIZE = False
DEPTH_SIZE = 78
# How big is the hidden layers are limitted by around ~1170 for the GPU
YEARS = 43

SEED = 0
np.random.seed(SEED)  # THIS IS VERY IMPORTANT BECAUSE WE NEED IT SO THAT THE NETWORKS ARE TRAINED AND TESTED WITH THE SAME LOCATIONS
_run_name = F"GoM3DLoc_NORM_{str(NORMALIZE)}_SEED_{str(SEED)}"


def get_preproc_config():
    model_config = {
        PreprocParams.input_folder_raw: "/nexsan/people/xbxu/hycom/GLBb0.08/profile/3d/",
        # PreprocParams.input_folder_raw: "/data/SubsurfaceFields/Input",
        PreprocParams.imgs_output_folder: join(_preproc_folder, "imgs"),
        PreprocParams.output_folder: _preproc_folder,
        ProjTrainingParams.bbox: [17.0, 32.5, 360.0-98.0, 360.0-74.5]  # Min max lat and min max lon
    }
    return model_config

def append_model_params(cur_config):
    model_config = {
        ModelParams.MODEL: AiModels.MULTISTREAM_CNN_RNN,
        ModelParams.DROPOUT: False,
        ModelParams.BATCH_NORMALIZATION: True,
        ModelParams.START_NUM_FILTERS: 8,
        ModelParams.FILTER_SIZE: 3,
        # One value for each location for SSH and SST. Also one for the day of the year
        ModelParams.INPUT_SIZE: [27,44,78],
        ModelParams.OUTPUT_SIZE: [27,44,78],
        ModelParams.ACTIVATION_HIDDEN_LAYERS: 'relu',
        ModelParams.ACTIVATION_OUTPUT_LAYERS: None,
        ModelParams.NUMBER_LEVELS: 3,
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
        ProjTrainingParams.normalize: NORMALIZE,
        ProjTrainingParams.input_folder_preproc: _preproc_folder,
        ProjTrainingParams.output_folder: join(_output_folder, "images"),
        ProjTrainingParams.output_folder_summary_models:  F"{join(_output_folder,'SUMMARY')}",
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


