import os
from ExtraUtils.VizUtilsProj import draw_profile, draw_profiles_comparison
from models.modelSelector import select_1d_model
from tensorflow.keras.utils import plot_model
from os.path import join
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import trainingutils as utilsNN

from config.MainConfig import get_prediction_params
from constants_proj.AI_proj_params import PredictionParams, ProjTrainingParams, PreprocParams
from models_proj.models import *
from constants.AI_params import TrainingParams, ModelParams, AiModels
from img_viz.common import create_folder
from io_project.read_utils import get_all_profiles, normDenormData
from sklearn.metrics import mean_squared_error

from ExtraUtils.NamesManipulation import *
from ExtraUtils.VizUtilsProj import chooseCMAP
import cmocean


def main():

    config = get_prediction_params()

    # -------- For all summary model testing --------------
    summary_file = "/data/SubsurfaceFields/Output/SUMMARY/summary.csv"
    df = pd.read_csv(summary_file)
    for model_id in range(len(df)):
        model = df.iloc[model_id]

        # Setting Network type (only when network type is UNET)
        name = model["Name"]
        split_name = name.split("_")
        hid_layers = int(split_name[5])
        hid_layer_size = int(split_name[3])
        RAND_LOC = int(split_name[1])
        seed = int(split_name[9])
        MAX_LOCATION = 500   # How many locations can we test
        np.random.seed(seed)  # THIS IS VERY IMPORTANT BECAUSE WE NEED IT SO THAT THE NETWORKS ARE TRAINED AND TESTED WITH THE SAME LOCATIONS
        if RAND_LOC == MAX_LOCATION: # Here we select the locations we want to use
            locations = range(RAND_LOC)
        else:
            locations = np.random.randint(0, MAX_LOCATION, RAND_LOC)
        config[ProjTrainingParams.locations] = locations
        config[ModelParams.HIDDEN_LAYERS] = hid_layers
        config[ModelParams.CELLS_PER_HIDDEN_LAYER] = [hid_layer_size for x in range(hid_layers) ]
        config[ModelParams.INPUT_SIZE] = int(split_name[1])*2 + 1
        config[ProjTrainingParams.normalize] = split_name[7] == "True"
        config[ModelParams.NUMBER_OF_OUTPUT_CLASSES] = RAND_LOC*config[ProjTrainingParams.tot_depths]*2
        config[PredictionParams.model_weights_file] = model["Path"]
        print(F"Model's weight file: {model['Path']}")
        # Set the name of the network
        run_name = name.replace(".hdf5", "")
        config[TrainingParams.config_name] = run_name
        test_model(config)

def test_model(config):
    input_folder_preproc = config[ProjTrainingParams.input_folder_preproc]
    output_folder = config[PredictionParams.output_folder]
    model_weights_file = config[PredictionParams.model_weights_file]
    output_imgs_folder = config[PredictionParams.output_imgs_folder]
    run_name = config[TrainingParams.config_name]
    val_perc = config[TrainingParams.validation_percentage]
    test_perc = config[TrainingParams.test_percentage]
    normalize = config[ProjTrainingParams.normalize]
    stats_input_file = config[ProjTrainingParams.stats_file]
    locations = config[ProjTrainingParams.locations]
    tot_loc = len(locations)

    output_imgs_folder = join(output_imgs_folder, run_name)
    create_folder(output_imgs_folder)

    model = select_1d_model(config)
    plot_model(model, to_file=join(output_folder, F'running.png'), show_shapes=True)

    # *********** Reads the weights***********
    print('Reading weights ....')
    model.load_weights(model_weights_file)

    # *********** Read files to predict***********

    [train_ids, val_ids, test_ids] = utilsNN.split_train_validation_and_test(216,
                                                                             val_percentage=val_perc,
                                                                             test_percentage=test_perc,
                                                                             shuffle_ids=False)

    print("Reading all data...")
    # TODO be able to read just some of the timesteps not all of them
    ssh, temp_profile, saln_profile, years, dyear, depths, latlons = get_all_profiles(input_folder_preproc, locations, test_ids)
    print("Done!")

    # ----------- Normalize data -------------------
    print("Normalizing data...")
    if normalize:
        # tstep = 0
        # c_id = 0
        # draw_profile(temp_profile[tstep,c_id,:], saln_profile[tstep,c_id,:], depths[c_id], F"Before Norm SSH:{ssh[tstep, c_id]} id:{c_id} ", join(config[ProjTrainingParams.input_folder_preproc], "imgs",F"{c_id}_BN.png"))
        norm_temp_profile, norm_saln_profile = normDenormData(stats_input_file, temp_profile, saln_profile)
        # draw_profile(temp_profile[tstep,c_id,:], saln_profile[tstep,c_id,:], depths[c_id], F"After Norm SSH:{ssh[tstep, c_id]} id:{c_id} ", join(config[ProjTrainingParams.input_folder_preproc], "imgs",F"{c_id}_BN.png"))
    print("Done! ...")

    for c_date_id, c_date_i in enumerate(test_ids):
        c_date = (c_date_i*10) % 365
        if normalize:
            sst = norm_temp_profile[c_date_id,:,0].flatten()
        else:
            sst = temp_profile[c_date_id,:,0].flatten()

        # tx = np.concatenate((ssh[c_id, :].flatten(), temp_profile[c_id,:,0].flatten(), [np.cos(dyear[c_id]*np.pi/365)]))
        X = [np.concatenate((ssh[c_date_id, :].flatten(), sst, [np.cos(dyear[c_date_id]*np.pi/365)]))]
        X = np.array(X)

        # Make the prediction of the network
        start = time.time()
        output_nn_original = model.predict([X], verbose=1)
        toc = time.time() - start

        # ****************************
        # TODO compute RMS between prediction and real, plot by location in a MAP awesome!!
        # data = np.reshape(output_nn_original[0], (tot_loc, 2, 78))
        data = np.reshape(output_nn_original[0], (2, tot_loc, 78))
        # Only for debugging
        # draw_profile(temp_profile[c_date_id,0,:], saln_profile[c_date_id,0,:], depths[0,:], F"TRUE {dyear[0]}")
        # draw_profile(data[0,0,:], data[1,0,:],  depths[0,:], F"Delete  dyear {dyear[0]}")
        nn_temp_profile = np.expand_dims(data[0,:,:], axis=0)
        nn_saln_profile = np.expand_dims(data[1,:,:], axis=0)
        if normalize:
            nn_temp_profile, nn_saln_profile = normDenormData(stats_input_file, nn_temp_profile, nn_saln_profile, normalize=False)

        # Iterate over the profiles being analyzed and the output of the NN
        for i, c_profile in enumerate(locations):
            nan_ids = np.isnan(temp_profile[c_date_id,i,:])
            nn_temp_profile[0,i,nan_ids] = np.nan
            nn_saln_profile[0,i,nan_ids] = np.nan
            file_name = join(output_imgs_folder, F"{c_profile}_{c_date_id}.png")
            rmse_temp = np.nanmean(((temp_profile[c_date_id,i,:] - nn_temp_profile[0,i,:])**2))
            rmse_saln = np.nanmean(((saln_profile[c_date_id,i,:] - nn_saln_profile[0,i,:])**2))

            draw_profiles_comparison(temp_profile[c_date_id,i,:], saln_profile[c_date_id,i,:],
                                     nn_temp_profile[0,i,:], nn_saln_profile[0,i,:],
                                     depths[i,:], F"Profile {c_profile} dyear {c_date}  RMSE Temp: {rmse_temp:0.2f}, Saln: {rmse_saln:0.2f}",
                                     file_name)

        # for c_prof in range(tot_loc):
            # draw_profile()

if __name__ == '__main__':
    main()
