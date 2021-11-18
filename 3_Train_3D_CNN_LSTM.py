# External
from os.path import join
import numpy as np
import pandas as pd
import os
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.utils import plot_model
# Common
import ai_common.training.trainingutils as utilsNN
from ai_common.models.modelSelector import select_2d_model
from ai_common.constants.AI_params import TrainingParams, ModelParams
from img_viz.common import create_folder
# This project
from config.MainConfig3D import get_training_2d
from AI.data_generation.Generator import data_gen_3d_from_preproc
from constants_proj.AI_proj_params import ProjTrainingParams, MAX_LOCATION, SEED
from models_proj.models import *
from io_project.read_utils import get_all_profiles, normDenormData
from metrics_proj.isop_metrics import swstate
from ExtraUtils.VizUtilsProj import draw_profile

def doTraining(conf):
    input_folder_preproc = config[ProjTrainingParams.input_folder_preproc]

    output_folder = config[TrainingParams.output_folder]
    eval_metrics = config[TrainingParams.evaluation_metrics]
    loss_func = config[TrainingParams.loss_function]
    batch_size = config[TrainingParams.batch_size]
    epochs = config[TrainingParams.epochs]
    run_name = config[TrainingParams.config_name]
    optimizer = config[TrainingParams.optimizer]
    val_perc = config[TrainingParams.validation_percentage]
    test_perc = config[TrainingParams.test_percentage]

    print("Selecting and generating the model....")
    now = datetime.utcnow().strftime("%Y_%m_%d_%H_%M")
    model_name = F'{run_name}_{now}'

    output_folder = join(output_folder, model_name)
    split_info_folder = join(output_folder, 'Splits')
    parameters_folder = join(output_folder, 'Parameters')
    weights_folder = join(output_folder, 'models')
    logs_folder = join(output_folder, 'logs')
    create_folder(split_info_folder)
    create_folder(parameters_folder)
    create_folder(weights_folder)
    create_folder(logs_folder)

    # Each array has date as its first index and location as its second index
    print("Reading all data...")
    file_names = os.listdir(input_folder_preproc)
    file_names.sort()
    file_paths = [join(input_folder_preproc, c_file) for c_file in file_names]
    total_timesteps = len(file_names)
    print("Done!")

    # ================ Split definition =================
    np.random.seed(SEED)  # THIS IS VERY IMPORTANT BECAUSE WE NEED IT SO THAT THE NETWORKS ARE TRAINED AND TESTED WITH THE SAME LOCATIONS
    [train_ids, val_ids, test_ids] = utilsNN.split_train_validation_and_test(total_timesteps,
                                                                             val_percentage=val_perc,
                                                                             test_percentage=test_perc,
                                                                             shuffle_ids=False)

    print(F"Train examples (total:{len(train_ids)}) :{train_ids}")
    print(F"Validation examples (total:{len(val_ids)}) :{val_ids}:")
    print(F"Test examples (total:{len(test_ids)}) :{test_ids}")


    # ******************* Selecting the model **********************
    model = select_2d_model(config)

    plot_model(model, to_file=join(output_folder,F'{model_name}.png'), show_shapes=True)

    print("Saving split information...")
    file_name_splits = join(split_info_folder, F'{model_name}.txt')
    utilsNN.save_splits(file_name=file_name_splits, train_ids=train_ids, val_ids=val_ids, test_ids=test_ids)


    print("Compiling model ...")
    model.run_eagerly = False
    model.compile(loss=loss_func, optimizer=optimizer, metrics=eval_metrics)

    print("Getting callbacks ...")

    [logger, save_callback, stop_callback] = utilsNN.get_all_callbacks(model_name=model_name,
                                                                       early_stopping_func=F'val_{eval_metrics[0].__name__}',
                                                                       weights_folder=weights_folder,
                                                                       logs_folder=logs_folder)

    # ----------- Normalize data -------------------
    print("Normalizing data...")
    # if normalize:
        # tstep = 0
        # c_id = 0
        # draw_profile(temp_profile[tstep,c_id,:], saln_profile[tstep,c_id,:], depths[c_id], F"Before Norm SSH:{ssh[tstep, c_id]} id:{c_id} ", join(config[ProjTrainingParams.input_folder_preproc], "imgs",F"{c_id}_BN.png"))
        # temp_profile, saln_profile = normDenormData(stats_input_file, temp_profile, saln_profile, loc=locations)
        # draw_profile(temp_profile[tstep,c_id,:], saln_profile[tstep,c_id,:], depths[c_id], F"After Norm SSH:{ssh[tstep, c_id]} id:{c_id} ", join(config[ProjTrainingParams.input_folder_preproc], "imgs",F"{c_id}_BN.png"))
    print("Done! ...")

    # ----------- Using preprocessed data -------------------
    print("Training ...")
    generator_train = data_gen_3d_from_preproc(config, file_paths, train_ids)
    generator_val = data_gen_3d_from_preproc(config,  file_paths, val_ids)

    # Decide which generator to use
    model.fit_generator(generator_train, steps_per_epoch=int(np.ceil(len(train_ids)/batch_size)),
                        validation_data=generator_val,
                        validation_steps=int(np.ceil(len(val_ids)/batch_size)),
                        use_multiprocessing=False,
                        workers=1,
                        # validation_freq=10, # How often to compute the validation loss
                        epochs=epochs, callbacks=[logger, save_callback, stop_callback])


if __name__ == '__main__':
    config = get_training_2d()
    # ======================= Single training =======================
    doTraining(config)

    # ======================= Multiple training =======================
    # # normalize = [True, False]
    # normalize = [True]
    # # rand_loc = [2, 4, 100, 200, 400, 600, 636]
    # # rand_loc = [100, 200, 400, 600, 636]
    # rand_loc = [100, 200]
    # # rand_loc = [2]
    # DEPTH_SIZE = 78
    # hid_layers = 3  # Number of hidden layers
    #
    # for RAND_LOC in rand_loc:
    #     for NORMALIZE in normalize:
    #         # How big is the hidden layers are limitted by around ~1170 for the GPU
    #         hid_lay_size = int(DEPTH_SIZE*min(RAND_LOC, 15))
    #         np.random.seed(SEED)  # THIS IS VERY IMPORTANT BECAUSE WE NEED IT SO THAT THE NETWORKS ARE TRAINED AND TESTED WITH THE SAME LOCATIONS
    #         # _run_name = F"GoMLocMeanSSTAndDerivativeForTime_{RAND_LOC:05d}_hidcells_{hid_lay_size}_hidlay_{hid_layers}_NORM_{str(NORMALIZE)}_SEED_{str(SEED)}_Adam"
    #         _run_name = F"GoMLoc_GRADIENT_{RAND_LOC:05d}_hidcells_{hid_lay_size}_hidlay_{hid_layers}_NORM_{str(NORMALIZE)}_SEED_{str(SEED)}_Adam"
    #         # _run_name = F"GoMLocDateDirectly_{RAND_LOC:05d}_hidcells_{hid_lay_size}_hidlay_{hid_layers}_NORM_{str(NORMALIZE)}_SEED_{str(SEED)}_Adam"
    #         # _run_name = F"GoMLocSineDate_{RAND_LOC:05d}_hidcells_{hid_lay_size}_hidlay_{hid_layers}_NORM_{str(NORMALIZE)}_SEED_{str(SEED)}_Adam"
    #         # _run_name = F"GoMLocNoDateInfo{RAND_LOC:05d}_hidcells_{hid_lay_size}_hidlay_{hid_layers}_NORM_{str(NORMALIZE)}_SEED_{str(SEED)}_Adam"
    #         output_size = RAND_LOC*DEPTH_SIZE*2  # We want to output all the profiles depths for temperature and salinity
    #
    #         if RAND_LOC == MAX_LOCATION: # Here we select the locations we want to use
    #             LOCATIONS = range(RAND_LOC)
    #         else:
    #             LOCATIONS = np.random.choice(range(MAX_LOCATION), RAND_LOC, replace=False)
    #
    #         config[ModelParams.INPUT_SIZE] = RAND_LOC*2 + 1
    #         config[ProjTrainingParams.locations] = LOCATIONS
    #         config[ModelParams.NUMBER_OF_OUTPUT_CLASSES] = output_size  # All depth levels for T and S
    #         config[ModelParams.CELLS_PER_HIDDEN_LAYER] = [hid_lay_size for x in range(hid_layers)]  # All depth levels for T and S
    #         config[ProjTrainingParams.normalize] = NORMALIZE
    #         config[TrainingParams.config_name] = _run_name
    #         # # ====================== Using custom loss ====================
    #         # myloss = force_monotonic_density_loss(config)
    #         # config[TrainingParams.loss_function] = myloss
    #         # config[TrainingParams.evaluation_metrics] = [myloss]  # Metrics to show in tensor flow in the training
    #         # ====================== Using custom loss ====================
    #         doTraining(config)
