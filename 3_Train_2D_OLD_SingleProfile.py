from datetime import datetime

from config.MainConfig import get_training_2d
from AI.data_generation.Generator import data_gen_from_preproc

from constants_proj.AI_proj_params import ProjTrainingParams
import trainingutils as utilsNN
from models.modelSelector import select_1d_model
from models_proj.models import *
from img_viz.common import create_folder
from io_project.read_utils import get_profiles_byloc

from os.path import join
import numpy as np
import os
from constants.AI_params import TrainingParams, ModelParams

from tensorflow.keras.utils import plot_model

def doTraining(conf):
    input_folder_preproc = config[ProjTrainingParams.input_folder_preproc]
    # output_field = config[ProjTrainingParams.output_fields]

    output_folder = config[TrainingParams.output_folder]
    eval_metrics = config[TrainingParams.evaluation_metrics]
    loss_func = config[TrainingParams.loss_function]
    batch_size = config[TrainingParams.batch_size]
    epochs = config[TrainingParams.epochs]
    run_name = config[TrainingParams.config_name]
    optimizer = config[TrainingParams.optimizer]

    output_folder = join(output_folder, run_name)
    split_info_folder = join(output_folder, 'Splits')
    parameters_folder = join(output_folder, 'Parameters')
    weights_folder = join(output_folder, 'models')
    logs_folder = join(output_folder, 'logs')
    create_folder(split_info_folder)
    create_folder(parameters_folder)
    create_folder(weights_folder)
    create_folder(logs_folder)

    # Compute how many cases
    ssh, temp_profile, saln_profile, years = get_profiles_byloc(input_folder_preproc, 0)

    # ================ Split definition =================
    train_ids = np.where((years == 2001) | (years == 2002) | (years == 2003) | (years == 2004))[0]
    val_ids = np.where(years == 2005)[0]
    test_ids = np.where(years == 2006)[0]

    print(F"Train examples (total:{len(train_ids)}) :{train_ids}")
    print(F"Validation examples (total:{len(val_ids)}) :{val_ids}:")
    print(F"Test examples (total:{len(test_ids)}) :{test_ids}")

    print("Selecting and generating the model....")
    now = datetime.utcnow().strftime("%Y_%m_%d_%H_%M")
    model_name = F'{run_name}_{now}'

    # ******************* Selecting the model **********************
    model = select_1d_model(config)

    # https://www.tensorflow.org/api_docs/python/tf/keras/applications/ResNet50
    # model = tf.keras.applications.ResNet50(include_top=True, weights='imagenet',
    #                                        input_shape=config[ModelParams.INPUT_SIZE],
    #                                        pooling=max, classes=4)

    plot_model(model, to_file=join(output_folder,F'{model_name}.png'), show_shapes=True)

    print("Saving split information...")
    file_name_splits = join(split_info_folder, F'{model_name}.txt')
    utilsNN.save_splits(file_name=file_name_splits, train_ids=train_ids, val_ids=val_ids, test_ids=test_ids)

    print("Compiling model ...")
    model.compile(loss=loss_func, optimizer=optimizer, metrics=eval_metrics)

    print("Getting callbacks ...")

    [logger, save_callback, stop_callback] = utilsNN.get_all_callbacks(model_name=model_name,
                                                                       early_stopping_func=F'val_{eval_metrics[0].__name__}',
                                                                       weights_folder=weights_folder,
                                                                       logs_folder=logs_folder)

    print("Training ...")
    # ----------- Using preprocessed data -------------------
    generator_train = data_gen_from_preproc(config, ssh, temp_profile, saln_profile, train_ids)
    generator_val = data_gen_from_preproc(config, ssh, temp_profile, saln_profile, val_ids)

    # Decide which generator to use
    model.fit_generator(generator_train, steps_per_epoch=int(np.ceil(1000/batch_size)),
                        validation_data=generator_val,
                        # validation_steps=min(100, len(val_ids)),
                        validation_steps=int(np.ceil(100/batch_size)),
                        use_multiprocessing=False,
                        workers=1,
                        # validation_freq=10, # How often to compute the validation loss
                        epochs=epochs, callbacks=[logger, save_callback, stop_callback])


if __name__ == '__main__':
    config = get_training_2d()
    # Single training
    doTraining(config)
