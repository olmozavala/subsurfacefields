import tensorflow as tf
from tensorflow.keras import backend as K
K.set_image_data_format('channels_last')
from constants.AI_params import TrainingParams
from io_project.read_utils import stringToArray
from constants_proj.AI_proj_params import *
from metrics_proj.isop_metrics import swstate_tf
import pandas as pd
import numpy as np

from constants_proj.AI_proj_params import *
# def proj_mse(y_true, y_pred):
#     # Flatten all the arrays
#     y_true_f = K.flatten(y_true)
#     y_pred_f = K.flatten(y_pred)
#     return tf.math.squared_difference(y_true_f, y_pred_f)

def force_monotonic_density_loss(config):

    LOCATIONS = config[ProjTrainingParams.locations]
    depth_size = 78
    stats_file = config[ProjTrainingParams.stats_file]
    batch_size = config[TrainingParams.batch_size]
    tot_loc = len(LOCATIONS)

    # =================================== READING DENORM DATA =================================================
    # Read the statistics file
    df = pd.read_csv(stats_file)
    all_mean_temp = np.zeros((len(LOCATIONS), depth_size), dtype=np.float32)
    all_mean_saln = np.zeros((len(LOCATIONS), depth_size), dtype=np.float32)
    all_std_temp = np.zeros((len(LOCATIONS), depth_size), dtype=np.float32)
    all_std_saln = np.zeros((len(LOCATIONS), depth_size), dtype=np.float32)
    max_depth_idx = np.zeros(len(LOCATIONS), dtype=np.int)

    for i_loc, c_loc in enumerate(LOCATIONS):
        all_mean_temp[i_loc,:] = stringToArray(df.loc[c_loc, "mean_temp"])
        all_mean_saln[i_loc,:] = stringToArray(df.loc[c_loc, "mean_saln"])
        all_std_temp[i_loc,:] = stringToArray(df.loc[c_loc, "std_temp"])
        all_std_saln[i_loc,:] = stringToArray(df.loc[c_loc, "std_saln"])
        max_depth_idx[i_loc] = np.where(np.isnan(all_mean_temp[i_loc,:]))[0][0] - 1

    all_mean_temp_tf = tf.constant(all_mean_temp, dtype=tf.float32)
    all_mean_saln_tf = tf.constant(all_mean_saln, dtype=tf.float32)
    all_std_temp_tf = tf.constant(all_std_temp, dtype=tf.float32)
    all_std_saln_tf = tf.constant(all_std_saln, dtype=tf.float32)
    max_depth_idx_tf = tf.constant(max_depth_idx, dtype=tf.int32)

    # Generate a tensor with the all_depths
    all_depths = tf.constant([0.0,2.0,4.0,6.0,8.0,10.0,15.0,20.0,25.0,30.0,35.0,40.0,45.0,50.0,55.0,60.0,65.0,70.0,75.0,80.0,85.0,90.0,95.0,100.0,110.0,120.0,130.0,140.0,150.0,160.0,170.0,180.0,190.0,200.0,220.0,240.0,260.0,280.0,300.0,350.0,400.0,500.0,600.0,700.0,800.0,900.0,1000.0,1100.0,1200.0,1300.0,1400.0,1500.0,1600.0,1800.0,2000.0,2200.0,2400.0,2600.0,2800.0,3000.0,3200.0,3400.0,3600.0,3800.0,4000.0,4200.0,4400.0,4600.0,4800.0,5000.0,5200.0,5400.0,5600.0,5800.0,6000.0,6200.0,6400.0,6600.0])
    # =================================== Computing the loss function =================================================

    # @tf.function
    def loss(y_true, y_pred):
        # Here we should start the loss function
        # tf.print(F"========================{y_pred.shape}============================")
        # tf.print(F"Type of output: {y_pred.dtype}")
        # tf.print(F"Number of LOCATIONS: {tot_loc}")
        # tf.print(F"y_pred Running on GPU: {y_pred.device.endswith('GPU:0')}")
        y_pred_res = tf.reshape(y_pred, [batch_size, 2, tot_loc, 78])
        y_true_res = tf.reshape(y_true, [batch_size, 2, tot_loc, 78])

        # # Denormalize the data
        for c_batch in range(batch_size):
            # tf.print(F"------------------------{c_batch}----------------------------")
            for i in range(tot_loc):
                c_max_depth = max_depth_idx_tf[i]
                t = (y_pred_res[c_batch, 0, i, :c_max_depth]*all_std_temp_tf[i,:c_max_depth]) + all_mean_temp_tf[i,:c_max_depth]
                s = (y_pred_res[c_batch, 1, i, :c_max_depth]*all_std_saln_tf[i,:c_max_depth]) + all_mean_saln_tf[i,:c_max_depth]
                # ============ COMPUTE DENSITY HERE
                d = swstate_tf(s, t, all_depths[:c_max_depth])
                diff = d[:-2] - d[1:-1]
                if i == 0 and c_batch == 0:
                    error_mon = tf.add(0.0, tf.reduce_sum(diff[diff > 0])/tot_loc)
                else:
                    error_mon = error_mon + tf.reduce_sum(diff[diff > 0])/tot_loc

                # true_t = (y_true_res[c_batch, 0, i, :c_max_depth]*all_std_temp_tf[i,:c_max_depth]) + all_mean_temp_tf[i,:c_max_depth]
                # true_s = (y_true_res[c_batch, 1, i, :c_max_depth]*all_std_saln_tf[i,:c_max_depth]) + all_mean_saln_tf[i,:c_max_depth]
                # true_d = swstate_tf(true_s, true_t, all_depths[:c_max_depth])
                # for j in range(max_depth_idx[i] - 2): # The -2 is because we are also printing the difference and there we lost one index
                #     tf.print(F"%%%%%%%%%%%%% {j} %%%%%%%%%%%%%%%%%%%%%%")
                #     tf.print(y_true_res[c_batch,0,i,j])
                #     tf.print(y_true_res[c_batch,1,i,j])
                #     tf.print(y_pred_res[c_batch,0,i,j])
                #     tf.print(y_pred_res[c_batch,1,i,j])
                #     tf.print(F"************* {j} **********************")
                #     tf.print(true_t[j])
                #     tf.print(true_s[j])
                #     tf.print(t[j])
                #     tf.print(s[j])
                #     tf.print(F"------------- {j} ----------------------")
                #     tf.print(true_d[j])
                #     tf.print(d[j])
                #     tf.print(F"&&&&&&&&&&&&")
                #     tf.print(diff[j])

        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        rmse = tf.math.reduce_mean(tf.math.squared_difference(y_true_f, y_pred_f))
        tf.print(F"RMSE:")
        tf.print(rmse)
        tf.print(F"ERROR:")
        tf.print(error_mon)
        return rmse + error_mon
        # return rmse

    return loss

def only_ocean_mse(y_true, y_pred, smooth=1.0):
    eps = .00001
    # Flatten all the arrays
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    # The true values on land should be -0.5 in order for this to work
    temp = tf.math.ceil(y_true_f + eps)  # This should make the ocean values = 1 and the land values = 0
    y_pred_c = y_pred_f * temp # Make 0 the prediction outside the prostate
    # Keep only the water values
    y_true_c = y_true_f * temp

    return tf.math.squared_difference(y_true_c, y_pred_c)

# ========== Just for testing ============
if __name__ == '__main__':

    from config.MainConfig import get_training_2d
    config = get_training_2d()
    LOCATIONS = config[ProjTrainingParams.locations]
    batch_size = config[TrainingParams.batch_size]
    RAND_LOC = len(LOCATIONS)
    depth_size = 78
    output_size = RAND_LOC*depth_size*2  # We want to output all the profiles all_depths for temperature and salinity

    y_true = tf.random.uniform([batch_size, output_size])
    y_pred = tf.random.uniform([batch_size, output_size])

    # TODO !!!!!!!!!!!!!!!!!!!!!
    # TODO Parece ser que el resample no funciona bien porque aparecen nans en el mismo lugar

    myloss = force_monotonic_density_loss(config)
    lval = myloss(y_true, y_pred)
    print(F"Test loss value: {lval}")