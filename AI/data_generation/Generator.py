import numpy as np
from constants.AI_params import TrainingParams
from constants_proj.AI_proj_params import ProjTrainingParams
from ExtraUtils.VizUtilsProj import draw_profile
from os.path import join


def data_gen_from_preproc(config, ssh, temp_profile, saln_profile, dyear, ids):
    """
    This generator should generate X and Y for a CNN
    :param path:
    :param file_names:
    :return:
    """
    ex_id = -1
    np.random.shuffle(ids)
    batch_size = config[TrainingParams.batch_size]
    tot_ids = len(ids)
    # Make all the nans to 0
    np.nan_to_num(temp_profile, copy=False, nan=0.0)
    np.nan_to_num(saln_profile, copy=False, nan=0.0)

    mean_dyear = np.array([24.46995,24.19154,24.02029,23.85625,23.82423,23.79793,23.85859,24.05513,24.32891,24.60809,24.99460,25.53948,26.12239,26.70698,27.36962,27.83087,28.27534,28.64029,28.94974,29.15493,29.30180,29.41270,29.47937,29.45516,29.35265,29.13019,28.83846,28.40806,27.94954,27.49181,27.01199,26.51777,26.10526,25.69155,25.31612,24.90690])
    mean_dyear_dt = np.zeros(len(mean_dyear))
    mean_dyear_dt[1:] = mean_dyear[1:] - mean_dyear[:-1]
    mean_dyear_dt[0] = mean_dyear[0] - mean_dyear[-1]
    print(mean_dyear_dt)

    while True:
        try:
            succ_attempts = 0
            X = []
            Y = []
            while succ_attempts < batch_size:
                if ex_id < (tot_ids - 1): # We are not supporting batch processing right now
                    ex_id += 1
                else:
                    ex_id = 0
                    np.random.shuffle(ids) # We shuffle the folders every time we have tested all the examples

                c_id = ids[ex_id]
                dyear_idx = int(dyear[c_id]/10 - 1) # This index should go from 0 to 35
                try:
                    # tx = np.concatenate((temp_profile[c_id,:,0].flatten(), ssh[c_id, :].flatten(), [mean_dyear[dyear_idx]], [mean_dyear_dt[dyear_idx]]))
                    # tx = np.concatenate((temp_profile[c_id,:,0].flatten(), ssh[c_id, :].flatten(), [np.sin(dyear[c_id]*np.pi/365)]))
                    # tx = np.concatenate((temp_profile[c_id,:,0].flatten(), ssh[c_id, :].flatten(), [dyear[c_id]]))
                    # tx = np.concatenate((temp_profile[c_id,:,0].flatten(), ssh[c_id, :].flatten(), [0]))
                    tx = np.concatenate((temp_profile[c_id,:,0].flatten(), ssh[c_id, :].flatten(), [mean_dyear_dt[dyear_idx]]))
                    # tx = np.concatenate((temp_profile[c_id,:,0].flatten(), temp_profile[c_id,:,0].flatten(), [dyear[c_id]]))

                    ty = np.concatenate((temp_profile[c_id,:,:].flatten(), saln_profile[c_id, :, :].flatten()))
                    # Just for debugging
                    # t = temp_profile[c_id,:,:][0]
                    # t[t==0] = np.nan
                    # s = saln_profile[c_id,:,:][0]
                    # s[s==0] = np.nan
                    # draw_profile(t, s, depths[0], F"SSH:{ssh[c_id, :]} id:{c_id} ", join(config[ProjTrainingParams.input_folder_preproc], "imgs",F"gen_{c_id}.png"))

                    X.append(tx)
                    Y.append(ty) # The data should be appended like this: temp_0, saln_0, temp_1, saln_1
                except Exception as e:
                    print(F"Failed {e}")
                    continue

                succ_attempts += 1

            X = np.array(X)
            Y = np.array(Y)
            # Only for debugging
            # Y_res = np.resize(Y,(2,78))
            # print(F"Temp {Y_res[0,0:3]}")
            # print(F"Salinity {Y_res[1,0:3]}")

            yield [X], [Y]
        except Exception as e:
            print(F"----- Not able to generate for file number (from batch):  {succ_attempts} ERROR: ", str(e))

