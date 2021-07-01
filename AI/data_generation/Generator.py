import numpy as np
from constants.AI_params import TrainingParams
from constants_proj.AI_proj_params import ProjTrainingParams
from ExtraUtils.VizUtilsProj import draw_profile
from os.path import join


def data_gen_from_preproc(config, ssh, temp_profile, saln_profile, depths, years, dyear, ids):
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
                try:
                    tx = np.concatenate((temp_profile[c_id,:,0].flatten(), ssh[c_id, :].flatten(), [np.sin(dyear[c_id]*np.pi/365)]))
                    # tx = np.concatenate((ssh[c_id, :].flatten(), temp_profile[c_id,:,0].flatten(), [0]))
                    ty = np.concatenate((temp_profile[c_id,:,:].flatten(), saln_profile[c_id,:,:].flatten()))
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

