from multiprocessing import Pool
from constants_proj.AI_proj_params import PreprocParams, ProjTrainingParams
import matplotlib.pyplot as plt
from config.MainConfig import get_preproc_config
from inout_common.io_common import create_folder
from os.path import join
import skimage.io as io
from PIL import Image
import xarray as xr
import numpy as np
import pandas as pd
import os
from ExtraUtils.VizUtilsProj import draw_profile

NUM_PROC = 1

def main():
    # ----------- Parallel -------
    p = Pool(NUM_PROC)
    p.map(PreprocData, range(NUM_PROC))

def PreprocData(proc_id):
    """
    From the BIG profiles files, it generates individual files for random locations.
    :param proc_id:
    :return:
    """
    np.random.seed(0)

    config = get_preproc_config()
    input_folder = config[PreprocParams.input_folder_raw]
    output_folder = config[PreprocParams.output_folder]
    n_random_loc = config[ProjTrainingParams.locations]
    n_depths = config[ProjTrainingParams.tot_depths]

    all_files = os.listdir(input_folder)
    five_deg_files = [x for x in all_files if "05deg" in x]
    # five_deg_files = [x for x in all_files if "05deg" not in x]
    five_deg_files.sort()

    print("Reading data and selecting locations....")
    ds = xr.load_dataset(join(input_folder, five_deg_files[0]))
    all_lats = ds['lat_nn']
    all_lons = ds['lon_nn']
    tot_loc = len(all_lats)

    print(F"Total number of profiles: {tot_loc}")
    all_loc = np.arange(tot_loc)  # Adding some extra
    lats = []
    lons = []

    selected_loc = 0
    iloc = 0
    np.random.shuffle(all_loc)
    # Select random lats and lons. To be searched in all the files. Remember that each location
    # is repeated several times in each files
    test = []
    while selected_loc < (n_random_loc+50): # Adding 50 extra locations just in case
        if (all_lats[all_loc[iloc]] not in lats) and (all_lons[all_loc[iloc]] not in lons):
            lats.append(all_lats[all_loc[iloc]].item())
            lons.append(all_lons[all_loc[iloc]].item())
            test.append(F"{all_lats[all_loc[iloc]].item()},{all_lons[all_loc[iloc]].item()}")
            selected_loc += 1
        iloc +=1
    print(F"Unique locations= {len(np.unique(test))}")

    np.random.shuffle(all_loc)
    # print(F"Selected loc: \n {lats} \n {lons}")

    # Iterate in each year file and look for the selected locations
    loc_std_temp = np.full((n_random_loc, n_depths), np.nan)
    loc_mean_temp  = np.full((n_random_loc, n_depths), np.nan)
    loc_std_saln = np.full((n_random_loc, n_depths), np.nan)
    loc_mean_saln  = np.full((n_random_loc, n_depths), np.nan)

    tot_time_steps = -1
    f_lats = []
    f_lons = []
    for i, c_file in enumerate(five_deg_files):
        print(F"Preprocessing file {c_file}")
        ds = xr.load_dataset(join(input_folder, c_file))
        year = ds['year'][0]

        selected_loc = 0
        c_loc = 0
        while selected_loc < n_random_loc:
            ids = np.array(np.where(ds['lat_nn'].isin(lats[c_loc]) & ds['lon_nn'].isin(lons[c_loc])))[0]
            c_loc += 1
            if len(ids) > 36:
                print("ERROR it found more coordinates that expected!")
                continue

            # print(F"{ids} for {c_file} loc: {c_loc}")
            subds = ds.sel(num_profs=ids.squeeze())
            if tot_time_steps == -1:
                tot_time_steps = subds.ssh.size * len(five_deg_files)
                print(F"Number of timesteps: {tot_time_steps}")

            if (i == 0):
                f_lats.append(lats[c_loc])
                f_lons.append(lons[c_loc])
                loc_mean_saln[selected_loc, :] = np.sum(subds.saln_level.values, axis=0)/tot_time_steps
                loc_mean_temp[selected_loc, :] = np.sum(subds.temp_level.values, axis=0)/tot_time_steps
            else:
                loc_mean_saln[selected_loc, :] += np.sum(subds.saln_level.values, axis=0)/tot_time_steps
                loc_mean_temp[selected_loc, :] += np.sum(subds.temp_level.values, axis=0)/tot_time_steps
            output_file = join(output_folder,F"{int(year.values)}_loc_{selected_loc:04}.nc")
            print(F"Saving file {output_file}")
            subds.to_netcdf(output_file)
            subds.close()
            selected_loc += 1
            # print(F"Done!")
        ds.close()

    # Just to plot the mean profiles

    out_img_folder = join(output_folder, "imgs")
    create_folder(out_img_folder)
    # for j in range(n_random_loc):
    #     draw_profile(loc_mean_temp[j,:], loc_mean_saln[j,:], ds.depth, "delete", join(out_img_folder,F"{j:04}_MeanProfile.png"))

    # Computing STD
    print("================ STD ==================")
    for i, c_file in enumerate(five_deg_files):
        print(F"Preprocessing file {c_file}")
        ds = xr.load_dataset(join(input_folder, c_file))
        selected_loc = 0
        c_loc = 0
        while selected_loc < n_random_loc:
            ids = np.array(np.where(ds['lat_nn'].isin(lats[c_loc]) & ds['lon_nn'].isin(lons[c_loc])))[0]
            c_loc += 1
            if len(ids) > 36:
                print("ERROR it found more coordinates that expected!")
                continue

            subds = ds.sel(num_profs=ids.squeeze())
            if (i == 0):
                loc_std_saln[selected_loc, :] = np.sqrt(np.sum((subds.saln_level.values - loc_mean_saln[selected_loc][:])**2, axis=0)/tot_time_steps)
                loc_std_temp[selected_loc, :] = np.sqrt(np.sum((subds.temp_level.values - loc_mean_temp[selected_loc][:])**2, axis=0)/tot_time_steps)
            else:
                loc_std_saln[selected_loc, :] += np.sqrt(np.sum((subds.saln_level.values - loc_mean_saln[selected_loc][:])**2, axis=0)/tot_time_steps)
                loc_std_temp[selected_loc, :] += np.sqrt(np.sum((subds.temp_level.values - loc_mean_temp[selected_loc][:])**2, axis=0)/tot_time_steps)
            selected_loc += 1

    # for j in range(n_random_loc):
    #     draw_profile(loc_std_temp[j,:], loc_std_saln[j,:], ds.depth, "delete", join(out_img_folder,F"{j:04}_STDProfile.png"))

    df = pd.DataFrame({'locations':range(n_random_loc), 'lats': f_lats, 'lons':f_lons,
                       'mean_saln':[x for x in loc_mean_saln], 'mean_temp':[x for x in loc_mean_temp],
                       'std_saln':[x for x in loc_std_saln], 'std_temp':[x for x in loc_std_temp]})
    df.to_csv(join(output_folder, "MEAN_STD_by_loc.csv"))

if __name__ == '__main__':
    main()
