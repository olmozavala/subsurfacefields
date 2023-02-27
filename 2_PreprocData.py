import sys
sys.path.append("ai_common/")
sys.path.append("eoas_pyutils/")

from multiprocessing import Pool
from constants_proj.AI_proj_params import PreprocParams, ProjTrainingParams
import matplotlib.pyplot as plt
from config.MainConfig import get_preproc_config
from io_utils.io_common import create_folder

from os.path import join
import skimage.io as io
from PIL import Image
import xarray as xr
import numpy as np
import pandas as pd
from metrics_proj.isop_metrics import swstate
import os
from ExtraUtils.VizUtilsProj import draw_profile

##

def PreprocDataBBOXSubsample(proc_id):
    """
    From the BIG profiles files, it generates individual files for locations surrounding a BBOX
    :param proc_id:
    :return:
    """
    np.random.seed(0)

    config = get_preproc_config()
    input_folder = config[PreprocParams.input_folder_raw]
    output_folder = config[PreprocParams.output_folder]
    # n_random_loc = config[ProjTrainingParams.locations]
    bbox = config[ProjTrainingParams.bbox]
    n_depths = config[ProjTrainingParams.tot_depths]

    all_files = os.listdir(input_folder)
    # Each file has the data divided by profiles. For example the 5deg ones have ~5 million locations,
    # for each location they have ssh, year, month, day, yday, depth, and temp(num_profiles, level) and saln(num_profiles, levels)
    five_deg_files = [x for x in all_files if "05deg.nc" in x]
    # five_deg_files = [x for x in all_files if "05deg" not in x]
    five_deg_files.sort()

    # ==================== READ DATA SEELCTS THE LAT AND LON INDEXES ============================
    print(F"Reading data and selecting locations with BBOX: {bbox} ......")
    ds = xr.open_dataset(join(input_folder, five_deg_files[0]))
    indexes = np.logical_and(np.logical_and(ds.lat_nn >= bbox[0], ds.lat_nn <= bbox[1]),
                              np.logical_and(ds.lon_nn >= bbox[2], ds.lon_nn <= bbox[3]))

    all_lats = ds.lat_nn[indexes].values
    all_lons = ds.lon_nn[indexes].values

    tot_loc = int(len(all_lats)/36)
    print(F"Total number of profiles to read: {tot_loc}")
    lats = all_lats[0:tot_loc]
    lons = all_lons[0:tot_loc]

    # print(F"Selected loc: \n {lats} \n {lons}")
    generalPreproc(lats, lons, tot_loc, n_depths, five_deg_files, input_folder, output_folder)


def PreprocDataRandomSubsample(proc_id):
    """
    From the BIG profiles files, it generates individual files for locations surrounding a BBOX
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
    # Each file has the data divided by profiles. For example the 5deg ones have ~5 million locations,
    # for each location they have ssh, year, month, day, yday, depth, and temp(num_profiles, level) and saln(num_profiles, levels)
    five_deg_files = [x for x in all_files if "05deg" in x]
    # five_deg_files = [x for x in all_files if "05deg" not in x]
    five_deg_files.sort()

    # ==================== READ DATA AND MAKE A RANDOM SUBSAMPLING ============================
    print("Reading data and selecting locations....")
    ds = xr.open_dataset(join(input_folder, five_deg_files[0]))
    all_lats = ds['lat_nn']
    all_lons = ds['lon_nn']
    tot_loc = len(all_lats)

    print(F"Total number of profiles: {tot_loc}")
    # all_loc = np.arange(tot_loc)  # Adding some extra
    all_loc = np.arange(int(tot_loc/36))  # The locations are repeated every
    lats = []
    lons = []

    selected_loc = 0
    iloc = 0
    np.random.shuffle(all_loc)  # Here is where we shuffle all the locations
    # Select random lats and lons. To be searched in all the files. Remember that each location
    # is repeated several times in each files
    test = []
    print(F"Selecting {n_random_loc}+50 random locations.....")
    while selected_loc < (n_random_loc+50): # Adding 50 extra locations just in case
        # The problem here is that the coordinates are repeated by the different days of the year. So we need
        # to verify that the selected random location is not alrady selected
        if (all_lats[all_loc[iloc]] not in lats) and (all_lons[all_loc[iloc]] not in lons):
            lats.append(all_lats[all_loc[iloc]].item())
            lons.append(all_lons[all_loc[iloc]].item())
            test.append(F"{all_lats[all_loc[iloc]].item()},{all_lons[all_loc[iloc]].item()}")
            selected_loc += 1
            print(F"{selected_loc}/{n_random_loc}")
        iloc +=1
    print(F"Unique locations= {len(np.unique(test))}")

    # print(F"Selected loc: \n {lats} \n {lons}")
    generalPreproc(lats, lons, n_random_loc, n_depths, five_deg_files, input_folder, output_folder)


def generalPreproc(lats, lons, tot_locs, n_depths, file_names, input_folder, output_folder):
    """
    From the selected lats and lots it subsamples the files and generates individual files for each location
    :param lats:
    :param lons:
    :param tot_locs:
    :param n_depths:
    :param file_names:
    :param input_folder:
    :param output_folder:
    :return:
    """
    # ==================== FROM THE SELECTED RANDOM LOCATIONS CREATE THE NEW FILES ====================
    # Iterate in each year file and look for the selected locations
    create_folder(output_folder)

    years = len(file_names)
    tot_time_steps = years*36

    # These are the final arrays of the netcdf
    id_locs = np.arange(tot_locs)
    time_steps = np.arange(0, tot_time_steps*10, 10)
    t = np.zeros((tot_locs, tot_time_steps, n_depths))  # id_loc, day_after_1963, depth
    s = np.zeros((tot_locs, tot_time_steps, n_depths))  # id_loc, day_after_1963, depth
    ssh = np.zeros((tot_locs, tot_time_steps))  # id_loc, day_after_1963
    years = np.zeros((tot_time_steps))
    dyear = np.zeros((tot_time_steps))
    start_year = 1963


    for i, c_file in enumerate(file_names):
    # for i, c_file in enumerate(file_names[0:1]):
        print(F"--------------------- Preprocessing file {c_file} year {start_year+i}")
        ds = xr.open_dataset(join(input_folder, c_file))

        selected_loc = 0
        c_loc = 0
        # This file which dates indices correspond
        from_time = 36*i
        to_time = 36*i + 36
        if i == 0:
            depths = ds.depth.values

        years[from_time:to_time] = start_year+i
        dyear[from_time:to_time] = np.arange(10,365,10)

        # Iterate over all the selected lats and lons and save them
        while selected_loc < tot_locs:
        # while selected_loc < 2:
            if selected_loc % 30 == 0:
                print(F"{selected_loc}....")
            ids = np.array(np.where(ds['lat_nn'].isin(lats[c_loc]) & ds['lon_nn'].isin(lons[c_loc])))[0]
            if len(ids) > 36:
                c_loc += 1
                print("ERROR it found more coordinates that expected!")
                continue

            # print(F"{ids} for {c_file} loc: {c_loc}")
            # Here we subsample the dataset with the selected ids (subsampled locations)
            subds = ds.sel(num_profs=ids.squeeze())

            t[c_loc, from_time:to_time, :] = subds.temp_level.values
            s[c_loc, from_time:to_time, :] = subds.saln_level.values
            ssh[c_loc, from_time:to_time] = subds.ssh.values

            selected_loc += 1
            c_loc += 1

        ds.close() # Closes current file

    temp = xr.DataArray(t, dims=['id', 'time', 'depth'])
    saln = xr.DataArray(s, dims=['id', 'time', 'depth'])
    sshout = xr.DataArray(ssh, dims=['id', 'time'])
    da_years = xr.DataArray(years, dims=['time'])
    da_dyear= xr.DataArray(dyear, dims=['time'])
    # Coordinates
    times = time_steps

    dsout = xr.Dataset(
        {
            "t": (('id', 'time', 'depth'), temp),
            "s": (('id', 'time', 'depth'), saln),
            "ssh": (('id', 'time'), sshout),
            "year": (('time'), da_years),
            "dyear": (('time'), da_dyear),
        },
        {"time": times, "lat": lats, "lon": lons, "id": id_locs, "depth":depths}
    )

    output_file = join(output_folder, "all_data.nc")
    dsout.to_netcdf(output_file)
    dsout.close()


def computeMeanSTD():
    """
    It computes the mean and STD for all the locations stored in the specified folder for Temperature, Salinity, Density
    :param input_folder:
    :return:
    """
    config = get_preproc_config()
    input_folder = config[PreprocParams.output_folder]
    n_depths = config[ProjTrainingParams.tot_depths]
    # Computing STD
    file_name = "all_data.nc"

    print("Reading data...")
    ds = xr.open_dataset(join(input_folder, file_name))
    lats = ds.lat.values
    lons = ds.lon.values
    tot_locs = len(lats)

    # They have coordinates id_profile, timesteps, depths
    temp = ds.t.values
    saln = ds.s.values
    depths = ds.depth.values

    print("Computing density...")
    _, sigma = swstate(saln, temp, depths)

    print("Saving results ...")
    df = pd.DataFrame({'locations':range(tot_locs), 'lats': lats, 'lons':lons,
                       'mean_saln':  [x for x in np.mean(saln, axis=1)],
                       'mean_temp':  [x for x in np.mean(temp, axis=1)],
                       'mean_sigma': [x for x in np.mean(sigma, axis=1)],
                       'std_saln':   [x for x in np.std(saln, axis=1)],
                       'std_temp':   [x for x in np.std(temp, axis=1)],
                       'std_sigma':  [x for x in np.std(sigma, axis=1)]})
    df.to_csv(join(input_folder, "MEAN_STD_by_loc.csv"))
    print("Done!")

if __name__ == '__main__':
    NUM_PROC = 1
    # ----------- Parallel -------
    # p = Pool(NUM_PROC)
    # p.map(PreprocDataRandomSubsample(), range(NUM_PROC))
    # PreprocDataRandomSubsample(0)
    # Generates a single training file called all_data.nc
    PreprocDataBBOXSubsample(0)
    # computeMeanSTD()