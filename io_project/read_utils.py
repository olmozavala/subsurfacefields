from preproc.UtilsDates import get_days_from_month
from constants_proj.AI_proj_params import PreprocParams
import pandas as pd
from pandas import DataFrame
import numpy.ma as ma
from os.path import join, isfile
import xarray as xr
import os
import numpy as np
from datetime import date
from ExtraUtils.VizUtilsProj import draw_profile
import re

def get_all_profiles(input_folder, all_loc, time_steps=range(216)):
    """
    This function reads all the requested locations and generates arrays separated by dates.
    :param input_folder: Where the profile files are stored (the ones splitted by locations)
    :param all_loc: An array of integers indicating the number of the locations
    :return: Each output array should contain 216 time steps
    """

    tot_loc = len(all_loc)
    tot_time_steps = len(time_steps)

    for i, c_loc in enumerate(all_loc):  # Iterates over the locations
        print(F"Reading location {c_loc}...")
        ssh, temp_profile, saln_profile, years, dyear, depths, latlon = get_profiles_byloc(input_folder, c_loc, time_steps)
        if i == 0:
            z_levels = temp_profile.shape[1]

            # We assume the dyear and years are the same for all profiles
            dyear = dyear
            all_years = years

            all_ssh = np.zeros((tot_time_steps, tot_loc))
            all_saln = np.zeros((tot_time_steps, tot_loc, z_levels))
            all_temp = np.zeros((tot_time_steps, tot_loc, z_levels))
            all_depths = np.zeros((tot_loc, z_levels))
            all_latlons = np.zeros((tot_loc, 2))

        # Appends the corresponding value into the global arrays
        all_ssh[:, i] = ssh
        all_saln[:, i, :] = saln_profile
        all_temp[:, i, :] = temp_profile
        all_depths[i, :] = depths
        all_latlons[i, :] = latlon

    # The first two indexes are the timestep and the location
    return all_ssh, all_temp, all_saln, all_years, dyear, all_depths, all_latlons


def get_profiles_byloc(input_folder, loc_prof, time_steps):
    """
    It reads the information for one specific location
    :param input_folder:
    :param loc_prof:
    :param time_steps:
    :return:
    """
    loc_files = os.listdir(input_folder)
    loc_files = [x for x in loc_files if x.find(F"{loc_prof:04}") != -1]
    loc_files.sort()

    ssh = []
    temp_profile = []
    saln_profile = []
    years = []
    dyear = []
    depth = []

    # Decide which files do we need to read
    days_per_year = 36
    min_tstep = np.amin(time_steps)
    max_tstep = np.amax(time_steps)
    years_to_read = range(int(np.floor(min_tstep/days_per_year)), int(np.ceil(max_tstep/days_per_year)))

    # Iterate for each selected year
    for i in years_to_read:
        c_file = loc_files[i]
        ds = xr.open_dataset(join(input_folder, c_file))
        if i == years_to_read[0]:  # Because the depths for this profile are always the same we store it only once
            depth = list(ds.depth)
            latlon = np.array([ds.lat_nn.values[0], ds.lon_nn.values[0]])

        ids_year = [int(np.max((0, min_tstep - i*days_per_year))), int(np.min((days_per_year, max_tstep - i*days_per_year + 1)))]
        ssh += list(ds.ssh.values[ids_year[0]:ids_year[1]])
        temp_profile += list(ds.temp_level[ids_year[0]:ids_year[1]])
        saln_profile += list(ds.saln_level[ids_year[0]:ids_year[1]])
        years += list(ds.year[ids_year[0]:ids_year[1]])
        dyear += list(ds.yday[ids_year[0]:ids_year[1]])

    return np.array(ssh), np.array(temp_profile), np.array(saln_profile), np.array(years, dtype=np.int), np.array(dyear, dtype=np.int), np.array(depth, dtype=np.int), latlon


def stringToArray(st_orig):
    # TODO this is very bad, improve it. We should save a np array somehow. How is it we cant read it?
    st_orig = re.sub("\ +", " ", st_orig)
    str_array = st_orig.replace("[ ","").replace("[","").replace("]","").replace("\n","").split(" ")
    return np.array([float(x) if x != "nan" else np.nan for x in str_array])


def normDenormData(stats_file, t, s, normalize=True, loc="all"):
    """
    Normalizes and denormalizes the temperature and salinity profiles
    :param stats_file: this file is normally in the Preproc folder and is called MEAN_STD_by_loc.csv
    :param t: temperature profile with dimensions [TIMESTEPS, LOCATIONS, DEPTHS]
    :param s: salinity profile with dimensions [TIMESTEPS, LOCATIONS, DEPTHS]
    :param normalize: boolean indicating true for nomalization false for denormalization
    :param loc: which locations to normalize or denormalize
    :return:
    """
    # Read the statistics file
    df = pd.read_csv(stats_file)

    # Fixing the string with pandas
    tot_loc = t.shape[1]
    if loc == "all":
        loc = range(tot_loc)

    for i_loc, c_loc in enumerate(loc):
        # print(c_loc)
        # Get current mean and STD for this location
        mean_temp = stringToArray(df.loc[c_loc, "mean_temp"])
        mean_saln = stringToArray(df.loc[c_loc, "mean_saln"])
        # mean_sigma = stringToArray(df.loc[c_loc, "mean_sigma"])
        std_temp = stringToArray(df.loc[c_loc, "std_temp"])
        std_saln = stringToArray(df.loc[c_loc, "std_saln"])
        # std_sigma = stringToArray(df.loc[c_loc, "std_sigma"])

        if normalize:
            # Update t and s values with the normalized value
            # t has dimensions Locations, 2(t,s), 78(depth) --> time, locations, depth
            t[:, i_loc, :] = (t[:, i_loc, :] - mean_temp)/std_temp
            s[:, i_loc, :] = (s[:, i_loc, :] - mean_saln)/std_saln
        else:
            t[:, i_loc, :] = (t[:, i_loc, :]*std_temp) + mean_temp
            s[:, i_loc, :] = (s[:, i_loc, :]*std_saln) + mean_saln

    return t, s

