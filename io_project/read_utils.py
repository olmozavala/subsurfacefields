from preproc.UtilsDates import get_days_from_month
from constants_proj.AI_proj_params import PreprocParams
import pandas as pd
from pandas import DataFrame
import numpy.ma as ma
from os.path import join, isfile
import xarray as xr
import os
import numpy as np
import cv2
from datetime import date
from ExtraUtils.VizUtilsProj import draw_profile
import re
# Common
from ai_common.constants.AI_params import TrainingParams, ModelParams, AiModels

def get_all_profiles(input_folder, all_loc, time_steps=np.arange(1548)):
    """
    This function reads all the requested locations and generates arrays separated by dates.
    :param input_folder: Where the profile files are stored (the ones splitted by locations)
    :param all_loc: An array of integers indicating the number of the locations
    :return: Each output array should contain 216 time steps
    """

    input_file = join(input_folder, "all_data.nc")

    # Iterate for each selected year
    ds = xr.open_dataset(input_file)

    ssh = ds.ssh.values[all_loc,:][:,time_steps]
    saln =  ds.s.values[all_loc, :, :][:, time_steps, :]
    temp = ds.t.values[all_loc, :, :][:, time_steps, :]
    depths = ds.depth.values
    latlons = np.array([ds.lat.values[all_loc], ds.lon.values[all_loc]])
    years = ds.year.values[time_steps]
    dyear = ds.dyear.values[time_steps]

    ssh = np.swapaxes(ssh, 0, 1)
    saln = np.swapaxes(saln, 0, 1)
    temp = np.swapaxes(temp, 0, 1)
    latlons= np.swapaxes(latlons, 0, 1)
    # The first two indexes are the timestep and the location
    return ssh, temp, saln, years, dyear, depths, latlons


def get_profiles_byloc(input_folder, loc_prof, time_steps):
    """
    It reads the information for one specific location
    :param input_folder:
    :param loc_prof:
    :param time_steps:
    :return:
    """
    input_file = join(input_folder, "all_data.nc")

    # Iterate for each selected year
    ds = xr.open_dataset(input_file)
    depth = list(ds.depth)
    latlon = np.array([ds.lat.values, ds.lon.values])

    ssh = list(ds.ssh.values[loc_prof, time_steps])
    temp_profile = list(ds.t.values[loc_prof, time_steps, :])
    saln_profile = list(ds.s.values[loc_prof, time_steps, :])

    # return np.array(ssh), np.array(temp_profile), np.array(saln_profile), np.array(years, dtype=np.int), np.array(dyear, dtype=np.int), np.array(depth, dtype=np.int), latlon
    return np.array(ssh), np.array(temp_profile), np.array(saln_profile), -1, np.array(time_steps, dtype=np.int), np.array(depth, dtype=np.int), latlon


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

    tout = np.zeros(t.shape)
    sout = np.zeros(s.shape)

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
            tout[:, i_loc, :] = (t[:, i_loc, :] - mean_temp)/std_temp
            sout[:, i_loc, :] = (s[:, i_loc, :] - mean_saln)/std_saln
        else:
            tout[:, i_loc, :] = (t[:, i_loc, :]*std_temp) + mean_temp
            sout[:, i_loc, :] = (s[:, i_loc, :]*std_saln) + mean_saln

    return tout, sout

def generateXY(input_file, config):
    dims = config[ModelParams.INPUT_SIZE]
    """
    Generates the X,Y from the input file
    :param input_file:
    :return:
    """
    ds = xr.load_dataset(input_file)
    # ------------------------------ ORIGINAL VERSION ---------------------------------
    X = 1
    Y = 1

    return X, Y