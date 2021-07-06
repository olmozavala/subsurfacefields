from multiprocessing import Pool
from constants_proj.AI_proj_params import PreprocParams
from os.path import join
import skimage.io as io
import xarray as xr
import numpy as np
from datetime import date
import geopandas as geopd
import os
from ExtraUtils.VizUtilsProj import draw_profile
import matplotlib.pyplot as plt
import pandas as pd
import re

def create_folder(output_folder):
    """ It simply verifies if a folder already exists, if not it creates it"""
    if not(os.path.exists(output_folder)):
        os.makedirs(output_folder)

def main():
    # ----------- Parallel -------
    img_generation_all()
    # img_generation_3D()

def data_summary(ds):
    print("------------- Data summary ---------------------")
    print(ds.head())
    df = ds.to_dataframe()
    print(df.describe())

def plotSparseDataFiles(file_names, input_folder, output_folder, file_prefix=""):
    """This function makes maps of the locations where the profiles are and
     also pltos some random profiles. """
    path = geopd.datasets.get_path('naturalearth_lowres')
    world = geopd.read_file(path)

    # Plotting type1 file ( sparse )
    c_file = file_names[0]
    print(F"Reading file {c_file}....")
    ds = xr.open_dataset(join(input_folder, c_file))
    data_summary(ds)
    tot_prof = len(ds['lat_nn']) # Total number of profiles

    n_plot = 10  # How many random profiles do we want to plot
    ids_to_plot = np.random.randint(0,tot_prof,n_plot)

    lats = ds['lat_nn']
    lons = ds['lon_nn']
    depths = ds['depth']
    years = ds['year']
    months = ds['month']
    days = ds['day']
    yday = ds['yday']

    # ---------- Plot locations on a map
    print("\tPlotting map locations...")
    for plot_day in range(1, n_plot):
        print("\t",plot_day)
        c_day_id = yday == plot_day
        c_lats = lats[c_day_id]
        c_lons = lons[c_day_id]
        f_lons = c_lons >= 180
        c_lons[f_lons] = c_lons[f_lons] - 360
        df = geopd.GeoDataFrame({
            "ID": range(len(c_lats)),
            "geometry": geopd.points_from_xy(c_lons, c_lats)
        })
        ax = world.plot()
        df.plot(ax=ax, color='r')
        plt.title(F"Locations for day: {years[0].values:0.0f} day of year {plot_day} ")
        out_file = join(output_folder, F"{file_prefix}Global_Locations_{years[0].values:0.0f}_{plot_day}.png")
        plt.savefig(out_file)
        plt.close()
    print("\tDone!...")
    dates = [date(years[i], months[i], days[i]) for i in range(tot_prof)]

    # Plot a random number of profiles
    print("\tPlotting random profiles...")
    for c_id in ids_to_plot:
        temp = ds['temp_level'][c_id, :]
        sal = ds['saln_level'][c_id, :]

        title = F"{dates[c_id]} at {lats[c_id].values:0.2f}, {lons[c_id].values:0.2f}"
        out_file = join(output_folder, F"{file_prefix}{dates[c_id].year:02d}{dates[c_id].month:02d}-{dates[c_id].day:02d}_{lats[c_id].values:0.0f}lat_{lons[c_id].values:0.2f}lon.png")
        draw_profile(temp, sal, depths, title, out_file)
    print("\tDone!...")

    print("Done!")

def plot5DegDataFiles(file_names, input_folder, output_folder, file_prefix=""):
    """
    Makes rando profile plots from files
    :param file_names:
    :param input_folder:
    :param output_folder:
    :param file_prefix:
    :return:
    """

    # Plotting type1 file ( sparse )
    c_file = file_names[0]
    print(F"Reading file {c_file}....")
    ds = xr.open_dataset(join(input_folder, c_file))
    # data_summary(ds)
    # TODO we are going to plot a single day
    tot_prof = ds.ssh.shape[0]
    n_plot = 10

    # Plot a random number of profiles
    print("\tPlotting random profiles...")
    for c_id in np.random.randint(0, tot_prof, n_plot):
        print(F"\tPlotting profile with id: {c_id}")

        temp = ds.temp_level[c_id, :]
        sal = ds.saln_level[c_id, :]
        day = ds.yday[0].item()
        year = ds.year[0].item()
        lat = ds.lat_nn[c_id].item()
        lon = ds.lon_nn[c_id].item()

        title = F"Day {year}-{day} at {lat:0.2f}, {lon:0.2f}"
        out_file = join(output_folder, F"{file_prefix}{year:0.0f}_{day:0.0f}_latlon_{lat:0.2f}_{lon:0.2f}.png")
        draw_profile(temp, sal, ds.depth.values, title, out_file)
    print("\tDone!...")

    print("Done!")

def img_generation_3D():
    """
    Makes images of the available data (Free run, DA and Observations)
    :param proc_id:
    :return:
    """
    input_folder = "/data/COAPS_nexsan/people/xbxu/hycom/GLBb0.08/profile/3d"
    output_folder = "/data/SubsurfaceFields/PreprocBK/imgs/3d"
    create_folder(output_folder)

    print("Reading files...")
    all_files = os.listdir(input_folder)
    for c_file in all_files:
        ds = xr.open_dataset(join(input_folder, c_file))
        ds.ssh[0,:,:].plot(figsize=(10,5))
        plt.show()
        ds.temp[0,0,:,:].plot(figsize=(10,5))
        plt.show()
        ds.saln[0,0,:,:].plot(figsize=(10,5))
        plt.show()

def img_generation_all():
    """
    Makes images of the available data (Free run, DA and Observations)
    :param proc_id:
    :return:
    """
    _preproc_folder = "/data/SubsurfaceFields/Preproc"
    # input_folder = "/data/COAPS_nexsan/people/xbxu/hycom/GLBb0.08/profile"
    input_folder = "/data/SubsurfaceFields/Input"
    output_folder = join(_preproc_folder, "imgs")
    # output_folder = "/home/data/Subsurface/imgs"
    create_folder(output_folder)

    print("Reading files...")
    all_files = os.listdir(input_folder)
    sparse_files = [x for x in all_files if "05deg" not in x]
    five_deg_files = [x for x in all_files if "05deg" in x]

    sparse_files.sort()
    five_deg_files.sort()

    print("Plotting sparse data files...")
    plotSparseDataFiles(sparse_files, input_folder, output_folder)
    print("Plotting 1/2 Deg files...")
    plot5DegDataFiles(five_deg_files, input_folder, output_folder, "5Deg")
    # plot5DegDataFiles(sparse_files, input_folder, output_folder, "5Deg")
    print("Done!")

def stringToArray(st_orig):
    # TODO this is very bad, improve it. We should save a np array somehow. How is it we cant read it?
    st_orig = re.sub("\ +", " ", st_orig)
    str_array = st_orig.replace("[ ","").replace("[","").replace("]","").replace("\n","").split(" ")
    return np.array([float(x) if x != "nan" else np.nan for x in str_array])

def plot_obtained_stats():
    all_depths = np.array([0.0,2.0,4.0,6.0,8.0,10.0,15.0,20.0,25.0,30.0,35.0,40.0,45.0,50.0,55.0,60.0,65.0,70.0,75.0,80.0,85.0,90.0,95.0,100.0,110.0,120.0,130.0,140.0,150.0,160.0,170.0,180.0,190.0,200.0,220.0,240.0,260.0,280.0,300.0,350.0,400.0,500.0,600.0,700.0,800.0,900.0,1000.0,1100.0,1200.0,1300.0,1400.0,1500.0,1600.0,1800.0,2000.0,2200.0,2400.0,2600.0,2800.0,3000.0,3200.0,3400.0,3600.0,3800.0,4000.0,4200.0,4400.0,4600.0,4800.0,5000.0,5200.0,5400.0,5600.0,5800.0,6000.0,6200.0,6400.0,6600.0])
    stats_file = "/data/SubsurfaceFields/PreprocGoM/MEAN_STD_by_loc.csv"
    df = pd.read_csv(stats_file)
    loc = np.arange(0,500)  # which locations to plot
    np.random.shuffle(loc)
    max_depth = 50
    for c_loc in loc:
        mean_t= stringToArray(df.loc[c_loc, "mean_temp"])
        mean_s = stringToArray(df.loc[c_loc, "mean_saln"])
        mean_d = stringToArray(df.loc[c_loc, "mean_sigma"])
        std_t = stringToArray(df.loc[c_loc, "std_temp"])
        std_s = stringToArray(df.loc[c_loc, "std_saln"])
        std_d = stringToArray(df.loc[c_loc, "std_sigma"])

        c_max_depth = np.where(np.isnan(mean_t))[0][0] - 1
        c_max_depth = np.min((c_max_depth, max_depth))

        t = mean_t[0:c_max_depth]
        s = mean_s[0:c_max_depth]
        d = mean_d[0:c_max_depth]
        std_t = std_t[0:c_max_depth]
        std_s = std_s[0:c_max_depth]
        std_d = std_d[0:c_max_depth]

        # -------------------- temperature
        ax1 = plt.subplot(1,3,1)
        ax1.errorbar(t, all_depths[:c_max_depth], xerr=std_t, c='r')
        ax1.set_xlabel('Temp')
        ax1.invert_yaxis()
        # -------------------- Salinity
        ax1 = plt.subplot(1,3,2)
        ax1.errorbar(s, all_depths[:c_max_depth], xerr=std_s, c='g')
        ax1.set_xlabel('Salinity')
        ax1.invert_yaxis()
        # -------------------- Density
        ax1 = plt.subplot(1,3,3)
        ax1.errorbar(d, all_depths[:c_max_depth], xerr=std_d, c='b')
        ax1.set_xlabel('Density')
        # ax1.set_xlim([0,40])
        ax1.invert_yaxis()
        plt.show()




if __name__ == '__main__':
    # main()
    plot_obtained_stats()

##

