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

NUM_PROC = 1


def create_folder(output_folder):
    """ It simply verifies if a folder already exists, if not it creates it"""
    if not(os.path.exists(output_folder)):
        os.makedirs(output_folder)

def main():
    # ----------- Parallel -------
    p = Pool(NUM_PROC)
    p.map(img_generation_all, range(NUM_PROC))

def data_summary(ds):
    print("------------- Data summary ---------------------")
    print(ds.head())
    df = ds.to_dataframe()
    print(df.describe())


def plotSparseDataFiles(file_names, input_folder, output_folder, file_prefix=""):
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
        out_file = join(output_folder, F"{file_prefix}Gloal_Locations_{years[0].values:0.0f}_{plot_day}.png")
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
    path = geopd.datasets.get_path('naturalearth_lowres')
    world = geopd.read_file(path)

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

def img_generation_all(proc_id):
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
    # plotSparseDataFiles(sparse_files, input_folder, output_folder)
    print("Plotting 5Deg files...")
    plot5DegDataFiles(five_deg_files, input_folder, output_folder, "5Deg")
    # plot5DegDataFiles(sparse_files, input_folder, output_folder, "5Deg")
    print("Done!")

if __name__ == '__main__':
    main()

##

