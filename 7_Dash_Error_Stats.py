import json
from io_project.read_utils import get_all_profiles
import trainingutils as utilsNN
from textwrap import dedent as d
from models.modelSelector import select_1d_model
import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from metrics_proj.isop_metrics import swstate, MLD
import plotly.express as px
import numpy as np

import pandas as pd
from config.MainConfig import get_prediction_params
from constants_proj.AI_proj_params import PredictionParams, ProjTrainingParams, MAX_LOCATION
from constants.AI_params import TrainingParams, ModelParams
from img_viz.common import create_folder
from dash_utils.dash_utils_oz import FiguresAndPlots

from os.path import join

deg_txt = u"\N{DEGREE SIGN}C"
# https://dash.plot.ly/interactive-graphing
# https://plot.ly/python-api-reference/

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

config = get_prediction_params()

# -------- Read the summary file (with all the models already trained) --------------
summary_file = "/data/SubsurfaceFields/Output/SUMMARY/summary.csv"
df = pd.read_csv(summary_file)
# Adam [True] 2, 6, 10, 14, 18, 22
# Adam [False] 0, 4, 8, 12, 16, 20
# model = df.iloc[27]  # Here we identify which model we want to use
model = df.iloc[7]  # Here we identify which model we want to use
port = 8085

# ======== Updating the config info  ==========
# -------- Setting locations an dnetwork parameters ----
MODEL_NAME = model["Name"]
split_name = MODEL_NAME.split("_")
hid_layers = int(split_name[5])
hid_layer_size = int(split_name[3])
RAND_LOC = int(split_name[1])
seed = int(split_name[9])
np.random.seed(seed)  # THIS IS VERY IMPORTANT BECAUSE WE NEED IT SO THAT THE NETWORKS ARE TRAINED AND TESTED WITH THE SAME LOCATIONS
if RAND_LOC == MAX_LOCATION: # Here we select the locations we want to use
    locations = range(RAND_LOC)
else:
    locations = np.random.choice(range(MAX_LOCATION), RAND_LOC, replace=False)

config[ProjTrainingParams.locations] = locations
config[ModelParams.HIDDEN_LAYERS] = hid_layers
config[ModelParams.CELLS_PER_HIDDEN_LAYER] = [hid_layer_size for x in range(hid_layers) ]
config[ModelParams.INPUT_SIZE] = int(split_name[1])*2 + 1
config[ProjTrainingParams.normalize] = split_name[7] == "True"
config[ModelParams.NUMBER_OF_OUTPUT_CLASSES] = RAND_LOC*config[ProjTrainingParams.tot_depths]*2
config[PredictionParams.model_weights_file] = model["Path"]
print(F"Model's weight file: {model['Path']}")
# Set the MODEL_NAME of the network
run_name = MODEL_NAME.replace(".hdf5", "")
config[TrainingParams.config_name] = run_name
input_folder_preproc = config[ProjTrainingParams.input_folder_preproc]
output_folder = config[PredictionParams.output_folder]
model_weights_file = config[PredictionParams.model_weights_file]
output_imgs_folder = config[PredictionParams.output_imgs_folder]
run_name = config[TrainingParams.config_name]
val_perc = config[TrainingParams.validation_percentage]
test_perc = config[TrainingParams.test_percentage]
stats_input_file = config[ProjTrainingParams.stats_file]
locations = config[ProjTrainingParams.locations]
years = config[ProjTrainingParams.years]
tot_loc = len(locations)

output_imgs_folder = join(output_imgs_folder, run_name)
create_folder(output_imgs_folder)

# *********** Read files to predict***********
total_timesteps = int(years*36)
[train_ids, val_ids, test_ids] = utilsNN.split_train_validation_and_test(total_timesteps,
                                                                         val_percentage=val_perc,
                                                                         test_percentage=test_perc,
                                                                         shuffle_ids=False)

print(F"Reading all data LOCATIONS {locations}...")
ssh, temp_profile, saln_profile, years, dyear, depths, latlons = get_all_profiles(input_folder_preproc, locations, test_ids)
print("Done!")
# locations, day_year, depths, t/s
nn_predictions = np.load(join(output_folder, run_name, "nn_prediction.npy"))

# Computing the corresponging density profiles
density_profile = np.zeros(temp_profile.shape)
nn_predictions_density = np.zeros(nn_predictions.shape[0:3])
for loc_id in range(density_profile.shape[1]):
    for c_day in range(density_profile.shape[0]):
        _, density_profile[c_day, loc_id, :] = swstate(saln_profile[c_day, loc_id, :], temp_profile[c_day, loc_id, :], depths)
        _, nn_predictions_density[loc_id, c_day, :] = swstate(nn_predictions[loc_id, c_day, :, 1], nn_predictions[loc_id, c_day, :, 0], depths)

meta = {'loc_index': np.array(range(tot_loc))}

# ========== Compute summary comparisons
nn_predictions_t = np.swapaxes(nn_predictions[:,:,:,0],0,1)
nn_predictions_s = np.swapaxes(nn_predictions[:,:,:,1],0,1)

# ============= RMSE by day of year
u_dyear = np.unique(dyear)
rmse_by_dyear_t = np.zeros((depths.shape[0], 36))  # Locations, dayyear, depths
rmse_by_dyear_s = np.zeros((depths.shape[0], 36))  # Locations, dayyear, depths
for i, c_dyear in enumerate(u_dyear):
    dyear_idxs = dyear == c_dyear
    rmse_by_dyear_t[:, i] = np.sqrt(np.nanmean((temp_profile[dyear_idxs, :, :] - nn_predictions_t[dyear_idxs, :, :])**2, axis=(0,1)))
    rmse_by_dyear_s[:, i] = np.sqrt(np.nanmean((saln_profile[dyear_idxs, :, :] - nn_predictions_s[dyear_idxs, :, :])**2, axis=(0,1)))

# ============= RMSE by depth
rmse_by_depth_t = np.sqrt(np.nanmean((temp_profile - nn_predictions_t)**2, axis=(0,1)))
rmse_by_depth_s = np.sqrt(np.nanmean((saln_profile - nn_predictions_s)**2, axis=(0,1)))

# ============= RMSE by location
rmse_by_geo_t = np.sqrt(np.nanmean((temp_profile - nn_predictions_t)**2, axis=(0,2)))
rmse_by_geo_s = np.sqrt(np.nanmean((saln_profile - nn_predictions_s)**2, axis=(0,2)))

depths_int = [int(x) for x in depths]

MyFigObj = FiguresAndPlots(latlons, meta, u_dyear, depths_int)

# Gets the default layout
app.layout = dbc.Container([
        dbc.Row(
            dbc.Col(dcc.Graph(figure=MyFigObj.getLocationsMapWithSpecifiedColor(-1, rmse_by_geo_t, rmse_by_geo_s, MODEL_NAME), id="id-map"), width=12)
        ),
        dbc.Row([
            dbc.Col(["Maximum depth to show:", dcc.Dropdown(
                id='depth-selection',
                options=[{'label': F"{x} mts", 'value': i} for i, x in enumerate(depths_int)],
                value=40)], width=2),
            dbc.Col( [
                html.Div("Select date:", id="id-dayyear"), dcc.Slider(
                id='day-selection',
                min=0,
                max=len(dyear),
                step=1,
                # marks={i : str(x) for i,x in enumerate(dyear)},
                value=0)], width=8)
                ]),
        dbc.Row( [
            dbc.Col(dcc.Graph(figure=MyFigObj.getProfilesPlot(temp_profile, nn_predictions[:,:,:,0], 0, 0,  "Temperature", 78, 0, 0, deg_txt), id="t-scatter"), width=4),
            dbc.Col(dcc.Graph(figure=MyFigObj.getProfilesPlot(saln_profile, nn_predictions[:,:,:,1], 0, 0, "Salinity", 78, 0, 0), id="s-scatter"), width=4),
            dbc.Col(dcc.Graph(figure=MyFigObj.getProfilesPlot(density_profile, nn_predictions_density, 0, 0, "Density", 78, 0, 0), id="sigma-scatter"), width=4)
        ]),
        dbc.Row( [
            dbc.Col(dcc.Graph(figure=MyFigObj.getErrorByDepth(rmse_by_depth_t, 78, "RMSE by depth Temperature", "C"), id="id-error-t"), width=3),
            dbc.Col(dcc.Graph(figure=MyFigObj.getErrorByDepth(rmse_by_depth_s, 78, "RMSE by depth Salinity"), id="id-error-s"), width=3),
            dbc.Col(dcc.Graph(figure=MyFigObj.getErrorByYearPlot(rmse_by_dyear_t, 0, "RMSE by Day of year Temperature", deg_txt), id="id-errorbyday-t"), width=3),
            dbc.Col(dcc.Graph(figure=MyFigObj.getErrorByYearPlot(rmse_by_dyear_s, 0, "RMSE by Day of year Salinity"), id="id-errorbyday-s"), width=3),
        ]),
], fluid=True)

@app.callback(
    [Output('t-scatter', 'figure'),
     Output('s-scatter', 'figure'),
     Output('sigma-scatter', 'figure'),
     Output('id-error-t', 'figure'),
     Output('id-error-s', 'figure'),
     # Output('id-errorbyday-t', 'figure'),
     # Output('id-errorbyday-s', 'figure'),
     Output('id-map', 'figure'),
     Output('id-dayyear', 'children'),
     ],
    [Input('id-map', 'clickData'),
     Input('depth-selection', 'value'),
     Input('day-selection', 'value'),
     ])
def display_hover_data(map_data, depth_id, day_year):
    if map_data == None:
        loc_id = 0
    else:
        # Here we modify the location to its index
        if 'customdata' in map_data['points'][0].keys():
            loc_id = map_data['points'][0]['customdata']
        else:
            loc_id = 0

    max_depth = depth_id
    # locations, day_year, depths, t/s
    mld_nn = MLD(nn_predictions[loc_id, day_year, 0:max_depth,1], nn_predictions[loc_id, day_year, 0:max_depth,1], depths[0:max_depth])
    mld = MLD(saln_profile[day_year, loc_id, 0:max_depth], temp_profile[day_year, loc_id, 0:max_depth], depths[0:max_depth])
    mld_all = [MLD(saln_profile[day_year, i, :], temp_profile[day_year, i, :], depths) for i in range(latlons.shape[0])]

    # (data, nn_predictions, loc_id, day_year, id_field, title):
    return [MyFigObj.getProfilesPlot(temp_profile, nn_predictions[:,:,:,0], loc_id, day_year,  F"Temperature day {dyear[day_year]} Loc {loc_id} ", max_depth, mld, mld_nn, deg_txt),
            MyFigObj.getProfilesPlot(saln_profile, nn_predictions[:,:,:,1], loc_id, day_year, F"Salinity day {dyear[day_year]} Loc {loc_id} ",     max_depth, mld, mld_nn),
            MyFigObj.getProfilesPlot(density_profile, nn_predictions_density, loc_id, day_year, F"Density day {dyear[day_year]} Loc {loc_id} ",    max_depth, mld, mld_nn),
            MyFigObj.getErrorByDepth(rmse_by_depth_t, max_depth, "RMSE by depth Temperature (All locations)", deg_txt),
            MyFigObj.getErrorByDepth(rmse_by_depth_s, max_depth, "RMSE by depth Salinity (All locations)"),
            # getErrorByDyearPlot(rmse_by_dyear_t, loc_id, "RMSE by day of year Temperature (All locations)"),
            # getErrorByDyearPlot(rmse_by_dyear_s, loc_id, "RMSE by day of year Salinity (All locations)"),
            MyFigObj.getLocationsMapWithSpecifiedColor(loc_id, rmse_by_geo_t, rmse_by_geo_s, MODEL_NAME),
            F"Year {years[day_year]}  day {dyear[day_year]}"
            ]

# @app.callback(
#     Output('click-data', 'children'),
#     [Input('id-map', 'clickData')])
# def display_click_data(clickData):
#     return json.dumps(clickData, indent=2)
#
#
# @app.callback(
#     Output('selected-data', 'children'),
#     [Input('id-map', 'selectedData')])
# def display_selected_data(selectedData):
#     return json.dumps(selectedData, indent=2)

if __name__ == '__main__':
    # app.run_server(debug=True)
    app.run_server(debug=False, port=port, host='146.201.212.115')
