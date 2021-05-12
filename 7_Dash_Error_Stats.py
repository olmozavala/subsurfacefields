import json
from io_project.read_utils import get_all_profiles
import trainingutils as utilsNN
from textwrap import dedent as d
from models.modelSelector import select_1d_model
import cmocean
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

from os.path import join

# https://dash.plot.ly/interactive-graphing
# https://plot.ly/python-api-reference/

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

config = get_prediction_params()

# -------- Read the summary file (with all the models already trained) --------------
summary_file = "/data/SubsurfaceFields/Output/SUMMARY/summary.csv"
df = pd.read_csv(summary_file)
# Adam [True] 2, 6, 10, 14, 18, 22
# Adam [False] 0, 4, 8, 12, 16, 20
model = df.iloc[22]  # Here we identify which model we want to use
port = 8080

# Setting Network type (only when network type is UNET)
name = model["Name"]
split_name = name.split("_")
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
# Set the name of the network
run_name = name.replace(".hdf5", "")
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

print("Reading all data...")
ssh, temp_profile, saln_profile, years, dyear, depths, latlons = get_all_profiles(input_folder_preproc, locations, test_ids)
print("Done!")
# locations, day_year, depths, t/s
nn_predictions = np.load(join(output_folder, run_name, "nn_prediction.npy"))

# Computing the corresponging density profiles
density_profile = np.zeros(temp_profile.shape)
nn_predictions_density = np.zeros(nn_predictions.shape[0:3])
for loc_id in range(density_profile.shape[1]):
    for c_day in range(density_profile.shape[0]):
        _, density_profile[c_day, loc_id, :] = swstate(saln_profile[c_day, loc_id, :], temp_profile[c_day, loc_id, :], depths[loc_id,:])
        _, nn_predictions_density[loc_id, c_day, :] = swstate(nn_predictions[loc_id, c_day, :, 1], nn_predictions[loc_id, c_day, :, 0], depths[loc_id,:])

meta = {'loc_index': np.array(range(tot_loc))}

# =========== The easiest way is to use scatter_mapbox from a dataframe or from data
# https://plotly.github.io/plotly.py-docs/generated/plotly.express.scatter_mapbox.html

depths_int = [int(x) for x in depths[0,:]]

def getMap(selected_idx, mld):
    """
    Draws a map with the locations from the trained network. The selected id is drawn with a different color
    :param selected_idx:
    :return:
    """
    N = len(mld)
    if N > 0:
        cmdict = cmocean.tools.get_dict(cmocean.cm.thermal, N=N) # available colorpalettes here
        colors_str = ['#%02x%02x%02x' % (int(x[0]*255),int(x[0]*255),int(x[2]*255)) for x in cmdict['red']]
    else:
        colors_str = 'fuchsia'

    if selected_idx == -1:
        selected_idx = 0

    mydata = [dict(
        lat=latlons[:,0],
        lon=latlons[:,1],
        type="scattermapbox",
        customdata=meta['loc_index'],
        marker=dict(color=colors_str) ),
        dict(
            lat=[latlons[selected_idx,0]],
            lon=[latlons[selected_idx,1]],
            type="scattermapbox",
            marker=dict(color="green", size=10),
        )]
    fig = dict(
        data=mydata,
        layout=dict(
            mapbox=dict(
                center=dict(
                    lat=24, lon=-87
                ),
                style='open-street-map',
                # open-street-map, white-bg, carto-positron, carto-darkmatter,
                # stamen-terrain, stamen-toner, stamen-watercolor
                pitch=0,
                # zoom=1,
                zoom=4,
            ),
            height=800
            # autosize=True,
        )
    )
    return fig

def getScatterPlot(data, nn_predictions, loc_id, day_year, title, max_depth, mld):
    nonan = np.argmax(data[day_year, loc_id,:])
    if nonan > 0:
        max_depth = np.min([max_depth, nonan])
    mse_all = np.nanmean((data[:, loc_id, :] - nn_predictions[loc_id, :, :])**2)
    mse_current = np.nanmean((data[day_year, loc_id, :] - nn_predictions[loc_id, day_year, :])**2)
    return {
            'data': [
                {'x': np.mean(data[:, loc_id, 0:max_depth], axis=0), 'y': depths_int[0:max_depth], 'mode': 'lines', 'name': 'Mean/STD', 'opacity':0.5,  #  Mean and STD
                 'error_x': dict(type='data', array=np.std(data[:, loc_id, 0:max_depth], axis=0), visible=True), },
                {'x': np.mean(nn_predictions[loc_id, :, 0:max_depth], axis=0), 'y': depths_int[0:max_depth], 'mode': 'lines', 'name': 'NN Mean/STD', 'opacity':0.5,  #  Mean and STD
                 'error_x': dict(type='data', array=np.std(nn_predictions[loc_id, :, 0:max_depth], axis=0), visible=True), },
                {'x': data[day_year, loc_id, 0:max_depth], 'y': depths_int[0:max_depth], 'mode': 'markers', 'name': 'Model'}, # Model
                {'x': nn_predictions[loc_id, day_year, 0:max_depth], 'y': depths_int[0:max_depth], 'mode': 'markers', 'name': 'NN'}, # NN
                {'x': [np.nanmin(data[:, loc_id, 0:max_depth]), np.nanmax(data[:, loc_id, 0:max_depth])], 'y': [mld, mld], 'mode': 'line', 'name': F'MLD {mld}'}, # NN
            ],
            'layout': {
                'title': F"{title} MSE: {mse_current:0.3f}  Mean MSE {mse_all: 0.3f} (all dates)",
                # 'plot_bgcolor': colors['background'],
                # 'paper_bgcolor': colors['background'],
                # 'font':{
                #     'color': colors['text']
                # }
                'yaxis':{ 'autorange':'reversed'}
            }
        }

app.layout = dbc.Container([
        dbc.Row(
            dbc.Col(html.Div(name, className="centered"), width=12)
        ),
        dbc.Row(
            dbc.Col(dcc.Graph(figure=getMap(-1, []), id="id-map"), width=12)
        ),
        dbc.Row([
            dbc.Col( ["Maximum depth to show:", dcc.Dropdown(
                id='depth-selection',
                options=[{'label': F"{x} mts", 'value': i} for i, x in enumerate(depths_int)],
                value=40)], width=2),
            dbc.Col( ["Day of year:", dcc.Slider(
                id='day-selection',
                min=0,
                max=len(dyear),
                step=1,
                marks={i : str(x) for i,x in enumerate(dyear)},
                value=0)], width=8)
                ]),
        dbc.Row( [
            dbc.Col(dcc.Graph(figure=getScatterPlot(temp_profile, nn_predictions[:,:,:,0], 0, 0,  "Temperature", 78, 0), id="t-scatter"), width=6),
            dbc.Col(dcc.Graph(figure=getScatterPlot(saln_profile, nn_predictions[:,:,:,1], 0, 0, "Salinity", 78, 0), id="s-scatter"), width=6),
            dbc.Col(dcc.Graph(figure=getScatterPlot(density_profile, nn_predictions[:,:,:,0], 0, 0, "Density", 78, 0), id="sigma-scatter"), width=6)
        ]),
    ], fluid=True)

@app.callback(
    [Output('t-scatter', 'figure'),
     Output('s-scatter', 'figure'),
     Output('sigma-scatter', 'figure'),
     Output('id-map', 'figure')
     ],
    [Input('id-map', 'clickData'),
     Input('depth-selection', 'value'),
     Input('day-selection', 'value'),
     ])
def display_hover_data(map_data, depth_id, day_year):
    if map_data == None:
        loc_id = 0
    else:
        if 'customdata' in map_data['points'][0].keys():
            loc_id = map_data['points'][0]['customdata']
        else:
            loc_id = 0

    max_depth = depth_id
    mld = MLD(saln_profile[day_year, loc_id, 0:max_depth], temp_profile[day_year, loc_id, 0:max_depth], depths[loc_id,0:max_depth])
    mld_all = [MLD(saln_profile[day_year, i, :], temp_profile[day_year, i, :], depths[i,:]) for i in range(latlons.shape[0])]

    # (data, nn_predictions, loc_id, day_year, id_field, title):
    return [getScatterPlot(temp_profile, nn_predictions[:,:,:,0], loc_id, day_year,  F"Temperature day {dyear[day_year]} Loc {loc_id} ", max_depth, mld),
            getScatterPlot(saln_profile, nn_predictions[:,:,:,1], loc_id, day_year, F"Salinity day {dyear[day_year]} Loc {loc_id} ", max_depth, mld),
            getScatterPlot(density_profile, nn_predictions_density, loc_id, day_year, F"Density day {dyear[day_year]} Loc {loc_id} ", max_depth, mld),
            getMap(loc_id, mld_all)]

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
