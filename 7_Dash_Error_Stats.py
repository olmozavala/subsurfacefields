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
model = df.iloc[0]  # Here we identify which model we want to use
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
    # locations = np.random.randint(0, MAX_LOCATION, RAND_LOC)
    locations = range(RAND_LOC)
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
[train_ids, val_ids, test_ids] = utilsNN.split_train_validation_and_test(int(36*years),
                                                                         val_percentage=val_perc,
                                                                         test_percentage=test_perc,
                                                                         shuffle_ids=False)

print("Reading all data...")
ssh, temp_profile, saln_profile, years, dyear, depths, latlons = get_all_profiles(input_folder_preproc, locations, test_ids)
print("Done!")

# locations, day_year, depths, t/s
nn_predictions = np.load(join(output_folder, run_name, "nn_prediction.npy"))

meta = {'loc_index': np.array(range(tot_loc))}

# =========== The easiest way is to use scatter_mapbox from a dataframe or from data
# https://plotly.github.io/plotly.py-docs/generated/plotly.express.scatter_mapbox.html


depths_int = [int(x) for x in depths[0,:]]

def getMap(selected_idx):
    """
    Draws a map with the locations from the trained network. The selected id is drawn with a different color
    :param selected_idx:
    :return:
    """
    # fig = px.scatter_mapbox(meta, lat=latlons[:,0], lon=latlons[:,1], hover_data=["loc_index"],
    #                         color_discrete_sequence=["fuchsia"], zoom=1, height=600)
    # fig.update_layout(mapbox_style="open-street-map", margin={"r":0,"t":0,"l":0,"b":0})

    if selected_idx == -1:
        selected_idx = 0

    mydata =[dict(
        lat=latlons[:,0],
        lon=latlons[:,1],
        type="scattermapbox",
        customdata=meta['loc_index'],
        marker=dict(color="fuchsia") ),
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

def getScatterPlot(data, nn_predictions, loc_id, day_year, id_field, title, max_depth):
    nonan = np.argmax(data[day_year, loc_id,:])
    if nonan > 0:
        max_depth = np.min([max_depth, nonan])
    mse = np.nanmean((data[:, loc_id, :] - nn_predictions[loc_id, :, :, id_field])**2)
    return {
            'data': [
                {'x': np.mean(data[:, loc_id, 0:max_depth], axis=0), 'y': depths_int[0:max_depth], 'mode': 'lines', 'name': 'Mean/STD', 'opacity':0.5,  #  Mean and STD
                 'error_x': dict(type='data', array=np.std(data[:, loc_id, 0:max_depth], axis=0), visible=True), },
                {'x': np.mean(nn_predictions[loc_id, :, 0:max_depth, id_field], axis=0), 'y': depths_int[0:max_depth], 'mode': 'lines', 'name': 'NN Mean/STD', 'opacity':0.5,  #  Mean and STD
                 'error_x': dict(type='data', array=np.std(nn_predictions[loc_id, :, 0:max_depth, id_field], axis=0), visible=True), },
                {'x': data[day_year, loc_id, 0:max_depth], 'y': depths_int[0:max_depth], 'mode': 'markers', 'name': 'Model'}, # Model
                {'x': nn_predictions[loc_id, day_year, 0:max_depth, id_field], 'y': depths_int[0:max_depth], 'mode': 'markers', 'name': 'NN'}, # NN
            ],
            'layout': {
                'title': F"{title} MSE: {mse:0.3f}",
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
            dbc.Col(dcc.Graph(figure=getMap(-1), id="id-map"), width=12)
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
                step=None,
                marks={i : str(x) for i,x in enumerate(dyear)},
                value=0)], width=8)
                ]),
        dbc.Row( [
            dbc.Col(dcc.Graph(figure=getScatterPlot(temp_profile, nn_predictions, 0, 0, 0,  "Temperature", 78), id="t-scatter"), width=6),
            dbc.Col(dcc.Graph(figure=getScatterPlot(saln_profile, nn_predictions, 0, 0, 1, "Salinity", 78), id="s-scatter"), width=6)
        ]),
        # dbc.Row([
        #         dbc.Col( [
        #             dcc.Markdown(d(""" **Hover Data**
        #                     Mouse over values in the graph.""")),
        #             html.Pre(id='hover-data')], width=3),
        #         dbc.Col( [
        #             dcc.Markdown(d(""" **Clicked Data**
        #                             Mouse over values in the graph.""")),
        #             html.Pre(id='click-data')], width=3),
        #         dbc.Col( [
        #             dcc.Markdown(d(""" **Select Data**
        #                                     Mouse over values in the graph.""")),
        #             html.Pre(id='selected-data')], width=3),
        # ]),
    ])

@app.callback(
    [Output('t-scatter', 'figure'),
     Output('s-scatter', 'figure'),
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
    # (data, nn_predictions, loc_id, day_year, id_field, title):
    return [getScatterPlot(temp_profile, nn_predictions, loc_id, day_year, 0,  F"Temperature day {dyear[day_year]} Loc {loc_id} ", max_depth),
            getScatterPlot(saln_profile, nn_predictions, loc_id, day_year, 1,  F"Salinity day {dyear[day_year]} Loc {loc_id} ", max_depth),
            getMap(loc_id)]

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
    app.run_server(debug=True)
    # app.run_server(debug=False, port=port, host='146.201.212.115')
