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

# -------- Read the summary file (with all the models already trained) --------------
port = 8080
config = get_prediction_params()
input_folder_preproc = config[PredictionParams.input_folder]
locations = np.arange(637)
# locations = np.arange(0,100,2)
tot_loc = len(locations)
time_steps = np.arange(1400, 1530)

print(F"Reading all data LOCATIONS {locations}...")
ssh, temp_profile, saln_profile, years, dyear, depths, latlons = get_all_profiles(input_folder_preproc, locations, time_steps)
u_dyear = np.unique(dyear)
print("Done!")

# Computing the corresponging density profiles
density_profile = np.zeros(temp_profile.shape)
for loc_id in range(density_profile.shape[1]):
    for c_day in range(density_profile.shape[0]):
        _, density_profile[c_day, loc_id, :] = swstate(saln_profile[c_day, loc_id, :], temp_profile[c_day, loc_id, :], depths[loc_id,:])

meta = {'loc_index': np.array(range(tot_loc))}

depths_int = [int(x) for x in depths[0,:]]

MyFigObj = FiguresAndPlots(latlons, meta, u_dyear, depths_int)

# Gets the default layout
app.layout = dbc.Container([
        dbc.Row(
            dbc.Col(dcc.Graph(figure=MyFigObj.getLocationsMap(), id="id-map"), width=12)
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
            dbc.Col(dcc.Graph(figure=MyFigObj.getProfilesPlotSingle(temp_profile, 0, 0,  "Temperature", 78, "C"), id="t-scatter"), width=4),
            dbc.Col(dcc.Graph(figure=MyFigObj.getProfilesPlotSingle(saln_profile, 0, 0, "Salinity", 78, "S"), id="s-scatter"), width=4),
            dbc.Col(dcc.Graph(figure=MyFigObj.getProfilesPlotSingle(density_profile, 0, 0, "Density", 78, "D"), id="sigma-scatter"), width=4)
        ]),
    ], fluid=True)
#
@app.callback(
    [Output('t-scatter', 'figure'),
     Output('s-scatter', 'figure'),
     Output('sigma-scatter', 'figure'),
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
    # (data, nn_predictions, loc_id, day_year, id_field, title):
    return [MyFigObj.getProfilesPlotSingle(temp_profile, loc_id, day_year,  F"Temperature day {dyear[day_year]} Loc {locations[loc_id]} ", max_depth, deg_txt),
            MyFigObj.getProfilesPlotSingle(saln_profile, loc_id, day_year, F"Salinity day {dyear[day_year]} Loc {locations[loc_id]} ",     max_depth, "S"),
            MyFigObj.getProfilesPlotSingle(density_profile, loc_id, day_year, F"Densityday {dyear[day_year]} Loc {locations[loc_id]} ",     max_depth, "D"),
            MyFigObj.getLocationsMap(),
            F"Year {years[day_year]}  day {dyear[day_year]}"
            ]

if __name__ == '__main__':
    app.run_server(debug=True)
    # app.run_server(debug=False, port=port, host='146.201.212.115')
