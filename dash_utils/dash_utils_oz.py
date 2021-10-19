import cmocean
import numpy as np

class FiguresAndPlots:

    def __init__(self, latlons, meta, u_dyear, depths_int):
        self.latlons = latlons
        self.meta = meta
        self.u_dyear = u_dyear
        self.depths_int = depths_int

    def getLocationsMap(self):
        """
        Draws a map with the locations from the trained network. The selected id is drawn with a different color
        :param selected_idx:
        :return:
        """
        mymarker=dict(
            # cmin = np.amin(rmse_t),
            # cmax = np.amax(rmse_t),
            # color = rmse_t,
            colorscale="Oranges", # Greys,YlGnB u,Greens,YlOrRd,Bluered,RdBu,Reds,Blues,Picnic,Rainbow,Portland ,Jet,Hot,Blackbody,Earth,Electric,Viridis,Cividis.
            size=5,
            colorbar=dict(
                bgcolor="white",
                title="Locations"
            )
        )

        mydata = [
            # https://plotly.com/python-api-reference/generated/plotly.graph_objects.Scattermapbox.html
            dict(
                lat=self.latlons[:,0],
                lon=self.latlons[:,1]+.3,
                type="scattermapbox",
                customdata=self.meta['loc_index'],
                hovertemplate="Lat:%{lat:.2f} Lon:%{lon:.2f}",
                # hovertemplate="HOVER TEMPLATE <extra></extra>",
                marker=mymarker,
            ),
        ]
        fig = dict(
            data=mydata,
            layout=dict(
                mapbox=dict(
                    center=dict(
                        lat=24, lon=-87
                    ),
                    style='carto-positron',
                    # open-street-map, white-bg, carto-positron, carto-darkmatter,
                    # stamen-terrain, stamen-toner, stamen-watercolor
                    pitch=0,
                    # zoom=1,
                    zoom=4,
                ),
                height=600
                # autosize=True,
            )
        )
        return fig

    def getLocationsMapWithSpecifiedColor(self, selected_idx, rmse_t, rmse_s, title):
        """
        Draws a map with the locations from the trained network. The selected id is drawn with a different color
        :param selected_idx:
        :return:
        """
        N = len(rmse_t)
        if N > 0:
            mymarker_t=dict(
                # cmin = np.amin(rmse_t),
                # cmax = np.amax(rmse_t),
                color = rmse_t,
                colorscale="Oranges", # Greys,YlGnB u,Greens,YlOrRd,Bluered,RdBu,Reds,Blues,Picnic,Rainbow,Portland ,Jet,Hot,Blackbody,Earth,Electric,Viridis,Cividis.
                size=5,
                colorbar=dict(
                    bgcolor="white",
                    title="T RMSE"
                )
            )
            mymarker_s=dict(
                color = rmse_s,
                colorscale="Greens", # Greys,YlGnB u,Greens,YlOrRd,Bluered,RdBu,Reds,Blues,Picnic,Rainbow,Portland ,Jet,Hot,Blackbody,Earth,Electric,Viridis,Cividis.
                reversescale = True,
                size=5,
                colorbar=dict(
                    xpad=80,
                    title="S RMSE",
                    bgcolor="white",
                )
            )
        else:
            mymarker_t=dict(color='fuchsia')
            mymarker_s=dict(color='fuchsia')

        if selected_idx == -1:
            selected_idx = 0

        mydata = [
            # https://plotly.com/python-api-reference/generated/plotly.graph_objects.Scattermapbox.html
            dict(
                lat=self.latlons[:,0],
                lon=self.latlons[:,1]+.3,
                type="scattermapbox",
                customdata=self.meta['loc_index'],
                meta=rmse_s,
                # hovertemplate="Lat:%{lat:.2f} Lon:%{lon:.2f} RMSE:%{meta:.2f}",
                hovertemplate="RMSE %{meta:.2f} <extra></extra>",
                marker=mymarker_s,
            ),
            dict(
                lat=self.latlons[:,0],
                lon=self.latlons[:,1],
                type="scattermapbox",
                customdata=self.meta['loc_index'],
                meta=rmse_t,
                # hovertemplate="Lat:%{lat:.2f} Lon:%{lon:.2f} RMSE:%{meta:.2f}",
                hovertemplate="RMSE %{meta:.2f} C <extra></extra>",
                marker=mymarker_t,
            ),
            dict(
                lat=[self.latlons[selected_idx,0]],
                lon=[self.latlons[selected_idx,1]],
                meta=rmse_t,
                hovertemplate="Lat:%{lat:.2f} Lon:%{lon:.2f} RMSE:%{meta:.2f} <extra></extra>",
                type="scattermapbox",
                marker=dict(color="red", size=10)
            )
        ]
        fig = dict(
            data=mydata,
            layout=dict(
                title=title,
                mapbox=dict(
                    center=dict(
                        lat=24, lon=-87
                    ),
                    style='carto-positron',
                    # open-street-map, white-bg, carto-positron, carto-darkmatter,
                    # stamen-terrain, stamen-toner, stamen-watercolor
                    pitch=0,
                    # zoom=1,
                    zoom=4,
                ),
                height=500
                # autosize=True,
            )
        )
        return fig

    def getErrorByYearPlot(self, data, loc, title, units=""):
        """Gets the data to make the 'error by year' plot"""
        return {
            'data': [
                # {'x': u_dyear, 'y': data[loc,:], 'mode': 'line', 'name': 'Model', 'marker':{'color':'orange'}}, # Model
                {'x': self.u_dyear, 'y': np.nanmean(data,axis=0), 'mode': 'line', 'name': 'Model', 'marker':{'color':'orange'}}, # Model
            ],
            'layout': {
                'title': F"{title}",
                'yaxis':{ 'title':F"RMSE {units}"},
                'xaxis':{ 'title':'Day of Year'},
                'font': {'size': 15}
            }
        }

    def getErrorByDepth(self, data, max_depth, title, units=""):
        """Gets the data to make the 'error by depth' plot"""
        return {
            'data': [
                {'x': data[0:max_depth], 'y': self.depths_int[0:max_depth], 'mode': 'markers', 'name': 'Model'}, # Model
            ],
            'layout': {
                'title': F"{title}",
                'yaxis':{ 'title':'Depth (m)','autorange':'reversed'},
                'xaxis':{ 'title': units},
                'font': {'size': 15}
            }
        }

    def getProfilesPlot(self, data, nn_predictions, loc_id, day_year, title, max_depth, model_mld, nn_mld, xtitle=""):
        nonan = np.argmax(data[day_year, loc_id,:])
        if nonan > 0:
            max_depth = np.min([max_depth, nonan])
        rmse_all = np.sqrt(np.nanmean((data[:, loc_id, :] - nn_predictions[loc_id, :, :])**2))
        rmse_current = np.sqrt(np.nanmean((data[day_year, loc_id, :] - nn_predictions[loc_id, day_year, :])**2))
        return {
            'data': [
                {'x': np.mean(data[:, loc_id, 0:max_depth], axis=0), 'y': self.depths_int[0:max_depth], 'mode': 'lines', 'name': 'Model Mean/STD', 'opacity':0.25,  #  Mean and STD
                 'error_x': dict(type='data', array=np.std(data[:, loc_id, 0:max_depth], axis=0), visible=True), },
                {'x': np.mean(nn_predictions[loc_id, :, 0:max_depth], axis=0), 'y': self.depths_int[0:max_depth], 'mode': 'lines', 'name': 'NN Mean/STD', 'opacity':0.25,  #  Mean and STD
                 'error_x': dict(type='data', array=np.std(nn_predictions[loc_id, :, 0:max_depth], axis=0), visible=True), },
                {'x': data[day_year, loc_id, 0:max_depth], 'y': self.depths_int[0:max_depth], 'mode': 'markers', 'name': 'Model'}, # Model
                {'x': nn_predictions[loc_id, day_year, 0:max_depth], 'y': self.depths_int[0:max_depth], 'mode': 'markers', 'name': 'NN'}, # NN
                # {'x': [np.nanmin(data[:, loc_id, 0:max_depth]), np.nanmax(data[:, loc_id, 0:max_depth])], 'y': [model_mld, model_mld], 'mode': 'line', 'name': F'Model MLD {model_mld}'}, # NN
                # {'x': [np.nanmin(data[:, loc_id, 0:max_depth]), np.nanmax(data[:, loc_id, 0:max_depth])], 'y': [nn_mld, nn_mld], 'mode': 'line', 'name': F'NN MLD {nn_mld}'}, # NN
            ],
            'layout': {
                'title': F"{title} <br> RMSE: {rmse_current:0.3f}  <br> Mean RMSE {rmse_all: 0.3f} (all dates)",
                # 'plot_bgcolor': colors['background'],
                # 'paper_bgcolor': colors['background'],
                # 'font':{
                #     'color': colors['text']
                # }
                'yaxis':{ 'title':'Depth (m)','autorange':'reversed'},
                'xaxis':{ 'title':xtitle}
            }
        }

    def getProfilesPlotSingle(self, data, loc_id, day_year, title, max_depth, xtitle=""):
        nonan = np.argmax(data[day_year, loc_id,:])
        if nonan > 0:
            max_depth = np.min([max_depth, nonan])
        return {
            'data': [
                {'x': np.mean(data[:, loc_id, 0:max_depth], axis=0), 'y': self.depths_int[0:max_depth], 'mode': 'lines', 'name': 'Model Mean/STD', 'opacity':0.5,  #  Mean and STD
                 'error_x': dict(type='data', array=np.std(data[:, loc_id, 0:max_depth], axis=0), visible=True), },
                {'x': data[day_year, loc_id, 0:max_depth], 'y': self.depths_int[0:max_depth], 'mode': 'markers', 'name': 'Model'}
            ],
            'layout': {
                'title': F"{title} <br> (all dates)",
                # 'plot_bgcolor': colors['background'],
                # 'paper_bgcolor': colors['background'],
                # 'font':{
                #     'color': colors['text']
                # }
                'yaxis':{ 'title':'Depth (m)','autorange':'reversed'},
                'xaxis':{ 'title':xtitle}
            }
        }
