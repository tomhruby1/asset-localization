import base64
from pathlib import Path

import typing as T
import numpy as np

import pandas as pd
from pyproj import Proj, transform
# import plotly.express as px
# import plotly.graph_objects as go
import dash_leaflet as dl
from dash import Dash

from data_structures import Cluster

ICONS_DIR = 'map/assets/sign_icons'

def utm_to_latlon(easting, northing, zone):
    proj_utm = Proj(proj='utm', zone=zone, ellps='WGS84')
    proj_latlon = Proj(proj='latlong', datum='WGS84')
    lon, lat = transform(proj_utm, proj_latlon, easting, northing)
    return lat, lon

class MapVisualizer:
    def __init__(self, map_center=(50.0755, 14.4378), zoom=12):
        self.map_center = map_center
        self.zoom = zoom

        self.app = Dash(__name__)
        self.app.layout = dl.Map([
            dl.TileLayer(maxZoom=20),
        ], center=map_center, zoom=zoom, maxZoom=20, style={'height': '50vh'})

        
        # load icons
        self.icons = {}
        for icon_p in Path(ICONS_DIR).iterdir():
            with open(icon_p, "rb") as image_file:
                base64_str = base64.b64encode(image_file.read()).decode()
            icon_name = icon_p.stem
            self.icons[icon_name] = dict(
                iconUrl = f'data:image/png;base64,{base64_str}',
                iconSize = [24, 24]
            )


        # with open("/home/tomas/ass-loc/map/assets/sign_icons/a12.png", "rb") as image_file:
        #     base64_str = base64.b64encode(image_file.read()).decode()
        # self.icon = f'data:image/png;base64,{base64_str}'
        # self.icon = dict(
        #     iconUrl = self.icon,
        #     iconSize = [24, 24]
        # )
        print(f"map icons loaded: {self.icons.keys()}")

    def add_clusters(self, clusters: T.List[Cluster], utm_zone=33, coord_sys_translation=(0,0)):
        ''' Plot the cluster means using a scatter mapbox plot
        
        Args:
            - clusters: list of Cluster objects
            - utm_zone: UTM zone of the data
            - coord_sys_translation: will be added to cluster centroids to produce correct easting / northing values   
        
        '''
        
        data = {
            'Easting': [],
            'Northing': [],
            'Label': [],
            'Category': []
        }
        for i, cluster in enumerate(clusters):
            data['Easting'].append(cluster.centroid[0] + coord_sys_translation[0])
            data['Northing'].append(cluster.centroid[1] + coord_sys_translation[1])
            data['Label'].append(str(cluster))
            data['Category'].append(cluster.category)

        df = pd.DataFrame(data)
        df['Latitude'], df['Longitude'] = zip(*df.apply(lambda row: utm_to_latlon(row['Easting'], row['Northing'], utm_zone), axis=1))

        markers = [
            dl.Marker(
                position=[lat, lon],
                children=dl.Tooltip(label),
                icon=self.icons[cat.lower()] if cat.lower() in self.icons else self.icons['default'],
            )
            for lat, lon, label, cat in zip(df['Latitude'], df['Longitude'], df['Label'], df['Category'])
        ]

        self.app.layout.children.append(dl.LayerGroup(markers))

    def add_ground_truth(self, landmarks:dict, utm_zone=33, coord_sys_translation=(0,0)):
        ''' Plot the ground truth landmarks using a scatter mapbox plot
        
        Args:
            - landmarks: dict of landmark_id -> {landmark_info}
        
        '''
        data = {
            'Easting': [],
            'Northing': [],
            'Label': []
        }
        for l_id, landmark in landmarks.items():
            data['Easting'].append(landmark['projected_coord'][0] + coord_sys_translation[0])
            data['Northing'].append(landmark['projected_coord'][1] + coord_sys_translation[1])
            data['Label'].append(f"Landmark {l_id}")

        df = pd.DataFrame(data)
        df['Latitude'], df['Longitude'] = zip(*df.apply(lambda row: utm_to_latlon(row['Easting'], 
                                                                                  row['Northing'], 
                                                                                  utm_zone), axis=1))
        markers = [
            dl.Marker(
                position=[lat, lon],
                children=dl.Tooltip(label),
            )
            for lat, lon, label in zip(df['Latitude'], df['Longitude'], df['Label'])
        ]

        self.app.layout.children.append(dl.LayerGroup(markers))

    def show(self):
        self.app.run_server()


if __name__ == "__main__":
    import dash_leaflet as dl
    from dash import Dash
    import base64


    with open("/home/tomas/Downloads/a_traffic_sign_s.png", "rb") as image_file:
        base64_str = base64.b64encode(image_file.read()).decode()

    icon_url = f'data:image/png;base64,{base64_str}' 

    # Custom icon as per official docs https://leafletjs.com/examples/custom-icons/
    custom_icon = dict(
        iconUrl=icon_url,
        # shadowUrl='https://leafletjs.com/examples/custom-icons/leaf-shadow.png',
        iconSize=[32, 32],
        shadowSize=[50, 64],
        iconAnchor=[22, 94],
        shadowAnchor=[4, 62],
        popupAnchor=[-3, -76]
    )
    # Small example app.
    app = Dash()
    app.layout = dl.Map([
        dl.TileLayer(),
        dl.Marker(position=[55, 10]),
        dl.Marker(position=[57, 10], icon=custom_icon),
    ], center=[56, 10], zoom=6, style={'height': '50vh'})

    app.run_server()

        