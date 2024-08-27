from pyproj import CRS, Transformer, proj
import gpxpy
from pathlib import Path
import typing as T
import numpy as np
import pandas as pd
import json
from scipy.spatial.transform import Rotation


from common import load_calibration, SENSORS
from data_structures import FrameDetections

def convert_gpx_to_df(file, filename):
    '''stolen and modified from the sfms repo'''
    gpx = gpxpy.parse(file)
    track_counter = 0
    # (1)make DataFrame
    columns = ['Longitude', 'Latitude', 'Elevation', 'Point', 'Track_Origin', 'Filename', 'Frame', 'Roll', 'Pitch', 'Heading']
    track_data = []
    for track in gpx.tracks:
        for segment in track.segments:
            # Load the data into a Pandas dataframe (by way of a list)
            for point_idx, point in enumerate(segment.points):
                #Loaf frame extension into df as well
                frame = -1
                if(len(point.extensions) > 0 and point.extensions != None):
                    for ext in point.extensions:
                        if(ext.tag == 'frame'):     
                            frame = ext.text
                        if(ext.tag == 'roll'):     
                            roll = ext.text
                        if(ext.tag  == 'pitch'):     
                            pitch = ext.text
                        if(ext.tag == 'heading'):     
                            heading = ext.text
                
                track_data.append([point.longitude, point.latitude, point.elevation, 
                                   point, track_counter, filename, frame, roll, pitch, heading])
            track_counter += 1
    gpx_df = pd.DataFrame(track_data, columns=columns)
    return gpx_df


def get_trajectory(gpx_file_p, origin:np.ndarray=None, visualize=False) -> T.Tuple[pd.DataFrame, proj.Proj]:
    '''
    Load gpx, convert to pandas df, project GNSS coords to plane.
    Planar coodinates 'x', 'y', 'z' set such that origin of the coordinate frame is placed at the first trackpoint
    'x' and 'y' projected, 'z' is just elevation normalized by the first track point elevation
    
    args:
        - gpx_file_p: path to gpx file
        - origin: origin of the new coordinate frame, if None, the first trackpoint is used
    
    returns: 
        - resulting dataframe 
        - pyproj.proj projection
        
    '''
    with open(gpx_file_p) as f:
        gpx_df = convert_gpx_to_df(f,"reel_gpx")

    wgs84 = CRS.from_string("EPSG:4326") # LatLon with WGS84 datum used by GPS units and Google Earth 


    # if origin is not None:
    #     lat_0, lon_0, ele_0 = origin
    # else:
    # ^^^ THIS is in northing/easting but we need lat/long for GPX file
    
    lat_0 = gpx_df.iloc[0].Latitude
    lon_0 = gpx_df.iloc[0].Longitude
    ele_0 = gpx_df.iloc[0].Elevation # elevation zero 

    # https://gis.stackexchange.com/questions/330746/generating-a-custom-flat-projection-in-pyproj 
    myproj = proj.Proj(proj="aeqd", lat_0=lat_0, lon_0=lon_0, datum="WGS84", units="m")
    transformer = Transformer.from_proj(wgs84, myproj, always_xy=True) # init according to the first cam0?

    # out_arr= np.zeros([gpx_df.s]) let's keep it pandas
    def transform_project_df(row):
        cartesian_point = transformer.transform(row.Longitude, row.Latitude)
        return pd.Series([cartesian_point[0], cartesian_point[1], row.Elevation - ele_0], index=['x', 'y', 'z'])
        

    # xs = []; ys=[]
    # for idx, row in gpx_df.iterrows():
    #     cartesian_point = transformer.transform(row.Longitude, row.Latitude)
    #     xs.append(cartesian_point[0])
    #     ys.append(cartesian_point[1])
    #     print(f"cartesian_point: {cartesian_point}")

    gpx_df[['x', 'y', 'z']] = gpx_df.apply(transform_project_df, axis=1)

    return gpx_df, myproj

class TrajectoryData:
    '''Data container for both the gpx GNSS/XVN data and detections'''
    
    def __init__(self, detections_p:Path, gpx_p:Path, calibration_p:Path, landmark_gnss_p:Path=None, 
                 geo_coord_sys=None, geo_coord_sys_origin=None, detections_features:dict=None):
        '''
        args:
            - detections_p: path to detections json file
            - gpx_p: path to gpx file
            - calibration_p: path to calibration file
            - landmark_gnss_p: path to csv with ground truth landmarks
            - geo_coord_sys: coordinate system for the projected trajector
            - detection_features: dictionary containing list of feature vectors for each image
        '''
        self.detections_p = Path(detections_p)
        self.gpx_p = Path(gpx_p)
        self.calib_p = Path(calibration_p)
        if landmark_gnss_p is not None: self.landmark_gnss_p = Path(landmark_gnss_p)
        
        self.Ks, self.Ds, self.Ts, self.resolution, self.calib_chain = \
            load_calibration(calibration_p, return_chain=True)
        

        with open(detections_p, 'r') as f:
            self.detections = json.load(f)

        self.gpx_df, self.coord_proj = get_trajectory(gpx_p, origin=geo_coord_sys_origin)
        self.frames = self.gpx_df.Frame.values # list of string frame ids

        datum = CRS.from_string("EPSG:4326")
        self.coord_transformer = Transformer.from_proj(datum, self.coord_proj, always_xy=True)

        # build frame_id -> sensor -> img_name mapping 
        self.frame_image_map = {}
        for img_name in self.detections:
            sensor, _, frame_id = Path(img_name).stem.split('_')
            frame_id = str(int(frame_id)) # remove the zeros

            if frame_id not in self.frame_image_map:
                self.frame_image_map[frame_id] = {
                    cam: '' for cam in SENSORS           
                }
            self.frame_image_map[frame_id][sensor] = img_name

        self.detections_features = detections_features
        
        # load landmark csv
        self.landmarks = None
        if landmark_gnss_p:
            self.landmarks = {}
            gnss_df = pd.read_csv(self.landmark_gnss_p)
            for idx, row in gnss_df.iterrows():
                if row.Name in self.landmarks:
                    raise Exception(f"Multiple instances of the same ground truth landmark label {row.Name}")
                self.landmarks[row.Name] = {
                    'id': row.Name,
                    'class': row.Name.split('_')[0],
                    'latitude': row.Latitude,
                    'longitude': row.Longitude,
                    # 'projected_coord': self.coord_transformer.transform(row.Longitude, row.Latitude)
                }
                if geo_coord_sys_origin is not None:
                    self.landmarks[row.Name]['projected_coord'] = (row.Easting - geo_coord_sys_origin[0], 
                                                                   row.Northing - geo_coord_sys_origin[1],
                                                                   row.Elevation - geo_coord_sys_origin[2])
                else:
                    self.landmarks[row.Name]['projected_coord'] = (row.Easting, row.Northing)

    def get_frame_image_names(self, frame_id) -> dict:
        '''get image names from integer frame id'''
        return self.frame_image_map[frame_id]
    
    def get_sensor_intrinsics(self, sensor:str):
        '''returns: calibration matrix, distortion coeffs, principal_point, focal_length'''
        K = self.Ks[sensor]
        principal_point = (K[0,2], K[1,2]) 
        focal_length = (K[0,0], K[1,1]) 

        return K, self.Ds[sensor], principal_point, focal_length

    def get_detections(self, image_name:str) -> FrameDetections:
        dets = self.detections[image_name]
        bboxes = [det['bbox'] for det in dets]
        class_names = [det['category_name'] for det in dets]
        class_ids = [det['category_id'] for det in dets]
        scores = [det['score'] for det in dets]
        feats = [np.array(det['feature']) for det in dets]

        global_instances = None
        if len(dets) > 0:
            global_instances = None
            coords = None
            
            # add ground truth if available
            if 'global_instance' in dets[0]:
                global_instances = [det['global_instance'] for det in dets]
            
            if 'latitude' in dets[0] and 'longitude' in dets[0]:
                coords = [(det['latitude'], det['longitude']) for det in dets] # lat, long

            return FrameDetections(bboxes, class_names, class_ids, scores, feats,
                                   global_instances=global_instances, coordinates=coords) 
        else:
            print(f"No detections for {image_name}")
            return None

    def get_ground_truth(self):
        '''
        If data contains ground truth - global instances information

        return:
            - landmarks - if self.landmarks loaded from gnss csv
            - gt_instances: ground truth global instance labels
            
            ?also: 
            - observations: corresponding detections from different camera frames??
        '''
        if self.landmarks:
            return self.landmarks

        # if ground truth gnss not set up
        gt_instances = []
        
        for frame_id in self.frames:
            for sensor, img_name in self.get_frame_image_names(frame_id).items():
                for det in self.detections[img_name]:
                    if 'global_instance' in det:
                        gt_label = det['global_instance']
                        if gt_label not in gt_instances:
                            gt_instances.append(gt_label)
                    else:
                        print(f"Warning: Insconsistent ground truth data; for {img_name} global instance not found!")
        
        return gt_instances
    
    def get_gnss_T(self, frame_id:int) -> np.ndarray:
        '''
        Get single frame GNSS/XVN transformation.

        args: - frame_id: integer frame_id number such as '2526'
        
        returns: T_gnss_world - 4x4 transformation matrix 
        '''
        # frame id from image name: str(int(img_p.stem.split('_')[-1]))

        frame_id = str(frame_id)
        frame_record = self.gpx_df[self.gpx_df.Frame==frame_id]
        
        # TODO: be sure about the negative first value
        hpr = (-float(frame_record.Heading.values),
               float(frame_record.Pitch.values), 
               float(frame_record.Roll.values))
        # build rotation matrix out of heading-pitch-roll
        R_gnss_world = Rotation.from_euler('ZYX', hpr, degrees=True).as_matrix()
        T_gnss_world = np.eye(4)
        T_gnss_world[:3,:3] = R_gnss_world
        T_gnss_world[:3, 3] = np.array([float(frame_record.x.values), 
                                        float(frame_record.y.values), 
                                        float(frame_record.z.values)]) 
        
        return T_gnss_world
        
    def get_sensor_T(self, sensor:str) -> np.ndarray:
        '''
        return T_sensor_cam0: 4x4 transformation matrix from cam_n to cam_0 
        obrationed from calibration file
        '''
        # sensor tranformation from calibration file
        T_sensor_cam0 = self.Ts[sensor]
        T_sensor_cam0[3,3] = 1 #replace bottom right zero
        
        return T_sensor_cam0
    
    def get_trajectory(self, ax=None) -> T.Tuple[np.ndarray, np.ndarray]:
        '''
        Get the trajectory in the form of (x,y), (roll, pitch, heading)
        
        args: 
            - ax: matplotlib axes, if visualization will be triggered
        '''
        if ax:  ax.set_aspect('equal', adjustable='box')
        xs = []; ys=[]; hs=[]; ps=[]; rs=[]

        if ax: ax.grid(visible=True)

        for idx, row in self.gpx_df.iterrows():
            xs.append(row.x)
            ys.append(row.y)
            hs.append(float(row.Heading))
            ps.append(float(row.Pitch))
            rs.append(float(row.Roll))
        if ax: ax.scatter(xs, ys, color='green', alpha=0.3)
        for xi, yi, heading in zip(xs, ys, hs):
            heading = float(heading) + 180 # switch to opposite direction # datasets used with XVN mounted the opposite
            arrow_length = 0.2
            arrow_dx = arrow_length * np.sin(np.radians(heading))
            arrow_dy = arrow_length * np.cos(np.radians(heading))
            if ax: ax.arrow(xi, yi, arrow_dx, arrow_dy, head_width=0.05, head_length=0.1, fc='green', ec='green')
        
        if ax:
            ax.set_xlabel("x (easting)")
            ax.set_ylabel("y (northing)")
            # ax.set_title("XVN pose - x,y,heading")

        # also save the processed trajectory to data structure # TODO: probably do always
        traj_xy = np.array([xs, ys]).T
        traj_hpr = np.array([hs, ps, rs]).T

        return traj_xy, traj_hpr