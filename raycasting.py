## cluster rays based on euklidean distance and semantic information

import numpy as np
import cv2
import json
from scipy.spatial.transform import Rotation
from pathlib import Path

import rerun as rr

import config
from common import normalize_homcoords, get_coco_center, get_midpoint
# TODO: get back undistort imags but rename
# from undistort_images import get_undistort_map, undistort_img
from data_structures import Ray, Point, RaycastingResult
# from rr_visualization import visualize_rays
from trajectory_data import TrajectoryData

# CAMERA CALIBRATION 
# Metashape offset when orientation accuracy set to [0.0001, 0.0001, 0.0001]
X = -2.32089
Y = 1.86682
Z = -1.53331
YAW = 4.44562
PITCH = 50.8229
ROLL = 97.4261
# Ilias Camera to xvn
def get_reference_change() -> np.array:
    # y, p, r = 0, 180, 0
    # R = Rotation.from_euler('ZYX', np.array([y, p, r]), degrees=True).as_matrix()
    y, p, r = 0, 180, 0
    R1 = Rotation.from_euler('ZYX', np.array([y, p, r]), degrees=True).as_matrix()
    
    y, p, r = 90, 0, 0
    R2 = Rotation.from_euler('ZYX', np.array([y, p, r]), degrees=True).as_matrix()
 
    R = R2 @ R1
    # R = R1
    T = np.eye(4)
    T[:3,:3] = R.copy()
    return T
# Metashape calibration offset: T_gnss_cam0
R_cam0_gnss = Rotation.from_euler('ZYX', np.array([YAW, PITCH, ROLL]), degrees=True).as_matrix()
T_cam0_gnss = np.eye(4)
T_cam0_gnss[:3,:3] = R_cam0_gnss.copy()
T_cam0_gnss[:3, 3] = np.array([X,Y,Z])
# experimentally guessed heading fix to align cam with the vehicle
R_heading_fix = Rotation.from_euler('Y', 15, degrees=True).as_matrix()
T_heading_fix = np.eye(4)
T_heading_fix[:3,:3] = R_heading_fix
T_cam0_gnss = np.linalg.inv(T_cam0_gnss) @ T_heading_fix @ get_reference_change() 

# to unrotate the mx camera to align with detections
rotate_90_clockwise_mat = Rotation.from_euler('z', 90, degrees=True).as_matrix()
rotate_90_clockwise_mat_T = np.eye(4)
rotate_90_clockwise_mat_T[:3,:3] = rotate_90_clockwise_mat


# SUFFIX = "jpg"
# PROCESS_SENSORS = {"cam0", "cam1", "cam2", "cam3", "cam5"}  # {"cam2", "cam3"} 

# # FRAMES = {'2469', '2474', '2499'}
# # tiny
# # FRAMES = {'2469', '2470', '2471', '2472', '2473'} 
# # small
# # FRAMES = {'2469', '2470', '2471', '2472', '2473', '2474', '2475', '2476', '2477', '2478'}
# # small2
# # FRAMES = {'2505', '2506', '2507', '2508', '2509', '2510', '2511', '2512'}
# # FRAMES = {'2469', '2474', '2478', '2482', '2486', '2490'}       
# # FRAMES = {'2469', '2472', '2475', '2507', '2510', '2513'}
# # med
# # FRAMES = {'2469', '2470', '2471', '2472', '2473', '2474', '2475', '2476', '2477', '2478', 
#         #   '2507', '2508', '2509', '2510', '2511', '2512', '2513', '2514', '2515', '2516'}
# FRAMES = None

# EVERY_NTH_FRAME = 1 # if FRAMES not None should be 1
# UNDISTORT = False # Already undistorted?

# VIS_MIDPOINT_ORIGIN_DISTANCE_THRESHOLD = 200  
# VIS_FRAMES = False
# VIS_IMAGES = False
# VIS_MIDPOINTS = False

# MIDPOINT_MAX_DIST_FROM_CAM = 50 # max distance for midpoint to be considered from camera

# RAYS_2D = False
# # midpoint visualization thresholds
# MIDPOINT_RAY_THRESHOLD = 10 # keep only midpoints for rays, where the distance of the closest points is smaller than this
# MIDPOINT_ORIGIN_DISTANCE_THRESHOLD = 500   # outliers for from origin of the coordinate system
# if RAYS_2D: MIDPOINT_ORIGIN_DISTANCE_THRESHOLD = np.cbrt(MIDPOINT_ORIGIN_DISTANCE_THRESHOLD)**2
     
def main(cfg:config.Raycasting, extracted_frames_p:Path, detections_p:Path, calib_p:Path=None, 
         gpx_p:Path=None, export_p:Path=None):

    # load data
    if calib_p is None: calib_p = extracted_frames_p/'calib.yaml'
    if gpx_p is None: gpx_p = extracted_frames_p/'extracted.gpx'
    
    traj = TrajectoryData(detections_p, gpx_p, calib_p)
    traj.T_cam0_gnss = T_cam0_gnss
    traj.T_rotate_90_clockwise = rotate_90_clockwise_mat_T

    
    imgs = sorted(extracted_frames_p.rglob("*."+cfg.cam_frames_suffix))

    traj_xy, traj_hpr = traj.get_trajectory()

    # create list of frames to log if every n-th
    frames_to_vis = None
    if cfg.every_nth:
        frames_to_vis = []
        count = 0 
        for img_p in imgs:
            if 'cam0' in img_p.stem:
                if count % cfg.every_nth == 0:
                    frame_id = str(int(img_p.stem.split('_')[-1]))
                    frames_to_vis.append(frame_id)
                count += 1

    frame_rays = {} # dir of (origin, direction point) tuples for rays, indexed by frameid
    frame_ray_labels = {} # description labels for the 
    all_rays, all_ray_labels = [], []

    count = 0
    ray_id = 0
    for img_p in imgs:
        count += 1
        frame_id = str(int(img_p.stem.split('_')[-1])) # Frame in df as str
        img_label = img_p.name # maybe .name here? 
        if img_label not in traj.detections: continue
        sensor = img_p.stem.split('_')[0]
        if sensor not in cfg.process_sensors: continue

        # visualize only subset of frames
        if cfg.every_nth:
            if frame_id not in frames_to_vis: continue
        if cfg.frames != 'all':
            if frame_id not in cfg.frames: continue

        # get coco detections and load the image
        dets = traj.get_detections(img_label)
        img = cv2.imread(str(img_p))[:,:,[2,1,0]]
        img = np.rot90(img) # TODO: true only for MX I think

        T_gnss_world = traj.get_gnss_T(frame_id)
        T_sensor_cam0 = traj.get_sensor_T(sensor)

        K, d, principal_point, focal_length = traj.get_sensor_intrinsics(sensor)
        res = (3008, 4096) # (4096, 3008) # depends on whether rotated 90 

        # T_sensor_world transformation for current camera sensor frame
        current_T = T_gnss_world @ T_cam0_gnss @ T_sensor_cam0 @ rotate_90_clockwise_mat_T # @ rotate_90_clockwise_mat_T when rotated prior to fit the detections

        if cfg.visualize_frames:
            frame_entity = "world/frames/frame_"+frame_id
            cam_entity = frame_entity+"/"+sensor
            transform = rr.TranslationAndMat3x3(mat3x3=current_T[:3,:3], 
                                            translation=current_T[:3,3]) 
            rr.log(cam_entity, rr.Transform3D(transform))
            rr.log(cam_entity+"/image", rr.Pinhole(width=res[0], height=res[1], 
                                                focal_length=focal_length,
                                                principal_point=principal_point,
                                                ))
            if cfg.visualize_frames and cfg.visualize_images:
                rr.log(cam_entity+"/image", rr.Image(img))
            if dets:
                rr.log(cam_entity+"/image", rr.Boxes2D(array=np.array(dets.bboxes), 
                                                       array_format=rr.Box2DFormat.XYWH,
                                                       class_ids=dets.class_ids,
                                                       labels=dets.class_names
                                                    ))
        if dets is None:
            continue
                
        # OPENCV undistort and get point in 3D coords relative to the camera
        x_det_dist = np.asarray([get_coco_center(bbox) for bbox in dets.bboxes])
        if len(x_det_dist) > 0:
            x_det_dist = np.expand_dims(x_det_dist, axis=1) # expand dim for fisheye.undistortPoints to work
            if cfg.undistort:
                x_undist = cv2.fisheye.undistortPoints(x_det_dist, K, d)
            else:
                x_undist = cv2.undistortPoints(x_det_dist, K, d) # not sure what is the meaning of distortion coeffs. here 
            
            # add 1 to get 3d homcoords.
            x_3d = np.asarray([np.append(p[0], [1,1]) for p in x_undist]) # dim(x_det_dist) = (7,1,2)
            x_3d_transf = [normalize_homcoords(current_T @ x) for x in x_3d] # point on ray in global coordinate frame
            
            if cfg.visualize_frames: 
                rr.log(f"world/projected_points/p_{count}", rr.Points3D(x_3d_transf, radii=0.02))

            # now rays in 3d
            # ray origin target sensor's camera center
            ray_origin = current_T[:3, 3]
            rays = []
            ray_labels = []
            for i, x in enumerate(x_3d_transf):
                ray_dir = x - ray_origin # np.abs() ?  # get ray direction
                ray_label = f"{frame_id}_{sensor}_{i}" 

                # flatten?
                if cfg.flatten_rays:
                    ray_origin[2] = 0
                    ray_dir[2] = 0

                if dets.global_instances is None:
                    ray = Ray(ray_id, frame_id, sensor, dets.class_ids[i], dets.class_names[i], dets.scores[i], 
                              dets.bboxes[i], ray_origin, ray_dir)
                # add ground truth if available
                else:
                    ray = Ray(ray_id, frame_id, sensor, dets.class_ids[i], dets.class_names[i], dets.scores[i], 
                              dets.bboxes[i], ray_origin, ray_dir, global_instance=dets.global_instances[i])                    
                ray_id += 1   
                rays.append(ray)
                all_rays.append(ray)
                ray_labels.append(ray_label)
                all_ray_labels.append(ray_label)
            frame_rays[frame_id] = rays
            frame_ray_labels[frame_id] = ray_labels
            
    
    # visualize_rays(all_rays, descriptions=all_ray_labels)

    ## Midpoint stuff
    midpoints = [] # list of Midpoint instances
    # midpoint_points = [] # plain list of points for geometry
    # midpoint_labels = [] # labels for rr points log
    for idx_f1, f_id1 in enumerate(frame_rays):
        for idx_f2, f_id2 in enumerate(frame_rays):
            if idx_f1 != idx_f2: # f_id1 != f_id2: one object might be seen in one frame 2 sensors
                # rays from different frames with each other
                for i, ray1 in enumerate(frame_rays[f_id1]):
                    for j, ray2 in enumerate(frame_rays[f_id2]):
                        midpoint, dist, l1, l2 = get_midpoint(ray1, ray2, l=True)
                        
                        # just for visualization
                        # if dist < MIDPOINT_RAY_THRESHOLD and np.linalg.norm(midpoint) < MIDPOINT_ORIGIN_DISTANCE_THRESHOLD:
                        #     midpoint_points.append(midpoint)
                        #     midpoint_labels.append(f"midpoint: {frame_ray_labels[f_id1][i]}x{frame_ray_labels[f_id2][j]}")

                        # first basic midpoint filtering done HERE 
                        # TODO: make it more formal
                        # use l1 and l2 for distance from midpoint
                        if l1 > 0 and l2 > 0:
                            midpoint = Point(ray1, ray2, dist, midpoint, (l1,l2))
                            midpoints.append(midpoint)
                        # else:
                        #     print("midpoint removed based on the cheirality constraint")

    result = RaycastingResult(midpoints, traj, all_rays)
 
    return result


if __name__=='__main__':    
    extracted_p = "/media/tomas/samQVO_4TB_D/asset-detection-datasets/drtinova_small_u_track/data/reel_0050_20231107-122212_prague_MX_XVN"
    # detections_p = "/media/tomas/samQVO_4TB_D/asset-detection-datasets/drtinova_small_u_track/detections/detections.json"
    detections_p = "/media/tomas/samQVO_4TB_D/asset-detection-datasets/drtinova_small_u_track/annotations.json"
    
    main(Path(extracted_p), Path(detections_p))