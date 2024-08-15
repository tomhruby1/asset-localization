from pathlib import Path
import typing as T
import copy

import numpy as np
import cv2
import rerun as rr
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from data_structures import Ray, Point, RaycastingResult
import config
from trajectory_data import TrajectoryData
from common import get_coco_center, normalize_homcoords, get_midpoint


# (x: right, y: up, z:back) to opencv by negating y,z 
R_to_opencv = np.eye(3)
R_to_opencv[1,1] = -1
R_to_opencv[2,2] = -1

def raycast(cfg:config.Raycasting, camera_transforms:dict, extracted_frames_p:Path, traj:TrajectoryData):
    ''' Perform raycasting + visualization of camera frames / detections given set of fixed camera transforms - 4x4 matrices

        args:
            - cfg: raycasting config
            - cameara_transforms: camera 4x4 pose transform matrices
            - extracted_frames_p: path to (presumably undistorted) data
            - traj: Trajectory Data (containing feature vectors for each of the detection in each image frame)
    '''

    # load data
    # traj = TrajectoryData(detections_p, gpx_p, calib_p)
    imgs = sorted(extracted_frames_p.rglob("*."+cfg.cam_frames_suffix))
    res = (4096, 3008)
    
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
    
    
    # INIT rerun
    COORD_ARROW = rr.Arrows3D(
                                vectors=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                                colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
                            )
    
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP)
    rr.log("world", COORD_ARROW)

    count = 0
    ray_id = 0
    for camera_frame in camera_transforms:
        count += 1
        frame_id = str(int(camera_frame.split('_')[-1]))
        img_name = camera_frame +"."+ cfg.cam_frames_suffix 
        if img_name not in traj.detections: continue
        sensor = camera_frame.split('_')[0]
        if sensor not in cfg.process_sensors: continue

        # visualize only subset of frames
        if cfg.every_nth:
            if frame_id not in frames_to_vis: continue
        if cfg.frames != 'all':
            print(frame_id)
            if frame_id not in cfg.frames and int(frame_id) not in cfg.frames:
                continue

        # get coco detections and load the image
        dets = traj.get_detections(img_name)
        img_p = extracted_frames_p / sensor / img_name
        img = cv2.imread(str(img_p))[:,:,[2,1,0]]

        # rotate 90 bounding boxes
        if dets is not None:
            bboxes_rot = []
            for bbox in dets.bboxes:
                x,y,w,h = bbox
                bboxes_rot.append([
                    res[0] - (y+h), x, h, w # rotating the bounding boxes 90-deg clockwise
                ])
            dets.bboxes = bboxes_rot
        else: continue

        # no visualization for now
        # if UNDISTORT and VIS_IMAGES:
        #     img = undistort_img(img, undistort_map, sensor)


        K, d, principal_point, focal_length = traj.get_sensor_intrinsics(sensor) 
        res = (4096, 3008) #(3008, 4096)  # depends on whether rotated 90 
 
        # T_sensor_world transformation for current camera sensor frame
        current_T = camera_transforms[camera_frame] 

        # TODO: make this separate from logic
        # visualize camera model + detection
        if cfg.visualize_frames:

            frame_entity = "world/frames/frame_"+frame_id
            cam_entity = frame_entity+"/"+sensor
            transform = rr.TranslationAndMat3x3(mat3x3=current_T[:3,:3], 
                                                translation=current_T[:3,3]) 
            rr.log(cam_entity, rr.Transform3D(transform))
            rr.log(cam_entity+"/image", rr.Pinhole(width=res[0], height=res[1], 
                                                   focal_length=focal_length,
                                                   principal_point=principal_point,
                                                   camera_xyz=rr.ViewCoordinates.RDF # correspond to open-CV x:right, y:down, z:forward
                                                ))
            if cfg.visualize_images:
                rr.log(cam_entity+"/image", rr.Image(img))
            
            if dets:
                rr.log(cam_entity+"/image", rr.Boxes2D(array=np.array(dets.bboxes), # np.array([1000,10,100,300])
                                                       array_format=rr.Box2DFormat.XYWH,
                                                       class_ids=dets.class_ids,
                                                       labels=dets.class_names
                                                    ))
        # RAYCASTING
        # OPENCV undistort and get point in 3D coords relative to the camera
        x_det_dist = np.asarray([get_coco_center(bbox) for bbox in dets.bboxes])
        if len(x_det_dist) > 0:
            x_det_dist = np.expand_dims(x_det_dist, axis=1) # expand dim for fisheye.undistortPoints to work
            # if cfg.undistort:
                # x_undist = cv2.fisheye.undistortPoints(x_det_dist, K, d)
            # else:
            x_undist = cv2.undistortPoints(x_det_dist, K, d) # not sure what is the meaning of distortion coeffs. here 
            
            # add 1 to get 3d homcoords.
            x_3d = np.asarray([np.append(p[0], [1,1]) for p in x_undist]) # dim(x_det_dist) = (7,1,2)
            x_3d_transf = [normalize_homcoords(current_T @ x) for x in x_3d] # point on ray in global coordinate frame
            
            # if cfg.visualize_frames: 
            rr.log(f"world/projected_points/p_{count}", rr.Points3D(x_3d_transf, radii=0.02))
            
            # CAST A RAY
            ray_origin = current_T[:3, 3] # ray origin target sensor's camera center
            rays = []
            ray_labels = []
            for i, x in enumerate(x_3d_transf):
                ray_dir = x - ray_origin # np.abs() ?  # get ray direction
                ray_label = f"{frame_id}_{sensor}_{i}" 

                # flatten?
                # if cfg.flatten_rays:
                #     ray_origin[2] = 0
                #     ray_dir[2] = 0

                if dets.global_instances is None:
                    ray = Ray(ray_id, frame_id, sensor, dets.class_ids[i], dets.class_names[i], dets.scores[i], 
                              dets.bboxes[i], ray_origin, ray_dir)
                # add ground truth if available
                else:
                    ray = Ray(ray_id, frame_id, sensor, dets.class_ids[i], dets.class_names[i], dets.scores[i], 
                              dets.bboxes[i], ray_origin, ray_dir, global_instance=dets.global_instances[i])                    
                ray.cls_feature = dets.feature_vectors[i]
                
                ray_id += 1   
                rays.append(ray)
                all_rays.append(ray)
                ray_labels.append(ray_label)
                all_ray_labels.append(ray_label)
                rr.log("world/rays/"+ray_label, rr.Arrows3D(origins=ray_origin, vectors=ray_dir))

            frame_rays[frame_id] = rays
            frame_ray_labels[frame_id] = ray_labels
    print(f"Ray projecting finished; {len(all_rays)} rays created")
    
    print("Building midpoints...")
    ## Midpoint stuff
    midpoints:T.List[Point] = []
    for i, ray1 in enumerate(tqdm(all_rays)):
        for j, ray2 in enumerate(all_rays):
            if i != j and ray1.frame_id != ray2.frame_id: 
                midpoint, dist, l1, l2 = get_midpoint(ray1, ray2, l=True)
                if l1 > 0 and l2 > 0:
                    midpoints.append(Point(ray1, ray2, dist, midpoint, (l1,l2)))

    result = RaycastingResult(midpoints, traj, all_rays)
 
    return result

def get_camera_transforms(cams_p:Path, sensors) -> T.Union[T.Dict[str, np.ndarray], np.ndarray]:
    ''' get camera transformation matrices given opk exported file.
        Already projected coordinets expected. Coordinate system origin is moved to the first camera frame
        TODO: take care of projection, datums, etc?
        
        args:
            - cams_p: path to the exported camera poses
            - sensor: list of camera sensors to process
        return: (dict of camera_frame -> 4x4 transformation matrix, origin)

    '''
    cams_p = Path(cams_p)
    camera_frames = {} # parsed data from file
    with open(Path(cams_p)) as f:
        for line in f:
            try:
                line_split = line.split("\t")
                frame_id, x, y, z, omega, phi, kappa, r11, r12, r13, r21, r22, r23, r31, r32, r33 = line_split
                camera_frames[frame_id] = {
                    'location': np.array((float(x),float(y),float(z))),
                    'opk': np.array((float(omega), float(phi), float(kappa))),
                    'rotmat': np.array([
                        [float(r11), float(r12), float(r13)],
                        [float(r21), float(r22), float(r23)],
                        [float(r31), float(r32), float(r33)]
                    ])
                }
            except Exception as ex:
                print(ex)

    origin = camera_frames[list(camera_frames.keys())[0]]['location']
    print("coordinate system origin set to:", origin)
    
    camera_transforms = {} # camera_frame -> 4x4 matrix
    frame_ids = np.unique([cam_frame.split('_')[-1] for cam_frame in camera_frames]) # whole rig frame first
    for fid in frame_ids: # also order by sensor
        for sensor in sensors:
            cam_frame_lbl = sensor + "_frame_" + fid
            pose = camera_frames[cam_frame_lbl]
            pose['location'] = pose['location'] - origin # normalize location
                
            T_rotmat = np.eye(4)
            # THIS!! -> negate Z axis in the image frame
            # https://s3.amazonaws.com/mics.pix4d.com/KB/documents/Pix4D_Yaw_Pitch_Roll_Omega_to_Phi_Kappa_angles_and_conversion.pdf
            # Z in image coordinate system: camera back (opposite to viewing direction through camera)
            # T_rotmat[:3,:3] = np.linalg.inv(R_neg_z @ pose['rotmat']) # inverse gets the results as with building the matrix from the opk; that should be world->image
            T_rotmat[:3,:3] = np.linalg.inv(R_to_opencv @ pose['rotmat'])
            T_rotmat[:3, 3] = pose['location']
            camera_transforms[cam_frame_lbl] = T_rotmat

    return camera_transforms, origin


if __name__=='__main__':
    cams_p = '/media/tomas/samQVO_4TB_D/assdet-experiments/drtinova_u_new/cameras_exported_local.txt'
    
    cams_p = '/media/tomas/samQVO_4TB_D/assdet-experiments/drtinova_u_new/cameras_exported_projected_wgs84_utm_33n.txt'
    extracted_frames_dir = '/media/tomas/samQVO_4TB_D/assdet-experiments/drtinova_u_new/reel_undistorted'
    detections_p = '/media/tomas/samQVO_4TB_D/assdet-experiments/drtinova_u_new/detections.json'

    cam_transfs = get_camera_transforms(cams_p)
    result = raycast(cam_transfs, Path(extracted_frames_dir), Path(detections_p))

    print("done")