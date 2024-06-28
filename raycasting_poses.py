from pathlib import Path
import typing as T

import numpy as np
import cv2
import rerun as rr

from common import TrajectoryData
from visualize_cameras import visualize_cams_rerun

VISUALIZATION = True
FRAMES = None #  {'502','552', '602', '702'}
EVERY_NTH_FRAME = 10 # if FRAMES not None should be 1

SENSORS = ['cam0', 'cam1', 'cam2', 'cam3', 'cam5']
# SENSORS = ['cam0', 'cam3']
SUFFIX = ".jpg"

def raycast(camera_transforms:dict, extracted_frames_p:Path, detections_p:Path,
            calib_p:Path=None, gpx_p:Path=None):
    ''' Perform raycasting + visualization of camera frames / detections'''
    # load data
    if calib_p is None: calib_p = extracted_frames_p/'calib.yaml'
    if gpx_p is None: gpx_p = extracted_frames_p/'extracted.gpx'

    traj = TrajectoryData(detections_p, gpx_p, calib_p)
    imgs = sorted(extracted_frames_p.rglob("*"+SUFFIX))
    
    # create list of frames to log if every n-th
    frames_to_vis = None
    if EVERY_NTH_FRAME:
        frames_to_vis = []
        count = 0 
        for img_p in imgs:
            if 'cam0' in img_p.stem:
                if count % EVERY_NTH_FRAME == 0:
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
    rr.init('raycasting', spawn=True)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP)
    rr.log("world", COORD_ARROW)

    pose_label_list=[] # just to debug

    count = 0
    vbray_id = 0
    for camera_frame in camera_transforms:
        count += 1
        frame_id = str(int(camera_frame.split('_')[-1]))
        img_name = camera_frame + SUFFIX
        if img_name not in traj.detections: continue
        sensor = camera_frame.split('_')[0]
        if sensor not in SENSORS: continue

        # visualize only subset of frames
        if EVERY_NTH_FRAME:
            if frame_id not in frames_to_vis: continue
        if FRAMES:
            if frame_id not in FRAMES: continue

        # get coco detections and load the image
        dets = traj.get_detections(img_name)
        img_p = extracted_frames_p / sensor / img_name
        img = cv2.imread(str(img_p))[:,:,[2,1,0]]
        
        # no visualization for now
        # if UNDISTORT and VIS_IMAGES:
        #     img = undistort_img(img, undistort_map, sensor)

        # img = np.rot90(img) # TODO: true only for MX I think -- then another T_rotate_90 needs to be added to the transform chain

        T_gnss_world = traj.get_gnss_T(frame_id)
        T_camn_cam0 = traj.get_sensor_T(sensor)

        K, d, principal_point, focal_length = traj.get_sensor_intrinsics(sensor)
        res = (4096, 3008) #(3008, 4096)  # depends on whether rotated 90 
 
        # T_sensor_world transformation for current camera sensor frame
        current_T = camera_transforms[camera_frame]   # T_gnss_world @ T_cam0_gnss @ T_camn_cam0 @ rotate_90_clockwise_mat_T # @ rotate_90_clockwise_mat_T when rotated prior to fit the detections

        # TODO: make this separate from logic
        # visualize camera model + detection
        if VISUALIZATION:

            frame_entity = "world/frames/frame_"+frame_id
            cam_entity = frame_entity+"/"+sensor
            transform = rr.TranslationAndMat3x3(mat3x3=current_T[:3,:3], 
                                                translation=current_T[:3,3]) 
            rr.log(cam_entity, rr.Transform3D(transform))
            rr.log(cam_entity+"/image", rr.Pinhole(width=res[0], height=res[1], 
                                                   focal_length=focal_length,
                                                   principal_point=principal_point,
                                                   camera_xyz=rr.ViewCoordinates.RUB
                                                ))
            # if VIS_IMAGES:
            rr.log(cam_entity+"/image", rr.Image(img))
            
            if dets:
                rr.log(cam_entity+"/image", rr.Boxes2D(array=np.array(dets.bboxes), 
                                                       array_format=rr.Box2DFormat.XYWH,
                                                       class_ids=dets.class_ids,
                                                       labels=dets.class_names
                                                    ))
            

def get_camera_transforms(cams_p:Path) -> T.Dict[str, np.ndarray]:
    ''' get camera transformation matrices given metashape opk exported file.
        Already projected coordinets expected. 
        TODO: take care of projection, datums, etc...
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
    print(origin)
    
    camera_transforms = {} # camera_frame -> 4x4 matrix
    frame_ids = np.unique([cam_frame.split('_')[-1] for cam_frame in camera_frames]) # whole rig frame first
    for fid in frame_ids: # also order by sensor
        for sensor in SENSORS:
            cam_frame_lbl = sensor + "_frame_" + fid
            pose = camera_frames[cam_frame_lbl]
            pose['location'] = pose['location'] - origin # normalize location
                
            T_rotmat = np.eye(4)
            # THIS!! -> negate Z axis in the image frame
            # https://s3.amazonaws.com/mics.pix4d.com/KB/documents/Pix4D_Yaw_Pitch_Roll_Omega_to_Phi_Kappa_angles_and_conversion.pdf
            # Z in image coordinate system: camera back (opposite to viewing direction through camera)
            # T_rotmat[:3,:3] = np.linalg.inv(R_neg_z @ pose['rotmat']) # inverse gets the results as with building the matrix from the opk; that should be world->image
            T_rotmat[:3,:3] = np.linalg.inv(pose['rotmat'])
            T_rotmat[:3, 3] = pose['location']
            camera_transforms[cam_frame_lbl] = T_rotmat

    return camera_transforms


if __name__=='__main__':
    cams_p = '/media/tomas/samQVO_4TB_D/assdet-experiments/drtinova_u_new/cameras_exported_local.txt'
    cams_p = '/media/tomas/samQVO_4TB_D/assdet-experiments/drtinova_u_new/cameras_exported_projected_wgs84_utm_33n.txt'
    extracted_frames_dir = '/media/tomas/samQVO_4TB_D/assdet-experiments/drtinova_u_new/reel_undistorted'
    detections_p = '/media/tomas/samQVO_4TB_D/assdet-experiments/drtinova_u_new/detections.json'

    cam_transfs = get_camera_transforms(cams_p)
    raycast(cam_transfs, Path(extracted_frames_dir), Path(detections_p))

    print("done")