# ALL RERUN BASD VSIUALIZATION should be here
import typing as T
import numpy as np
import rerun as rr
import cv2

from common import *
from trajectory_data import TrajectoryData
from data_structures import Ray, Cluster, Point

SOME_COLORS = [
    (0, 63, 92, 255),
    (47, 75, 124, 255),
    (102, 81, 145, 255),
    (160, 81, 149, 255),
    (212, 80, 135, 255),
    (249, 93, 106, 255),
    (255, 124, 67, 255),
    (255, 166, 0, 255),
]


CLASS_COLOR = {
    'a12': (240,0,0),
    'ip': (0,0,240),
    'e12': (240,240,240),
    'b28': (240,0,240),
    'other': (0,240,0)
}

def init_visualization():
    rr.init('reel_triangulation', spawn=True)

def get_class_color(r:Ray, opacity=150):
    for color_k in CLASS_COLOR.keys():
        if color_k in r.class_name.lower():
            return CLASS_COLOR[color_k] + (opacity,)
        
    return CLASS_COLOR['other'] + (opacity,)
            
def visualize_single_midpoint_cluster(cluster:Cluster, entity:str, color=(128,128,0,0), radii=0.3, 
                                      show_cluster_label=True, centroid_only=False):
    if not centroid_only:
        points3d = [p.point for p in cluster.points]
        point_labels = [str(p) for p in cluster.points]
        rr.log("world/assets/clusters/cl-"+entity,
            rr.Points3D(points3d, labels=point_labels, colors=[color]*len(points3d), radii=0.1))
    else:
        color = (255,0,0,255)
    
    if cluster.centroid is not None:
        cluster_label = None
        if show_cluster_label:
            cluster_label = str(cluster)
        rr.log("world/assets/clusters/"+entity+"/centroid",
               rr.Points3D(cluster.centroid, labels=cluster_label, colors=color, radii=0.2))
    else:
        print("Cluster doesn't have a centroid")


def visualize_midpoint_clusters(clusters:T.List[Cluster], show_centroids=True):
    points = []
    point_labels = []
    cluster_ids = []    
    centroids = []
    cluster_labels = []
    for i, cluster in enumerate(clusters): 
        # visualize_single_midpoint_cluster(cluster, str(i), color=SOME_COLORS[i%len(SOME_COLORS)], 
        #                                   show_cluster_label=True, centroid_only=False)
        cluster_3d_points = [p.point for p in cluster.points]
        points.extend(cluster_3d_points)
        cluster_ids.extend([cluster.id]*len(cluster_3d_points))
        point_labels.extend([str(p) for p in cluster.points])

        if cluster.centroid is not None:
            centroids.append(cluster.centroid)
            cluster_labels.append(str(cluster))


    rr.log("world/assets/clusters", rr.Points3D(
            points, radii=0.08, class_ids=cluster_ids, labels=point_labels))
    
    if show_centroids:
        rr.log("world/assets/cluster_means", rr.Points3D(
                centroids, radii=0.3, class_ids=[cl.id for cl in clusters], labels=cluster_labels))
    
def visualize_points(points:T.List[Point], color=(255,0,0,255), entity="world/assets/midpoints", size=0.3):
    labels = [str(p) for p in points]
    rr.log(entity,
           rr.Points3D([p.point for p in points], radii=size, colors=color, labels=labels))

def visualize_rays(rays:T.List[Ray], ids:T.List[int]=None, descriptions:T.List[str]=None,
                   arrow_scale=42, conf_threshold=0.3, extra_tag=None, category:T.List[str]=None):
    ''' Visualize rays 
        args: 
        - ids: list of cluster labels for each ray --> colorization
        - description: string to describe each ray
        - conf_threshold - don't visualize rays with class confidance lower than
        - category: show only rays of specified category names
    '''
    
    rays_filtered = []
    ids_filtered = []
    descs_filtered = []
    for i, r in enumerate(rays):
        if conf_threshold:
            if r.score < conf_threshold:
                continue
        if category:
            if r.class_name not in category:
                continue
        # ok
        rays_filtered.append(r)
        if ids: ids_filtered.append(ids[i])
        if descriptions: descs_filtered.append(ids[i])

    if len(rays_filtered) > 0:
        origins, vectors = np.zeros([len(rays_filtered), 3]), np.zeros([len(rays_filtered), 3])
        for i, r in enumerate(rays_filtered):
            origins[i] = r.origin
            vectors[i] = r.direction * arrow_scale

        if descriptions is None:
            descriptions_filtered = [f"{r.frame_id}/{r.sensor}-->{r.global_instance}" if r.global_instance is not None else
                                     f"{r.frame_id}/{r.sensor}-({r.id})->{r.class_name}[{r.score:.2f}]" for r in rays_filtered ]
            # descriptions_filtered = [str(r) for r in rays_filtered]

        tag = "world/rays/"
        if extra_tag:
            tag += extra_tag

        colors = [get_class_color(r) for r  in rays_filtered]
        rr.log(tag, rr.Arrows3D(origins=origins, 
                                vectors=vectors,
                                class_ids=ids_filtered,
                                labels=descriptions_filtered,
                                colors=colors
                                ))
        

def visualize_trajectory(traj:TrajectoryData):
    COORD_ARROW = rr.Arrows3D(
                                vectors=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                                colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
                            )
    rr.init('reel_triangulation', spawn=True)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, timeless=True) # setup rerun world coordinate system
    rr.log("world", COORD_ARROW)

    FLATTEN_Z_COORD = 1
    ARROW_LENGTH = 1
    traj_xy, traj_hpr = traj.get_trajectory()
    if traj_xy is not None:
        # RERUN VIS
        arrow_vects = [] # arrow directional vectors
        for i, hpr in enumerate(traj_hpr):
            arrow_dx = ARROW_LENGTH * np.sin(np.radians(hpr[0]))
            arrow_dy = ARROW_LENGTH * np.cos(np.radians(hpr[0]))
            arrow_vects.append(np.array([arrow_dx, arrow_dy, 0]))
            # TODO: add a point at the beginning of the array        
        # stack flat Z-coordinate to array
        trj_points3D = np.hstack([traj_xy, np.ones([len(traj_xy), 1]) * FLATTEN_Z_COORD ])
        rr.log('world/gnss_poses', rr.Arrows3D(
            origins = trj_points3D,
            vectors=arrow_vects,
            # labels=frames
            )
        )
    
def visualize_ray_clusters(ray_clusters: dict, ignore_single=False, conf_threshold=0.3, category=None):
    '''visualize ray cluster dict: label->[list of rays]'''
    # all_rays = []
    # all_ids = []
    # all_labels = []
    idx = 0
    for label, rays in ray_clusters.items():
        if ignore_single:
            if len(rays) == 1:
                continue
        # all_labels.extend([label]*len(rays))
        ids = ([idx]*len(rays))
        idx += 1

        visualize_rays(rays, ids=ids, conf_threshold=conf_threshold, extra_tag=str(idx), category=category)

def visualize_transf_matrices(matrices:T.List[np.ndarray], points:T.List[np.ndarray]=None,
                              points_labels:T.List[str]=None):
    '''visualize transformation matrices using rerun'''
    COORD_ARROW = rr.Arrows3D(
                            vectors=[[3, 0, 0], [0, 3, 0], [0, 0, 3]],
                            colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
                        )
    
    rr.init('TransformationMatrix_Visualization', spawn=True, strict=True)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, timeless=True) # setup rerun world coordinate system
    rr.log("world", COORD_ARROW)
    
    for i, t in enumerate(matrices):
        rr_transform = rr.TranslationAndMat3x3(mat3x3=t[:3,:3],
                                               translation=t[:3,3])
        rr.log(f"world/t_{i}", rr.Transform3D(rr_transform))

    rr.log("world/points", rr.Points3D(points, colors=(255,0,0), radii=0.4, labels=points_labels))    
    

def vis_cam_detections(img, detections, R, t, K, res=(3008, 4096), label='cam'):
    '''
        visualizing cameras and detections in 3d
        :param img: image, dim corresponding to res
        :param detections: list of coco detection dicts for img
        :param K: 3x3 calibration matrix
        :param res: camera sensor resolution (w,h)
    '''
    
    # TODO: visualize bounding box in camera

    P = get_P(R,t,K) # R, t, will come from XVN ... TODO: add to detections
    principal_point = (K[0,2], K[1,2]) 
    focal_length = (K[0,0], K[1,1]) 
    
    entity = "world/"+label

    # log the camera
    rr.log(entity+"/image", rr.Transform3D(translation=None, rotation=R))
    rr.log(entity+"/image", rr.Pinhole(width=res[0], height=res[1], 
                                focal_length=focal_length,
                                principal_point=principal_point))
    rr.log(entity+"/image", rr.Image(img))

    rays = []
    raylabels = []
    bboxes = []
    for i, det in enumerate(detections):
        bbox = det['bbox']
        bboxes.append(bbox)
        # get uv coordinates of the detections centers
        u = get_coco_center(bbox)
        # print(f"u: {u}")
        u = np.append(u, 1) # just add arbitrary w value - to project "a" point (raydir)
        d = np.linalg.pinv(P) @ u
        d = d[:3] 
        d = (d/np.linalg.norm(d))

        # log the rays TODO: parallelize
        ray = (t, d)
        rays.append(ray)
        print(ray)
        ray = rr.Arrows3D(origins=[t], vectors=[d*5]) # scale to make the arrow longer
        raylabel = label+"/ray_"+str(i) 
        raylabels.append(raylabel)
        rr.log(entity+"/ray_"+str(i), ray)

    # visualize 2d detection bounding boxes
    rr.log(entity+"/image", rr.Boxes2D(array=np.array(bboxes), array_format=rr.Box2DFormat.XYWH))
    return rays, raylabels   


def visualize_detection_raycasting(detections, P, entity, ray_origin):
    rays = []
    raylabels = []
    bboxes = []
    for i, det in enumerate(detections):
        bbox = det['bbox']
        bboxes.append(bbox)
        # get uv coordinates of the detections centers
        u = get_coco_center(bbox)
        # print(f"u: {u}")
        u = np.append(u, 1) # just add arbitrary w value - to project "a" point (raydir)
        d = np.linalg.pinv(P) @ u

        print(f"d: {d}")
        d = d[:3] # 4th coordinate equal to 0 ... ray
        # d = (d/np.linalg.norm(d)) # not needed 

        # log the rays TODO: parallelize --log as a batch! 
        ray = (ray_origin, d)
        rays.append(ray)
        print(ray)
        ray = rr.Arrows3D(origins=[ray_origin], vectors=[d*10]) # scale to make the arrow longer
        # raylabel = label+"/ray_"+str(i) 
        # raylabels.append(raylabel)
        # rr.log(entity+"/ray_"+str(1i), ray)
        rr.log(entity+"/ray_"+str(i), ray)

def visualize_detection_raycasting_points(detections, P, entity, ray_origin):
    rays = []
    raylabels = []
    bboxes = []
    for i, det in enumerate(detections):
        bbox = det['bbox']
        bboxes.append(bbox)
        # get uv coordinates of the detections centers
        u = get_coco_center(bbox)
        # print(f"u: {u}")
        u = np.append(u, 1) # just add arbitrary w value - to project "a" point (raydir)
        d = np.linalg.pinv(P) @ u

        print(f"d: {d}")
        d = d[:3] # 4th coordinate equal to 0 ... ray
        # d = (d/np.linalg.norm(d)) # not needed 

        # log the rays TODO: parallelize --log as a batch! 
        ray = (ray_origin, d)
        rays.append(ray)
        print(ray)
        ray = rr.Arrows3D(origins=[ray_origin], vectors=[d*10]) # scale to make the arrow longer
        # raylabel = label+"/ray_"+str(i) 
        # raylabels.append(raylabel)
        # rr.log(entity+"/ray_"+str(1i), ray)
        rr.log(entity+"/ray_"+str(i), ray)


def visualize_ground_truth_landmarks(landmarks:dict):
    points = []
    descriptions = []
    colors = []
    for landmark in landmarks.values():
        coord = list(landmark['projected_coord'])
        coord_already_exits = np.any([np.allclose(coord, p) for p in points])
        if coord_already_exits:
            coord[-1] += 1.0
        points.append(coord)
        descriptions.append(landmark['id'])
        
        if landmark['class'].lower() in CLASS_COLOR:
            colors.append(CLASS_COLOR[landmark['class'].lower()])
        else:
            colors.append(CLASS_COLOR['other'])

    rr.log("world/ground_truth_landmarks", 
           rr.Points3D(points, radii=0.3, labels=descriptions, colors=colors))