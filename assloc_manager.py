import sys
import toml
from pathlib import Path
import typing as T
import pickle
import shutil
import copy
import json

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances

import config
from clustering import clustering_bihierarchical, clustering_dbscan
from rerun_visualization import visualize_trajectory, visualize_midpoint_clusters, init_visualization, visualize_rays, visualize_points, SOME_COLORS, visualize_ground_truth_landmarks
from deep_features import genereate_deep_features_midpoints, generate_deep_features_2
from tools.undistort_images import undistort_imgs
from tools.rotate_images import rotate_imgs
from trajectory_data import TrajectoryData
from evaluation import fuzzy_PP
from map import MapVisualizer


UNDISTORTED_DIR = "reel_undistorted"
ROTATED_DIR = "data_rotated"
DETECTIONS_WITH_FEATURES = "detections2.json"

# FILTERING VISUALIZATION COLORS 

class AssetLocalizationManager:

    def __init__(self, config_path:Path):
        with open(config_path) as f:
            self.config_dict = toml.load(f)
        self.config = config.Config(self.config_dict)
        
        self.points = None
        self.traj = None
        
    def run(self):
        ''' execute the consecutive sections of pipeline '''
        for tag in self.config_dict:
            if tag.lower() in ['visualization']: # visualization is a global config section not a stage
                continue
            stage_func = getattr(self, tag.lower())
            stage_conf = getattr(self.config, tag.lower())
            # call the stage function
            stage_func(stage_conf)

        print("Asset localization pipeline run finished") 

    ## stage handling functions
    def initialization(self, cfg:config.Initialization):
        self.reel_frames_dir = Path(cfg.reel_frames)
        # find yaml calibration
        yamls = list(self.reel_frames_dir.glob("*.yaml"))
        print(f"searching for calib. file in {yamls}:{self.reel_frames_dir}")
        if len(yamls) > 1:
            print(f"several yaml calibration file candidates found: \n{yamls}")
            exit()
        if len(yamls) < 1:
            print("no calibration file found")
            exit()
        self.calib_p = yamls[0]
        self.work_dir = Path(cfg.work_dir)
        self.work_dir.mkdir(exist_ok=True, parents=True)

        if not (self.work_dir/ROTATED_DIR).exists():
            # undistort here
            undistort_imgs(self.reel_frames_dir, self.work_dir/UNDISTORTED_DIR, self.calib_p)
            rotate_imgs(self.work_dir/UNDISTORTED_DIR, self.work_dir/ROTATED_DIR)
        else:
            print('undistortion & rotation already done')
        # start rerun visualization
        init_visualization()
    
        print("initialized")

    def detection(self, cfg:config.Detection):
        if not (self.work_dir/'detections.json').exists():
            # TODO: replace this with an inference call 
            detections_p = '/media/tomas/samQVO_4TB_D/assdet-experiments/drtinova_u_new/detections.json'
            # MED = "/media/tomas/samQVO_4TB_D/asset-detection-datasets/drtinova_med_u_track/data_m/detections/detections.json"
            shutil.copyfile(detections_p, self.work_dir/'detections.json')
        
        self.detections_p = self.work_dir/'detections.json'


        # TODO: move to a separate stage
        with open(self.config.features.classes_info) as f:
            classes_data = json.load(f)
        self.id_to_label = classes_data['id_to_label']
        if not (self.work_dir/DETECTIONS_WITH_FEATURES).exists():
            self.deep_features = generate_deep_features_2(self.config.features.checkpoint, self.detections_p, self.work_dir/UNDISTORTED_DIR, 
                                    self.work_dir/DETECTIONS_WITH_FEATURES, self.id_to_label, debug=self.work_dir/'features2_debug')
        else:
            with open(self.work_dir/DETECTIONS_WITH_FEATURES) as f:
                self.deep_features = json.load(f)

    def raycasting(self, cfg:config.Raycasting):
        # in: undistorted_reel_frames, detection
        # out: raycasting_result
        import raycasting_poses as raycasting

        calib_p = self.work_dir/UNDISTORTED_DIR/'calib.yaml'
        gpx_p = self.work_dir/UNDISTORTED_DIR/'extracted.gpx'

        camera_poses, self.coordsys_origin = raycasting.get_camera_transforms(cfg.camera_poses, cfg.process_sensors)

        self.traj = TrajectoryData(self.work_dir/'detections2.json', gpx_p, calib_p, landmark_gnss_p=self.config.evaluation.ground_truth_landmarks, 
                                   geo_coord_sys_origin=self.coordsys_origin, detections_features=self.deep_features)
        with open(self.work_dir/'trajectory_data.pickle', 'wb') as f:
            pickle.dump(self.traj, f)
        
        visualize_ground_truth_landmarks(self.traj.landmarks)

        if not (self.work_dir/'raycasting_result.pickle').exists(): 
            # self.raycasting_result = raycasting.main(cfg, self.reel_frames_dir, self.detections_p)
            self.raycasting_result = raycasting.raycast(cfg, camera_poses, self.work_dir/UNDISTORTED_DIR, self.traj)
            
            with open(self.work_dir/'raycasting_result.pickle', 'wb') as f:
                pickle.dump(self.raycasting_result, f)
        else:
            print("raycasting already done, loading existing")
            with open(self.work_dir/'raycasting_result.pickle', 'rb') as f:
                self.raycasting_result = pickle.load(f)
            
        self.points = self.raycasting_result.points
        # self.traj = self.raycasting_result.traj

        if self.config.visualization.visualize_trajectory:
            visualize_trajectory(self.traj)
        if self.config.visualization.visualize_rays:
            rays_to_vis = [r for r in self.raycasting_result.rays 
                           if self.id_to_label[np.argmax(r.cls_feature)] == 'A12']
            visualize_rays(rays_to_vis)
        
        print("raycasting done")

    
    def prefiltering(self, cfg:config.Prefiltering):
        MAX_RAYS_DIST = 2 # TODO: somwhere else
        self.points_prefiltered = [p for (i,p) in enumerate(self.points) 
                                   if (p.l[0]+p.l[1])/2 < cfg.mean_dist_from_cam
                                   and p.l[0] < cfg.max_dist_from_cam and p.l[1] < cfg.max_dist_from_cam
                                   and np.min((p.r1.score, p.r2.score)) > cfg.min_score #]
                                   and p.dist < MAX_RAYS_DIST]
        self.points = self.points_prefiltered
                
        print(f"pre-filtered: {len(self.points)} / {len(self.raycasting_result.points)}")
        # TODO: if visualize filtered
        visualize_points(self.points, entity="world/prefiltered", color=SOME_COLORS[0])
    
    def features(self, cfg:config.Features):
        # with open(cfg.classes_info) as f:
        #     classes_data = json.load(f)
        # self.id_to_label = classes_data['id_to_label']

        # DEBUG_DIR = "features_debug"

        # if (self.work_dir/'features.npy').exists():
        #     print("features already computed, loading existing...")
        #     cls_features = np.load(self.work_dir/'features.npy')
        # else:
        #     debug = None
        #     if cfg.debug:
        #         debug = self.work_dir/DEBUG_DIR
                
        #     cls_features = genereate_deep_features_midpoints(self.points, self.work_dir, cfg.checkpoint, self.id_to_label, debug=debug, 
        #                                                      softmax=cfg.softmax, num_classes=len(self.id_to_label), return_labels=True, 
        #                                                      data_path=self.work_dir/UNDISTORTED_DIR, data_rotated=False, batch_size=160)
            
        #     with open(self.work_dir/'features.npy', 'wb') as f:
        #         np.save(f, cls_features)

        # self.semantic_features = cls_features[0] * cls_features[1] # p[i].r1 * p[i].r2
        
        # assign the cls_features # TODO: move totally elsewhere 
        for i,p in enumerate(self.points):
            p.cls_feature = np.asarray(p.r1.cls_feature) * np.asarray(p.r2.cls_feature) #self.semantic_features[i]
            # if np.any(self.semantic_features[i] != p.cls_feature):
            #     print(self.semantic_features[i])
            #     print("x")
            #     print(p.cls_feature)
            #     print()
            p.id = i # assign id to non-filtered point --so the cls_features matrices kept
            p.cls_feature_label = self.id_to_label[np.argmax(p.cls_feature)] 

        # spatial distance -- calculating this here is insane
        # points_loc = [p.point for p in self.points]
        # self.dist_spatial = euclidean_distances(points_loc, points_loc)
        # self.dist_spatial += np.eye(self.dist_spatial.shape[0]) * 555 # don't merge a point with itself # TODO: handle the diagonal less idiotically
        # # np.save(self.work_dir/'dist_spatial.npy', self.dist_spatial)

    
    def filtering(self, cfg:config.Filtering):
        print("FILTERING")
        # semantic filter
        self.points_filtered = [p for (i,p) in enumerate(self.points) 
                                if np.max(p.cls_feature) > cfg.product_feature_max_t]
        print(f"semantically filtered: {len(self.points_filtered)} / {len(self.raycasting_result.points)}")
        visualize_points(self.raycasting_result.points, entity="world/semantically_filtered", color=SOME_COLORS[1])

        # filter those too close to camera --> on the road
        # TODO: this distance search is super inefective
        DIST_TO_CAM_THRESHOLD = 0.8 # [m]
        traj_xy, traj_hpr = self.traj.get_trajectory()
        points_filtered = []
        for idx, p in enumerate(self.points_filtered):
            too_close = False
            for xy in traj_xy:
                d = np.linalg.norm(p.point[:2] - xy)
                if d < DIST_TO_CAM_THRESHOLD:
                   too_close = True
                   break
            if not too_close: # keep only of not too close to any trajecotry point
                points_filtered.append(p)
        self.points_filtered = points_filtered
        print(f"too close to track filtered: {len(self.points_filtered)} / {len(self.raycasting_result.points)}")

        # point.id --> point filtered index map 
        pid_2_filt_idx = {p.id: i for i,p in enumerate(self.points_filtered)}

        # build distance matrix indexed by filtered points
        points_loc = [p.point for p in self.points_filtered]
        self.dist_spatial = euclidean_distances(points_loc, points_loc)
        self.dist_spatial += np.eye(self.dist_spatial.shape[0]) * 555 # don't merge a point with itself # TODO: handle the diagonal less idiotically

        ## ray filter -- for each ray keep one point with the highest support
        ## statistics and stuff
        # TODO: move somewhere else
        ray_points = {} # map ray_id --> all the points along the ray
        for p in self.points_filtered:
            for ridx, rid in enumerate(['r1', 'r2']):
                r = getattr(p, rid)
                if r.id not in ray_points:
                    ray_points[r.id] = []
            
                # insert by distance
                for j in range(len(ray_points[r.id])+1):
                    if j < len(ray_points[r.id]):
                        if ray_points[r.id][j][0] > p.l[ridx]: # [r.id][j].l[ridx]
                            break
                ray_points[r.id].insert(j, (p.l[ridx], p)) # (dist_along_ray, point)
                
        # ray_id --> distance for each point along the ray
        self.ray_point_dists = { rid: [raypoint[0] for raypoint in rpoints] for rid, rpoints in ray_points.items() } 

        ray_neighborhood_magnitude = {}
        for ray_id in ray_points:
            neighborhood_magnitude = [0] * len(ray_points[ray_id]) 
            for i, dp in enumerate(ray_points[ray_id]):
                # for each point p along the ray get the number of points in the epsilon neighborhood = nbr_mag(p)
                d, p = dp
                close_points_ids = np.where(self.dist_spatial[pid_2_filt_idx[p.id],:] < cfg.ray_epsilon_neighborhood)[0]
                neighborhood_magnitude[i] = len(close_points_ids)
                
            ray_neighborhood_magnitude[ray_id] = neighborhood_magnitude

        # keep only one point per ray (by magnitude)
        points_filtered2 = []
        for ray_id, points in ray_points.items():
            best_point = points[np.argmax(ray_neighborhood_magnitude[ray_id])]
            points_filtered2.append(best_point[1])

        self.points_filtered = points_filtered2
        print(f"ray cluster analysis filter {len(self.points_filtered)} / {len(self.raycasting_result.points)}")
        visualize_points(self.points_filtered, entity="world/filtered", color=SOME_COLORS[2])

        # recalculate the distance for filtered points -> final dist matrices used for clustering
        # TODO: select from the original dist_spatial - don't compute again
        points_loc = [p.point for p in self.points_filtered]
        self.dist_spatial = euclidean_distances(points_loc, points_loc)
        self.dist_spatial += np.eye(self.dist_spatial.shape[0]) * 555 # don't merge a point with itself # TODO: handle the diagonal less idiotically

        visualize_points(self.points_filtered, entity="world/points_to_be_clustered")

        points_semantic_feat = [p.cls_feature for p in self.points_filtered]
        self.dist_semantic = cosine_distances(points_semantic_feat, points_semantic_feat)

    def clustering(self, cfg:config.Clustering):
        print("CLUSTERING")
        if 'dbscan' in self.config_dict['CLUSTERING']:
            self.clusters, _ = clustering_dbscan(self.dist_spatial, cfg.minPts, cfg.eps, self.points_filtered, semantic_cluster_splitting=cfg.semantic_cluster_splitting)
        elif 'bihierchical' in self.config_dict['CLUSTERING']:
            # TODO: ommit ray stuff here -- the above filtering enough
            self.clusters, stats = clustering_bihierarchical(self.dist_spatial, self.dist_semantic, self.points_filtered, self.raycasting_result.rays)
        else:
            raise Exception(f"Uknown clustering method: {self.config_dict['CLUSTERING']}")
        
        if self.config.visualization.visualize_clusters:
            visualize_midpoint_clusters(self.clusters, 
                                        show_centroids=self.config.visualization.visualize_cluster_centroids)

        for c in self.clusters:
            print(c)

        with open(self.work_dir/"config_used.toml", 'w') as f:
            toml.dump(self.config_dict, f)

        
        # # filter clusters too close to any camera trajectory pose -- def. faster then with points
        # DIST_TO_CAM_THRESHOLD = 0.5
        # traj_xy, traj_hpr = self.traj.get_trajectory()
        # cluster_dist_to_closest_traj_point = []
        # clusters_filtered = []
        # for cidx, c in enumerate(self.clusters):
        #     for xy in traj_xy:
        #         d = np.linalg.norm(c.centroid[:2] - xy)
        #         cluster_dist_to_closest_traj_point.append(d)
        #         if d < DIST_TO_CAM_THRESHOLD:
        #             print(f"cluster {cidx} too close to camera trajectory")
        #         else:
        #             clusters_filtered.append(c)
        # self.clusters = clusters_filtered
        # # TODO: seems ok, make it into the map

        if self.config.visualization.visualize_clusters:
            visualize_midpoint_clusters(self.clusters, 
                                        show_centroids=self.config.visualization.visualize_cluster_centroids)
            
        cluster_data_export_dir = self.work_dir/'cluster_data'
        import shutil
        try:
            shutil.rmtree(cluster_data_export_dir)
        except Exception as ex:
            print(ex)
        cluster_data_export_dir.mkdir()
        for i, cluster in enumerate(self.clusters):
            cluster.export((cluster_data_export_dir/f"cluster_{i}.json"))

    def evaluation(self, cfg:config.Evaluation):
        dist_threshold = 1 # target in meters
        dist_max_threshold = 2.5

        pp, fn, matches = fuzzy_PP(self.clusters, self.traj.landmarks, 
                                   t=dist_threshold, t_max=dist_max_threshold, planar=True)
        print(50*"-")
        print(f"PP: {pp}/{len(self.traj.landmarks)}")
        print(f"FN: {fn}/{len(self.traj.landmarks)}")

        
        precision = pp / len(self.clusters)
        recall = pp / len(self.traj.landmarks)


        print(f"precision: {precision}, recall: {recall}")
        print()
        # matches = {l:None for l in self.traj.landmarks}
        # unnassigned_clusters = copy.deepcopy(self.clusters)
        
        # for l_id, landmark in self.traj.landmarks.items():
        #     # find the index of the closest cluster with the correct semantic class
        #     cluster_dists = [np.linalg.norm(landmark['projected_coord'][:2] - c.centroid[:2]) 
        #                      for c in unnassigned_clusters if c.category == landmark['class']]
        #     if len(cluster_dists) > 0:
        #         cl_idx = np.argmin(cluster_dists)
        #         selected_cluster = unnassigned_clusters[cl_idx]
        #         # check whether the distance is below threshold    
        #         if np.linalg.norm(landmark['projected_coord'][:2] - selected_cluster.centroid[:2]) < dist_threshold:
        #             matches[l_id] = selected_cluster.id
        #             print(f"GT match: {l_id} <--> cluster-{selected_cluster.id}")
        #             print(f"class: {landmark['class']} x {selected_cluster.category}")
        #             unnassigned_clusters.pop(cl_idx)
        #         else:
        #             print(f"distance too high for {l_id} <--> cluster-{selected_cluster.id}")
        #             print(np.linalg.norm(landmark['projected_coord'][:2] - selected_cluster.centroid[:2]))

        #     if matches[l_id] is None:
        #         print(f"no match for {l_id}")
        
        # correct_count = 0
        # for m in matches.values():
        #     if m is not None:
        #         correct_count += 1

        # precision = correct_count / len(self.clusters)
        # recall = correct_count / len(self.traj.landmarks)
        # print(f"correctly localized: {correct_count} ")
        # print(f"precision: {precision}, recall: {recall}")

    def map(self, cfg:config.Map):
        map = MapVisualizer(map_center=cfg.map_center, zoom=cfg.map_zoom)
        
        map.add_clusters(self.clusters, utm_zone=cfg.utm_zone, 
                          coord_sys_translation=self.coordsys_origin)
        
        map.add_ground_truth(self.traj.landmarks, utm_zone=cfg.utm_zone, 
                             coord_sys_translation=self.coordsys_origin)
        map.show()   

        print(f"done") 

if __name__== "__main__":
    # config_path = Path(sys.argv[1])
    config_path = 'config/latest.toml'
    assloc = AssetLocalizationManager(config_path)
    assloc.run()