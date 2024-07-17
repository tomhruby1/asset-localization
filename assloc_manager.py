import sys
import toml
from pathlib import Path
import typing as T
import pickle
import shutil

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances

import config
from clustering import clustering_bihierarchical, clustering_dbscan
from rerun_visualization import visualize_trajectory, visualize_midpoint_clusters, init_visualization, visualize_rays, visualize_points, SOME_COLORS
from deep_features import genereate_deep_features_midpoints, get_id_to_label
from tools.undistort_images import undistort_imgs
from tools.rotate_images import rotate_imgs
# skipping stages if already output stored

UNDISTORTED_DIR = "reel_undistorted"
ROTATED_DIR = "data_rotated"

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


    def raycasting(self, cfg:config.Raycasting):
        # in: undistorted_reel_frames, detection
        # out: raycasting_result
        import raycasting_poses as raycasting

        cams_p = '/media/tomas/samQVO_4TB_D/assdet-experiments/drtinova_u_new/cameras_exported_projected_wgs84_utm_33n.txt'
        
        if not (self.work_dir/'raycasting_result.pickle').exists(): 
            # self.raycasting_result = raycasting.main(cfg, self.reel_frames_dir, self.detections_p)
            
            camera_poses = raycasting.get_camera_transforms(cams_p, cfg.process_sensors)
            self.raycasting_result = raycasting.raycast(cfg, camera_poses, self.work_dir/UNDISTORTED_DIR, self.detections_p)
            
            with open(self.work_dir/'raycasting_result.pickle', 'wb') as f:
                pickle.dump(self.raycasting_result, f)
        else:
            print("raycasting already done, loading existing")
            with open(self.work_dir/'raycasting_result.pickle', 'rb') as f:
                self.raycasting_result = pickle.load(f)
            
        self.points = self.raycasting_result.points
        self.traj = self.raycasting_result.traj

        if self.config.visualization.visualize_trajectory:
            visualize_trajectory(self.traj)
        if self.config.visualization.visualize_rays:
            visualize_rays(self.raycasting_result.rays)
        
        print("raycasting done")

    
    def prefiltering(self, cfg:config.Prefiltering):
        self.points_prefiltered = [p for (i,p) in enumerate(self.points) 
                                   if (p.l[0]+p.l[1])/2 < cfg.mean_dist_from_cam
                                   and p.l[0] < cfg.max_dist_from_cam and p.l[1] < cfg.max_dist_from_cam
                                   and np.min((p.r1.score, p.r2.score)) > cfg.min_score]
        self.points = self.points_prefiltered
                
        print(f"pre-filtered: {len(self.points)} / {len(self.raycasting_result.points)}")
        # TODO: if visualize filtered
        visualize_points(self.points, entity="world/prefiltered", color=SOME_COLORS[0])
    
    def features(self, cfg:config.Features):
        id_2_label = get_id_to_label()
        
        DEBUG_DIR = "features_debug"

        if (self.work_dir/'features.npy').exists():
            print("features already computed, loading existing")
            cls_features = np.load(self.work_dir/'features.npy')
        else:
            debug = None
            if cfg.debug:
                debug = self.work_dir/DEBUG_DIR
                
            cls_features = genereate_deep_features_midpoints(self.points, self.work_dir, debug=debug, 
                                                             softmax=cfg.softmax, num_classes=len(id_2_label), return_labels=True, 
                                                             data_path=self.work_dir/UNDISTORTED_DIR, data_rotated=False, batch_size=160)
            
            with open(self.work_dir/'features.npy', 'wb') as f:
                np.save(f, cls_features)

        self.semantic_features = cls_features[0] * cls_features[1]
        
        # assign the cls_features
        for i,p in enumerate(self.points):
            p.cls_feature = self.semantic_features[i]
            p.id = i # assign id to non-filtered point --so the cls_features matrices kept
            p.cls_feature_label = id_2_label[np.argmax(p.cls_feature)] 

        # spatial distance
        # --this probably shouldnt be computed for prefiltered, but to keep dimensions with semantic stuf --ok for now
        points_loc = [p.point for p in self.points]
        self.dist_spatial = euclidean_distances(points_loc, points_loc)
        self.dist_spatial += np.eye(self.dist_spatial.shape[0]) * 555 # don't merge a point with itself # TODO: handle the diagonal less idiotically
        # # np.save(self.work_dir/'dist_spatial.npy', self.dist_spatial)

    
    def filtering(self, cfg:config.Filtering):
        
        # semantic filter
        self.points_filtered = [p for (i,p) in enumerate(self.points) 
                                if np.max(self.semantic_features[i]) > cfg.product_feature_max_t]
        print(f"semantically filtered: {len(self.points_filtered)} / {len(self.raycasting_result.points)}")
        visualize_points(self.raycasting_result.points, entity="world/semantically_filtered", color=SOME_COLORS[1])


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
                p = dp[1]
                close_points_ids = np.where(self.dist_spatial[p.id,:] < cfg.ray_epsilon_neighborhood)[0] # TODO: param
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
        points_loc = [p.point for p in self.points_filtered]
        self.dist_spatial = euclidean_distances(points_loc, points_loc)
        self.dist_spatial += np.eye(self.dist_spatial.shape[0]) * 555 # don't merge a point with itself # TODO: handle the diagonal less idiotically

        visualize_points(self.points_filtered, entity="world/points_to_be_clustered")

        points_semantic_feat = [p.cls_feature for p in self.points_filtered]
        self.dist_semantic = cosine_distances(points_semantic_feat, points_semantic_feat)

    def clustering(self, cfg:config.Clustering):
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

if __name__== "__main__":
    # config_path = Path(sys.argv[1])
    config_path = 'config/config_v2.toml'
    assloc = AssetLocalizationManager(config_path)
    assloc.run()