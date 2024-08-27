# config dataclasses wrapper for .toml config
# can I generate all this somehow automatically based on TOML? 

from dataclasses import dataclass, field, fields
from abc import ABC
import typing as T
import toml
from pathlib import Path

# @dataclass
# class Common:
#     input_path: Path 
#     workdir: Path

@dataclass
class Visualization:
    visualize_trajectory: bool
    visualize_clusters: bool
    visualize_cluster_centroids: bool
    visualize_rays: bool
    visualize_filtered: bool

@dataclass
class Initialization:
    reel_frames: Path # input
    work_dir: Path # target
    undistort: bool
    # TODO: undistortion params

@dataclass
class Detection:
    command: str

@dataclass
class Raycasting:
    camera_poses: Path # file with camera poses # TODO: optional
    cam_frames_suffix: str
    process_sensors: T.List[str]
    frames: T.Union[list, str] # list of frames or 'all'
    every_nth: int
    undistort: bool # mostly expected already undistorted images, but might be handy
    flatten_rays: bool
    visualize_frames: bool
    visualize_images: bool # visualize images as well --> expensive
    debug: bool

@dataclass
class Prefiltering:
    mean_dist_from_cam: float
    max_dist_from_cam: float
    min_score: float
    max_dist_between_rays: float

@dataclass
class Features:
    checkpoint: Path
    classes_info: Path
    softmax: bool
    batch_size: int
    debug: bool

@dataclass
class Filtering:
    product_feature_max_t: float # semantic filtering
    ray_epsilon_neighborhood: float # size of the eps neighborhood to accept

@dataclass
class Clustering:
    weighted_centroid: bool

@dataclass
class ClusteringDBSCAN(Clustering):
    eps: int
    minPts: int
    semantic_cluster_splitting: bool

@dataclass
class ClusteringBihierarchical(Clustering):
    t1: float
    t2: float
    alpha: float
    minPts: int

@dataclass
class Evaluation:
    ground_truth_landmarks: Path
    
@dataclass
class Map:
    utm_zone: int
    map_center: int
    map_zoom: int


class Config:
    config_classes = {
        'initialization': Initialization,
        'detection': Detection,
        'visualization': Visualization,
        'raycasting': Raycasting,
        'prefiltering': Prefiltering,
        'features': Features,
        'filtering': Filtering,
        'clustering': {'dbscan': ClusteringDBSCAN, 
                       'bihierchical': ClusteringBihierarchical},
        'evaluation': Evaluation,
        'map':Map,
    }

    def __init__(self, config_dict:dict):
        ''' load from toml-dict '''
        classnames_unpacked = [key for key, value in Config.config_classes.items() 
                               for key in ([key] if not isinstance(value, dict) else [key] + list(value.keys()))]

        for tag, section_val in config_dict.items():
            section_tag = str.lower(tag)
            if section_tag in Config.config_classes:
                section_class = Config.config_classes[section_tag]
                section_args = section_val
                # if child object -> unpack params to init child of abstract class
                for k,v in section_val.items():
                    if isinstance(v, dict):
                        if k in classnames_unpacked:
                            section_class = Config.config_classes[section_tag][k]
                            section_args_without_this = {attr:val for attr,val in section_args.items() if attr != k}
                            section_args = {**section_args_without_this, **v}
                        else:
                            raise Exception(f"Unknow class {k}")
                # ClusteringDBSCAN()
                instance = section_class(**section_args)
                setattr(self, section_tag, instance)
            else:
                raise Exception(f"Unexpected config section; {section_tag}")
            
        print('config deserialized')

if __name__=='__main__':
    with open('config_default.toml') as f:
        conf_file = toml.load(f)
    config = Config(conf_file)    