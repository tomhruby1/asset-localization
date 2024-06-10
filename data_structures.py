# dataclasses stored here

# TRIANGULATION DATATYPES

from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import json
import typing as T
import pickle
from scipy.stats import entropy


# TODO: move this elsewhere
SIGNS_CATEGORY_NAMES = (
    'A10', 'A11', 'A12', 'A14', 'A15', 'A16', 'A17', 'A18', 'A19', 'A1a', 'A1b', 'A22', 'A24', 'A28', 'A29', 'A2a', 'A2b', 'A30', 'A31a', 'A31b', 'A31c', 'A32a', 'A32b', 'A4', 'A5a', 'A6a', 'A6b', 'A7a', 'A8', 'A9', 'B1', 'B11', 'B12', 'B13', 'B14', 'B15', 'B16', 'B17', 'B19', 'B2', 'B20a', 'B20b', 'B21a', 'B21b', 'B24a', 'B24b', 'B26', 'B28', 'B29', 'B32', 'B4', 'B5', 'B6', 'C1', 'C10a', 'C10b', 'C13a', 'C14a', 'C2a', 'C2b', 'C2c', 'C2d', 'C2e', 'C2f', 'C3a', 'C3b', 'C4a', 'C4b', 'C4c', 'C7a', 'C9a', 'C9b', 'E1', 'E11', 'E11c', 'E12', 'E13', 'E2a', 'E2b', 'E2c', 'E2d', 'E3a', 'E3b', 'E4', 'E5', 'E6', 'E7a', 'E7b', 'E8a', 'E8b', 'E8c', 'E8d', 'E8e', 'E9', 'I2', 'IJ1', 'IJ10', 'IJ11a', 'IJ11b', 'IJ14c', 'IJ15', 'IJ2', 'IJ3', 'IJ4a', 'IJ4b', 'IJ4c', 'IJ4d', 'IJ4e', 'IJ5', 'IJ6', 'IJ7', 'IJ8', 'IJ9', 'IP10a', 'IP10b', 'IP11a', 'IP11b', 'IP11c', 'IP11e', 'IP11g', 'IP12', 'IP13a', 'IP13b', 'IP13c', 'IP13d', 'IP14a', 'IP15a', 'IP15b', 'IP16', 'IP17', 'IP18a', 'IP18b', 'IP19', 'IP2', 'IP21', 'IP21a', 'IP22', 'IP25a', 'IP25b', 'IP26a', 'IP26b', 'IP27a', 'IP3', 'IP31a', 'IP4a', 'IP4b', 'IP5', 'IP6', 'IP7', 'IP8a', 'IP8b', 'IS10b', 'IS11a', 'IS11b', 'IS11c', 'IS12a', 'IS12b', 'IS12c', 'IS13', 'IS14', 'IS15a', 'IS15b', 'IS16b', 'IS16c', 'IS16d', 'IS17', 'IS18a', 'IS18b', 'IS19a', 'IS19b', 'IS19c', 'IS19d', 'IS1a', 'IS1b', 'IS1c', 'IS1d', 'IS20', 'IS21a', 'IS21b', 'IS21c', 'IS22a', 'IS22c', 'IS22d', 'IS22e', 'IS22f', 'IS23', 'IS24a', 'IS24b', 'IS24c', 'IS2a', 'IS2b', 'IS2c', 'IS2d', 'IS3a', 'IS3b', 'IS3c', 'IS3d', 'IS4a', 'IS4b', 'IS4c', 'IS4d', 'IS5', 'IS6a', 'IS6b', 'IS6c', 'IS6e', 'IS6f', 'IS6g', 'IS7a', 'IS8a', 'IS8b', 'IS9a', 'IS9b', 'IS9c', 'IS9d', 'O2', 'P1', 'P2', 'P3', 'P4', 'P6', 'P7', 'P8', 'UNKNOWN', 'X1', 'X2', 'X3', 'XXX', 'Z2', 'Z3', 'Z4a', 'Z4b', 'Z4c', 'Z4d', 'Z4e', 'Z7', 'Z9'
) 


@dataclass 
class Ray:
    '''Ray dataclass wrapping detection and it's 3d projection'''
    id: int
    # frame
    frame_id: str # frame id e.g. 4242
    sensor: str # sensor of origin cam0, cam1...    
    # detection 
    class_id: int # detection id  
    class_name: str
    score: float
    bbox: list
    # geometry
    origin: np.ndarray # 3D point
    direction: np.ndarray
    # ground truth from validation set --> global instance
    global_instance: str = None
    # bbox: list = None
    # added during inference
    cls_feature = None
    embedding = None
    # results
    point: 'Point' = None # reference to point this ray produces --assigned on Point construction
    all_points: T.List['Point'] = None # list of all midpoits this ray intersects
    
    def __str__(self):
        label = f"R_{self.id}"
        if self.global_instance is not None:
            label += f"{self.global_instance}" # just a quick peek to the instance
        else:
            label += f"{self.frame_id}/{self.sensor}--({self.id})->{self.class_name}[{self.score:.2f}]"

        label += f"|det_cls:{self.class_name}"
        
        return label

    # TODO: make this more general (not dependent on traffic_signs)
    # def simplify_label(self):
    #     '''simplify class label given mapping defined in traffic_signs'''
    #     # TODO: why some issues with class_name
    #     self.class_name, self.class_id = get_simplified_detection_label(self.class_name)

# TODO: cleanup extra fields
@dataclass
class Point: 
    # 3d point hypothesis as a product of 2 rays #TODO: rename? 
    r1: Ray
    r2: Ray
    dist: float # euklidean distance between r1, r2
    point: np.ndarray # 3D point - ndarray stored as a list (for serialization)
    l: T.Tuple[float, float] # dist from cam for r1, r2 respectivelly
    weight: float = 1
    embedding: np.ndarray = None # embedding feature vector, later in the pipeline based upon r1, r2
    cls_feature: np.ndarray = None
    cls_feature_label: str = None
    r: Ray = None # selected ray: either r1 or r2
    id: int = None
    cluster_id: int = None # optionally assign cluster ID (for vis/debug)
    
    def __init__(self, r1:Ray, r2:Ray, dist:float, point:np.ndarray, dist_from_cam:T.Tuple[float,float]):
        self.r1 = r1
        self.r2 = r2
        self.dist = dist
        self.point = point
        self.l = dist_from_cam # distance from r1, r2 respectively

        # offspring point reference to rays TODO: bleeeh ugly
        # # creates a stupid reference recursion 
        # if self.r1.all_points is None:
        #     self.r1.all_points = []
        # self.r1.all_points.append(self)
        # if self.r2 is None:
        #     self.r2.all_points = []
        # self.r1.all_points.append(self)

    def __str__(self):
        # if self.r1.global_instance is not None and self.r2.global_instance is not None:
        label = f"{str(self.r1)} x {str(self.r2)}"
        # if self.embedding is not None:
        #     label += f" |{np.argmax(self.embedding)}({np.max(self.embedding)})"
        # label = ""
        if self.id is not None:
            label = f"p{self.id}: "+label
        label += f" | l:{(self.l[0]+self.l[1])/2:.1f}"

        # label += f"w:{self.weight}"

        return label

# TODO: also add some id 
@dataclass
class Cluster:
    id: int
    points: T.List['Point']
    rays: T.List[Ray] = None
    category = None
    # purity = None
    # coverage = None
    centroid: np.ndarray = None
    weight_sum = None
    feature_purity = None

    def __init__(self, points:T.List[Point], id=int):
        self.id = id
        self.points = points
        self.rays = []
        for p in self.points:
            self.rays.append(p.r1)
            self.rays.append(p.r2)

        self.category_centroid_vote()
        self.calc_embedding_purity()
        
    def __str__(self, details=False):
        label = f"cluster-{self.id}"
        if self.category is not None:
            label += f"|cat:{self.category}"
        if self.feature_purity:
            label += f"|pur:{self.feature_purity}"

        return label
    
    def get_str_details(self) -> str:
        # if self.category:
        label = str(self)
        label += f" | has: {len(self.points)} points | cat: {self.category} | w:({self.weight_sum:.2f})"
        # label += f"\n{self.categories}"
        label += f" | feature_purity: {self.feature_purity}"

        return label
    
    def calc_embedding_purity(self):
        ''' Calculate semantic feature purity '''
        # TODO: think this through
        try:
            # or maybe muliply here ~ correlation over already correlated?
            self.avg_feature = np.sum([p.cls_feature for p in self.points], axis=0) / len(self.points)
            # self.feature_purity = np.var(self.avg_feature) 

            # quick unimodality measure
            # feat_max = np.max(self.avg_feature) # unsigned max
            # self.feature_purity = feat_max
            self.feature_purity = np.sum(self.avg_feature)  #feat_max / (np.sum(np.abs(self.avg_feature)))

        except Exception as ex:
            print(f"Error calculating emb; united embedding might not be defined for some points {ex}")
    

    def label_pure(self) -> bool:
        '''get purity for clusters based on ray clusters (intended to use only with GroundTruth info)'''
        all_cats = [] # flattened tuples 
        for cat_tuple in self.categories:
            all_cats.append(cat_tuple[0])
            all_cats.append(cat_tuple[1])
        purity = np.all([cat == all_cats[0] for cat in all_cats])

        return purity
    
    def category_centroid_vote(self):
        '''do a voting to select the most fitting class label for this cluster'''

        # here it would be definitely cleaner if classified based on the same features as with which clustering is done
        score_class = {cls_name:0 for cls_name in SIGNS_CATEGORY_NAMES}
        self.weight_sum = 0
        self.centroid = np.array([0.0,0.0,0.0]) # also get the mean while at this I guess
        self.categories = []

        for p in self.points:
            self.categories.append((p.r1.class_name, p.r2.class_name)) # debug

            score_class[p.r1.class_name] += p.r1.score
            score_class[p.r2.class_name] += p.r2.score
            weight = (p.r1.score + p.r2.score)/2 # TODO: think this through
            self.weight_sum += weight
            self.centroid += weight * p.point    
            # self.centroid += p.point
        
        self.centroid /= self.weight_sum
        # self.centroid /= len(self.points)
        
        best_label_id = np.argmax(list(score_class.values()))
        best_class_label = list(score_class.keys())[best_label_id]
        self.category = best_class_label
        # best_class_label += f" cls score:[{list(score_class.values())[best_label_id]:.2f} | w:({weight_sum:.2f})]"


@dataclass
class RaycastingResult:
    points: T.List[Point]
    traj: 'TrajectoryData'
    rays: T.List[Ray]
    projection: 'any' = None
    
    # def serialize(self, out_p:Path): 
    #     '''store as a pickle'''
    #     with open(Path(out_p), 'wb') as f:
    #         pickle.dump(self, f)

    # @classmethod
    # def deserialize(cls, out_p:Path) -> 'Point':
    #     with open(Path(out_p), 'rb') as f:
    #         return pickle.load(f)
        
    # def simplify_labels(self):
    #     '''run simplify label f
    #     or each point for each ray'''
    #     for p in self.points:
    #         p.r1.simplify_label()
    #         p.r2.simplify_label()

@dataclass
class FrameDetections:
    '''
    Object detections in single camera frame (img)
    attributes - lists with corresponding (by idx) bboxes, class labels, scores
    and global GT instances if any defined
    '''
    bboxes: list
    class_names: list
    class_ids: list
    scores: list
    global_instances: list=None # global instance labels -- ground truth corespondances
    coordinates: list=None # latitude, longitude coordinates -- ground truth gnss



## TODO: to eval/metrics/rays... ?
def get_frame_rays(rays:T.List[Ray]):
    '''order frames according to sensor frames for better search'''
    frame_rays = {}
    frame_instances = {} # frame_ident --> list of global instances
    global_instances = True if rays[0].global_instance else False
    for ray in rays:
        ray_ident = ray.frame_id+"_"+ray.sensor
        if ray_ident not in frame_rays:
            frame_rays[ray_ident] = []
            if global_instances: frame_instances[ray_ident] = []
        frame_rays[ray_ident].append(ray)

        if global_instances:        
            frame_instances[ray_ident].append(ray.global_instance)

    return frame_rays, frame_instances

def get_cluster_instersection_size(reference:T.List[Ray], hypothesis:T.List[Ray]):
    ''' Get set intersection between two ray clusters with ground truth information ''' 

    ref_frame_rays, ref_frame_inst_labels = get_frame_rays(reference)
    hyp_frame_rays, hyp_frame_inst_labels = get_frame_rays(hypothesis)

    frame_intersections = {label: [] for label in ref_frame_inst_labels} 
    total_intersection_size = 0
    for frame_ident, inst_lables in ref_frame_inst_labels.items():
        for instance_lbl in inst_lables:
            if frame_ident in hyp_frame_inst_labels:
                if instance_lbl in hyp_frame_inst_labels[frame_ident]:
                    frame_intersections[frame_ident].append(instance_lbl)
                    total_intersection_size += 1 
            else: # frame ident not in hyp cluster at all
                pass

    return total_intersection_size, frame_intersections 

def get_coverage(ref_clusters:T.Union[T.List[Cluster], T.List[T.List[Ray]]], 
                 hyp_clusters:T.Union[T.List[Cluster], T.List[T.List[Ray]]]):
    '''Compute cluster coverage metric between list of Reference ground truth clusters and the Hypothesised clustering'''

    if isinstance(ref_clusters[0], Cluster):
        ref_clusters = [cl.rays for cl in ref_clusters]
    if isinstance(hyp_clusters[0], Cluster):
        hyp_clusters = [cl.rays for cl in hyp_clusters]

    total_max_intersection = 0  # numerator
    total_ref_size = 0 # denom 
    for r in ref_clusters:
        intersection_sizes = []
        for h in hyp_clusters:
            intersection_size, frame_intersections = get_cluster_instersection_size(r,h)
            intersection_sizes.append(intersection_size)
        total_max_intersection += np.max(intersection_sizes)
        total_ref_size += len(r)
    
    return total_max_intersection / total_ref_size
        


if __name__ == '__main__':
    # test coverage
    
    # some clusters ground truth; single cluster, hypothesis 2 clusters
    FRAMES = 10
    NUM_INSTANCES = 2
    ref_cluster_a = []
    hyp_cluster_a = []
    hyp_cluster_b = []
    total_rays = 0
    for frame_idx in range(FRAMES):
        for cam_idx in range(5):
            for instance_idx in  range(NUM_INSTANCES):
                cam_label = 'cam'+str(cam_idx)
                instance_label = 'instance_'+str(instance_idx)
                if cam_idx < 3: 
                    r = Ray(str(frame_idx), cam_label, 0, 'class0', 1.0, None, np.array([0,0,0]), 
                            np.array([1,1,1]), global_instance = instance_label)
                    ref_cluster_a.append(r)
                    if frame_idx % 2 == 0:
                        hyp_cluster_a.append(r)
                    else:
                        hyp_cluster_b.append(r)
                    total_rays += 1

    ref_clusters = [ref_cluster_a]
    hyp_clusters = [hyp_cluster_a, hyp_cluster_b]

    print(f"total rays count: {total_rays}")
    intersection_size, frame_intersections = get_cluster_instersection_size(ref_cluster_a, hyp_cluster_a)
    print("intersection size:", intersection_size)
    print(frame_intersections)   

    print("coverage", get_coverage(ref_clusters, hyp_clusters))
    print("purity", get_coverage(hyp_clusters, ref_clusters))