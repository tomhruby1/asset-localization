import copy

import numpy as np

def fuzzy_PP(clusters, landmarks, t=1, t_max=3, planar=True):
    ''' 
    Calculate the fuzzy 'pure' positives for a cluster of means and a set of ground truth landmarks.

    args:
        - clusters: list of Cluster objects
        - landmarks: dict of landmark_id -> {landmark_info}
        - t: target threshold at which the TP = 1.0
        - t_max: max threshold at which the TP = 0.0
        - planar: if True, the distance is calculated in 2D
    '''
    matches = {l:None for l in landmarks}
    unnassigned_clusters = copy.deepcopy(clusters)
    PP_val = 0.0

    for l_id, landmark in landmarks.items():
        if planar:
            cluster_dists = [np.linalg.norm(landmark['projected_coord'][:2] - c.centroid[:2])
                             for c in unnassigned_clusters if c.category == landmark['class']]
        else:
            cluster_dists = [np.linalg.norm(landmark['projected_coord'] - c.centroid)
                             for c in unnassigned_clusters if c.category == landmark['class']]
          
        if len(cluster_dists) > 0:
            cl_idx = np.argmin(cluster_dists)
            selected_cluser = unnassigned_clusters[cl_idx]
            if cluster_dists[cl_idx] < t_max:
                # check that no other clusters with the same category are within the threshold
                of_same_class_within = [np.linalg.norm(landmark['projected_coord'] - c.centroid) < t_max 
                                        for c in clusters if c.category == landmark['class']]
                if sum(of_same_class_within) == 1:
                    matches[l_id] = selected_cluser.id
                    unnassigned_clusters.pop(cl_idx)
                    d = cluster_dists[cl_idx]
                    pp = 1.0 if cluster_dists[cl_idx] < t else -1.0*d / (t_max - t) + t_max / (t_max - t)
                    PP_val += pp
                    print(f"PP for landmark {l_id}: {pp}")
        
        if matches[l_id] is None:
            print(f"no match for landmark {l_id}")

    return PP_val, matches


def test():
    ''' test the fuzzy_TP function '''
    from dataclasses import dataclass
    import numpy as np

    @dataclass
    class Cluster:
        id: int
        centroid: np.ndarray
        category: str

    landmarks = {0: {'projected_coord': np.array([0, 0]), 'class': 'A'},
                 1: {'projected_coord': np.array([4, 4]), 'class': 'A'},
                 2: {'projected_coord': np.array([4, 4]), 'class': 'B'},
                 3: {'projected_coord': np.array([12, 12]), 'class': 'C'}}

    clusters = [Cluster(0, np.array([0, 0]), 'A'),
                Cluster(1, np.array([3, 3]), 'A'),
                Cluster(2, np.array([4, 4]), 'C'),
                Cluster(3, np.array([6, 6]), 'B')]

    TP, matches = fuzzy_PP(clusters, landmarks, t=1, t_max=3, planar=True)
    print(f"TP: {TP}")
    print(f"matches: {matches}")

    # assert TP == 6.0
    # assert all([matches[l] is not None for l in landmarks])


if __name__=='__main__':
    test()