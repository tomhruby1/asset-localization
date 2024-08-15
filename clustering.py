import typing as T

import numpy as np
from sklearn.cluster import DBSCAN

from data_structures import Ray, Point, Cluster

def clustering_dbscan(dist_spatial, min_samples, eps, points_filtered, semantic_cluster_splitting=True) -> T.Tuple[T.List[Cluster], dict]:
    ''' Run sklearn DBSCAN implementation. Expected (euclidean?) distance matrix

        args:
            - dist_spatial, dist_semantic: distance matrices
            - points_filtered: midpoint hypotheses
    '''
    cluster_model = DBSCAN(min_samples=min_samples, eps=eps, metric='precomputed')
    cluster_model.fit(dist_spatial)

    cluster_labels = np.unique(cluster_model.labels_)
    print(f"{len(cluster_labels)} clusters found: {cluster_labels}")

    midpoints_clustered = {label: [] for label in cluster_labels}
    for i, l in enumerate(cluster_model.labels_):
        midpoints_clustered[l].append(points_filtered[i])
    midpoint_clusters = [ (points, id) for id, points in midpoints_clustered.items()]


    # SEMANTIC CLUSTER SPLITTING; split spatial cluster based on point's hypotheses
    if semantic_cluster_splitting:
        midpoint_clusters = []
        for cluster_label, cluster_points in midpoints_clustered.items():
            if cluster_label == -1: continue
            cluster_split = {}
            # A) based on features -- this is better, but classifier is weird
            for p in cluster_points:
                cls = np.argmax(p.cls_feature)
                if cls not in cluster_split:
                    cluster_split[cls] = []
                cluster_split[cls].append(p)
            for cls, points in cluster_split.items():
                # use the same minimum points pts param as DBSCAN uses 
                if len(points) > min_samples:
                    midpoint_clusters.append(Cluster(points, len(midpoint_clusters)))
            # B) based on detector?

    return midpoint_clusters, {} # no stats.. TODO

    
def clustering_bihierarchical(dist_spatial:np.ndarray, dist_semantic:np.ndarray, points_filtered:T.List[Point], rays:T.List[Ray],
                              spat_t=5, sema_t=0.28, alpha=0.4, metric='min', ray_connection_pruning=False)  -> T.Tuple[T.List[Cluster], dict]:
    ''' Run custom hierarchical bi-modal clustering

        args:
            - dist_spatial, dist_semantic: distance matrices
            - points_filtered: midpoint hypotheses
            - ray_connection_pruning: prune all other hypotheses from the clustering algorithm, that share ray with currently merged
    '''
    
    if ray_connection_pruning:
        
        # this is probably not needed when filtering based on ray-density, but whatever
        import bimodal_clustering.custom_midpoint_aggl_clustering as custom_aggl_clustering # this one with pruning

        ## prepare ray-connection # TODO: eventually get rid of this using nicer data-structures
        ray_2_midpoints = [[] for i in range(len(rays))] # list of points corresponding to each ray
        for r_idx, ray in enumerate(rays):
            for p_idx, point in enumerate(points_filtered):
                if point.r1 == ray or point.r2 == ray:
                    ray_2_midpoints[r_idx].append(p_idx)
        # compact it to map: (point -> list of points) (correpsondance through rays)
        ray_connection = [[] for i in range(len(points_filtered))] 
        for p_idx, point in enumerate(points_filtered):
            for r_idx, ray in enumerate(rays):
                if ray == point.r1 or ray == point.r2:
                    ray_connection[p_idx].extend(ray_2_midpoints[r_idx])
                    ray_connection[p_idx].extend(ray_2_midpoints[r_idx])

        clusters, labels, distances, stats = custom_aggl_clustering.main(dist_spatial, dist_semantic, ray_connection,
                                                                         spat_T=spat_t, sema_T=sema_t, alpha=alpha, metric=metric)
    else:        
        import bimodal_clustering.custom_aggl_clustering as custom_aggl_clustering

        clusters, labels, distances, stats = custom_aggl_clustering.main(dist_spatial, dist_semantic, 
                                                                         spat_T=spat_t, sema_T=sema_t, alpha=alpha, metric=metric)
    print("bi-modal hierarchical clustering done")

    ## build the clusters
    midpoint_clusters = [None] * len(clusters)
    for i, cl_elements in enumerate(clusters): 
        cl_points = [points_filtered[el] for el in cl_elements]
        midpoint_clusters[i] = Cluster(cl_points, i)


    return midpoint_clusters, stats