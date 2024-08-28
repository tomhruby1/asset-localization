import numpy as np
import time

np.set_printoptions(precision=20, suppress=True)


# TODO: think through the cluster distance w.r.t. the multimodality
def max_dist(i, clusters, dist_orig):
    '''get max dist from i-th clusters to other clusters from the original distance matrix
    
    args:
        - i: cluster index
        - clusters: list of clusters containing indices corresponding to elements in dist
        - dist: 2xNxN distance matrix 
    
    returns: A max for both the spatial and semantic distance #TODO: idk if gut
    '''
    
    dists = np.ones((2,len(clusters))) * -1
    for j, cl in enumerate(clusters):
        if j != i:
            # dist_ij = np.max([dist_orig[:, i, el_id] for el_id in cl], axis=0) 
            # max distance between cluster[i] and cl
            dist_ij = np.max([dist_orig[:, i_element, cl_element] 
                              for cl_element in cl for i_element in clusters[i]], axis=0)
        else:
            dist_ij = np.array([0.0, 0.0])
        dists[:,j] = dist_ij
    assert -1 not in dists
    
    return dists

def cluster_dist(i, clusters, dist_orig, mode='max'):
    '''get max dist from i-th clusters to other clusters from the original distance matrix
    
    args:
        - i: cluster index
        - clusters: list of clusters containing indices corresponding to elements in dist
        - dist: 2xNxN distance matrix 
    
    returns: Max/avg/min for both the spatial and semantic distance #TODO: idk if gut
    '''
    
    dists = np.ones((2,len(clusters))) * -1
    for j, cl in enumerate(clusters):
        if j != i:
            # dist_ij = np.max([dist_orig[:, i, el_id] for el_id in cl], axis=0) 
            # max distance between cluster[i] and cl
            if mode == 'max':
                dist_ij = np.max([dist_orig[:, i_element, cl_element] 
                                  for cl_element in cl for i_element in clusters[i]], axis=0)
            elif mode == 'avg':
                dist_ij = np.average([dist_orig[:, i_element, cl_element] 
                                      for cl_element in cl for i_element in clusters[i]], axis=0)
            elif mode == 'min':
                dist_ij = np.min([dist_orig[:, i_element, cl_element] 
                                  for cl_element in cl for i_element in clusters[i]], axis=0)
            else:
                raise Exception('Unknown mode')
            
        else:
            dist_ij = np.array([0.0, 0.0])
        dists[:,j] = dist_ij
    assert -1 not in dists

    return dists    

def main(dists_spatial:np.ndarray, dists_semantic:np.ndarray, spat_T=0.2, sema_T=0.8, alpha=0.5, metric='max'):
    '''
    
    returns: list of lists containing input idx clustered, and corresponding distance matrix between the clusters
    '''


    N = dists_spatial.shape[0]
    assert N == dists_spatial.shape[1] == dists_semantic.shape[0] == dists_semantic.shape[1]
    
    clusters = [[i] for i in range(N)] # init: idx w.r.t the initial distances matrices indexes
    iter_times = []
    
    # TODO: order the matrix for a speedup?, make it faster: # https://stackoverflow.com/questions/66704859/increase-speed-of-finding-minimum-element-in-a-2-d-numpy-array-which-has-many-en
    # Set the indices of the upper triangle including the diagonal to inf
    # for now ignore TODO: add  --> measure potential speedup?
    # row_indices, col_indices = np.triu_indices(N)
    # dists_spatial[row_indices, col_indices] = np.inf  
    # dists_semantic[row_indices, col_indices] = np.inf

    # dists_spatial *= (1-alpha)
    # dists_semantic *= alpha
    
    dist_orig = np.array([dists_spatial, dists_semantic]) # concat spatial and semantic distances 
    dist = dist_orig.copy()
    # print(dist)
    threshold = np.array([spat_T, sema_T]) # alpha should not be included in thresholding?? or maybe compensate
    
    counter = 0
    while len(clusters) > 1: 
        iter_t = time.time()
        
        dist_joint = (1-alpha)*dist[0,:,:] + alpha*dist[1,:,:]
        N = len(clusters)
        assert len(dist_joint) == N

        tri_N = N*(N-1) // 2 # number of elements in triangular matrix
        row_indices, col_indices = np.triu_indices(N)
        dist_joint[row_indices, col_indices] = np.inf

        # find closest pair in joint distances that is lower than threshold
        indices_sorted = np.dstack(np.unravel_index(np.argsort(dist_joint.ravel()), dist_joint.shape))
        indices_sorted = indices_sorted[0,:tri_N,:]
        found = False
        for (i_select, j_select) in indices_sorted: # select i,j pair indexing the current dist matrix and clusters
            if np.all(dist[:, i_select, j_select] < threshold):
                d = dist[:, i_select, j_select]
                # print(f"d = {d}")
                found = True
                break
            else:
                # print(f"d = {dist[:, i_select, j_select]} does not fullfill the threshold condition")
                pass
        # end condition --> nothing to merge
        if not found:
            print(f"Finishing clustering, last checked distance: d = {dist[:, i_select, j_select]}")
            break

        # print(f"merging {(i_select, j_select)}: {clusters[i_select]} <-- {clusters[j_select]} d={dist[:, i_select, j_select]}")
        ## MERGUJ!!
        clusters[i_select].extend(clusters.pop(j_select))

        dist = np.delete(dist, j_select, axis=1) # row
        dist = np.delete(dist, j_select, axis=2) # column
        if len(clusters) == 1:
            break
        if i_select > j_select: # decrement the selected i if after deleted row
            i_select -= 1  
        # recompute dist to all other clusters in place of cl
        i_to_clusters_dist = cluster_dist(i_select, clusters, dist_orig, mode=metric)
        dist[:, i_select, :] = i_to_clusters_dist
        dist[:, :, i_select] = i_to_clusters_dist

        counter += 1
        iter_times.append(time.time() - iter_t)

        # print(clusters)

    print("finished\n")
    print(f"{len(clusters)} clusters found:")
    print(clusters)
    print("dists")
    print(dist) 

    ## return also list of labels for each of the elements
    labels = [-1] * len(dists_spatial)
    for cl_id, cluster in enumerate(clusters):
        for el in cluster:
            labels[el] = cl_id

    ## stats
    stats = {
        "iter_time": iter_times
    }

    return clusters, labels, dist, stats



## DEMO
if __name__=='__main__':
    ## test
    import matplotlib.pyplot as plt
    from sklearn.metrics.pairwise import cosine_distances, euclidean_distances

    def plot_2d_clusters(clusters, semantic=None, result_clusters=None, titles=['2D Clusters of Points', 'Result']):
        '''plot list of lists -- euklidean clusters, optionally semantic class (integer) can be passd'''    
    
        # Plot the clusters
        fig, ax = plt.subplots(2,2,figsize=(8,6))
        ax = ax.flatten()

        colors = [
            (1.0, 0.388, 0.278, 1.0), # - Tomato
            (0.255, 0.412, 0.882, 1.0), # - Royal Blue
            (1.0, 0.647, 0.0, 1.0), # - Orange
            (0.282, 0.819, 0.8, 1.0), # - Medium Aquamarine
            (1.0, 0.753, 0.796, 1.0), # - Pink
            (0.235, 0.702, 0.443, 1.0), # - Medium Sea Green
            (1.0, 0.549, 0.0, 1.0), # - Dark Orange
            (0.482, 0.408, 0.933, 1.0), # - Medium Slate Blue
            (1.0, 0.078, 0.576, 1.0), # - Deep Pink
        ]
        # ground truth
        for i in range(len(clusters)):
            if semantic is None:
                ax[0].scatter(clusters[i][:, 0], clusters[i][:, 1], c=colors[i%len(colors)], label=f'Cluster {i+1}')
            else:
                cs = [colors[cls] for cls in semantic[i]]
                ax[0].scatter(clusters[i][:, 0], clusters[i][:, 1], c=cs, label=f'Cluster {i+1}')
                ax[0].title.set_text(titles[0])
                ax[0].grid(True)
                ax[0].axis('equal')

        # results for comparison
        if result_clusters is not None:
            for k, clusters in enumerate(result_clusters):
                for i in range(len(clusters)):
                    ax[k+1].scatter(clusters[i][:, 0], clusters[i][:, 1], 
                                    color=colors[i%len(colors)], label=f'Cluster {i+1}')
                ax[k+1].title.set_text(titles[k+1])
                ax[k+1].legend()
                ax[k+1].grid(True)
                ax[k+1].axis('equal')

                if k == 3: break

        # ax.title.set_text(titles[0])
        # ax.legend()
        # ax.grid(True)
        # ax.set
        # plt.xlabel('X')
        # plt.ylabel('Y')
        plt.show()




    centers = np.array([[1, 1],
                        [18, 2],
                        [2, 18],
                        [18, 18]])

    # Define number of samples for each cluster
    num_samples_per_cluster = np.array([4,3,2,1]) #np.array([10, 7, 8, 6])
                                # [0,1,2,3] [4,5,6] [7,8] [9]
    # Function to generate points for each cluster
    def generate_cluster(center, num_samples):
        return np.random.randn(num_samples, 2) + center

    # Generate points for each cluster
    points_gt = [generate_cluster(center, num_samples) 
                 for center, num_samples in zip(centers, num_samples_per_cluster)]
    
    points_semantic_gt = []
    for cluster in points_gt:
        points_semantic_gt.append([1 if i % 2 == 0 else 0 for i in range(len(cluster))])
    
    # plot_2d_clusters(points_gt, semantic=points_semantic_gt, title='Ground truth clusters')

    # prepare input points
    points = [item for cluster in points_gt for item in cluster] # spatial
    points_sem = np.array([[item] for cluster in points_semantic_gt for item in cluster])
    
    # shuffle
    points_combined = list(zip(points, points_sem))
    np.random.shuffle(points_combined) 
    points, points_sem = zip(*points_combined)

    # calc distances
    dist1 = euclidean_distances(points, points)
    # dist2 = np.ones(len(points)) - np.eye(len(points))
    dist_sem = euclidean_distances(points_sem, points_sem)
    
    
    # CLUSTER
    # set alpha to 0 and spat_T absurdly large to cluster semantically only
    # ^^^ works
    SPAT_T_0 = 666; SEMA_T_0 = 0.1; ALPHA_0=0
    clusters_0, labels_0, dists_0, stats_0 = main(dist1, dist_sem, spat_T=SPAT_T_0, alpha=ALPHA_0, sema_T=SEMA_T_0, metric='avg')
    print(f"1. clustering found {len(clusters_0)} clusters")
    clustered_points_0 = [np.array([points[el_id] for el_id in cl]) for cl in clusters_0]

    SPAT_T_1 = 5; SEMA_T_1 = 0.1; ALPHA_1=0.3
    clusters_1, labels_1, dists_1, stats_1 = main(dist1, dist_sem, spat_T=SPAT_T_1, alpha=ALPHA_1, sema_T=SEMA_T_1, metric='avg')
    print(f"2. clustering found {len(clusters_1)} clusters")
    clustered_points_1 = [np.array([points[el_id] for el_id in cl]) for cl in clusters_1]

    SPAT_T_2 = 5; SEMA_T_2 = 999; ALPHA_2=1.0
    clusters_2, labels_2, dists_2, stats_2 = main(dist1, dist_sem, spat_T=SPAT_T_2, alpha=ALPHA_2, sema_T=SEMA_T_2, metric='avg')
    print(f"2. clustering found {len(clusters_1)} clusters")
    clustered_points_2 = [np.array([points[el_id] for el_id in cl]) for cl in clusters_2]
    
    plot_2d_clusters(points_gt, semantic=points_semantic_gt, result_clusters=[clustered_points_0, clustered_points_1, clustered_points_2], 
                     titles=['Data', 
                             f'$ T_2 = {SEMA_T_0}, \\alpha={ALPHA_0} $',
                             f'$ T_1 = {SPAT_T_1}, T_2 = {SEMA_T_1}, \\alpha={ALPHA_1} $', 
                             f'$ T_1 = {SPAT_T_2}, \\alpha={ALPHA_2} $',
                            ])
    
    
    # clustered_points = [np.array([points[el_id] for el_id in cl]) for cl in clusters_1]
    # plot_2d_clusters(points_gt, semantic=points_semantic_gt, result_clusters=clustered_points, 
    #                  titles=['Ground truth clusters', 'Resulting Clusters'])
    # print()