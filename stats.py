import matplotlib.pyplot as plt
import numpy as np 
from pathlib import Path
from scipy.stats import norm
import json

def plot_distances_along_single_ray_from_closest_point(work_dir:Path, show=False):
    '''plot and store histogram of distances of points along their rays wrt. 
       to closest points on ray for each ray across currently processed dataset
    '''

    with open(work_dir/'ray_point_dists.json', 'r') as f:
        ray_point_dists = json.load(f) # ray_id (assigned corresp. to raycasting_result.rays) --> distance for each point along the ray
    with open(work_dir/'ray_point_data.json', 'r') as f:
        ray_point_data = json.load(f)
    with open(work_dir/'ray_point_closest_to_lm.json', 'r') as f:
        ray_point_closest_to_lm = json.load(f)
    
    # normalized distances of points along their rays wrt. to closest points on ray
    all_points_along_rays = [] 
    all_points_along_rays_weights = [] 
    for rid in ray_point_closest_to_lm:
        closest_point_on_ray = ray_point_closest_to_lm[rid]['point_ray_distance']
        ray_points = ray_point_dists[rid]
        for i, p in enumerate(ray_points):
            all_points_along_rays.append((p - closest_point_on_ray))
            all_points_along_rays_weights.append(ray_point_data[str(rid)]['weight'][i])


    # stem = plt.stem(all_points_along_rays, all_points_along_rays_weights)
    # stem[1].set_linewidth(0.5)
    # stem[1].set_linestyles("dotted")
    # plt.show()
    # plt.scatter(all_points_along_rays, all_points_along_rays_weights)

    hist = plt.hist(all_points_along_rays, bins=100, density=True, alpha=0.5, label="unweighted histogram")
    histw = plt.hist(all_points_along_rays, bins=100, density=True, alpha=0.5, weights=all_points_along_rays_weights, label="weighted histogram") 

    # gaussian pdf --> prirazeni momentu
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    mean = np.mean(all_points_along_rays)
    std_dev = np.std(all_points_along_rays)
    w_mean = np.average(all_points_along_rays, weights=all_points_along_rays_weights)
    w_std_dev = np.sqrt(np.average((all_points_along_rays-w_mean)**2, weights=all_points_along_rays_weights))
    print(f"number of points: {len(all_points_along_rays)}")
    print(f"mean: {mean}, std_dev: {std_dev}")
    print(f"w_mean: {w_mean}, w_std_dev: {w_std_dev}")
    p = norm.pdf(x, mean, std_dev)
    plt.plot(x, p, 'blue', linewidth=1, 
             label=f"unweighted pdf;\n    $\\mathcal{{N}}(\\mu={mean:.2f}, \\sigma={std_dev:.2f})$")
    p = norm.pdf(x, w_mean, w_std_dev)
    plt.plot(x, p, 'orange', linewidth=1, 
             label=f"weighted pdf;\n   $\\mathcal{{N}}(\\mu={w_mean:.2f}, \\sigma={w_std_dev:.2f})$")
    plt.xlabel("normalized distance from $p*$")
    plt.legend()
    plt.savefig(work_dir/'distances_along_ray_from_closest_point.pdf')
    plt.show()