import matplotlib.pyplot as plt
from pathlib import Path
import json 
import numpy as np

def plot_rays(rays, scale=5, figsize=(16,16), labels=None, ax=None, 
              highlight_ray=None, title=None, x_lim=None, y_lim=None, points=None):
    """
    Plots the 2D projection of 3D rays onto the XY plane and automatically adjusts plot limits.

    Args:
    - rays: list of tuples, where each tuple contains:
        - origin: (x, y, z)
        - direction: (dx, dy, dz) (a unit vector)
    - scale: how far the ray should be extended for visualization.
    """
    # Lists to store all projected x and y coordinates for calculating limits
    x_coords = []
    y_coords = []

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    if labels is not None:
        assert len(labels) == len(rays)

    for idx, (origin, direction) in enumerate(rays):
        # Extract the components of the origin and direction
        ox, oy, oz = origin
        dx, dy, dz = direction

        # Project the origin and direction onto the XY plane (ignore z-component)
        x_proj = ox
        y_proj = oy
        x_end = ox + dx * scale
        y_end = oy + dy * scale

        # Plot the ray on the 2D plane
        label = labels[idx] if labels is not None else None
            
        ax.quiver(x_proj, y_proj, x_end - x_proj, y_end - y_proj, 
                  angles='xy', scale_units='xy', scale=1, color='gray', 
                  width=0.002, label=label)

        # Collect x and y coordinates for limits
        x_coords.extend([x_proj, x_end])
        y_coords.extend([y_proj, y_end])

    # one ray highlighted
    if highlight_ray is not None:
        origin, direction = highlight_ray
        ox, oy, oz = origin
        dx, dy, dz = direction

        x_proj = ox
        y_proj = oy
        x_end = ox + dx * scale
        y_end = oy + dy * scale

        ax.quiver(x_proj, y_proj, x_end - x_proj, y_end - y_proj, 
                  angles='xy', scale_units='xy', scale=1, color='green', 
                  width=0.002, label='highlight_ray')
        
    # Automatically determine the limits for x and y axis
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    # Add some padding around the limits for better visualization
    padding_x = (x_max - x_min) * 0.1
    padding_y = (y_max - y_min) * 0.1

    # autolimits
    if x_lim is None:
        x_lim = x_min - padding_x, x_max + padding_x
    if y_lim is None:
        y_lim = y_min - padding_y, y_max + padding_y

    # ray point coords
    if points is not None:
        points = np.array(points)
        plt.scatter(points[:,0], points[:,1], c='g', s=10)
    
    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)

    # Set labels and title
    ax.set_xlabel('Northing')
    ax.set_ylabel('Easting')
    ax.set_title(title)
    
    # Display grid and equal aspect ratio
    ax.grid(True)
    ax.set_aspect('equal')

    if ax is None: 
        plt.show()

    return x_lim, y_lim


def plot_point_dist_along_ray(work_dir: Path, ray_id:int, ax=None):
    
    with open(work_dir/'ray_point_dists.json', 'r') as f:
        ray_point_dists = json.load(f) # ray_id (assigned corresp. to raycasting_result.rays) --> distance for each point along the ray
    with open(work_dir/'ray_point_data.json', 'r') as f:
        ray_point_data = json.load(f)
    with open(work_dir/'ray_point_closest_to_lm.json', 'r') as f:
        ray_point_closest_to_lm = json.load(f)

    x = ray_point_dists[str(ray_id)]
    y = ray_point_data[str(ray_id)]['weight']
    lbls = ray_point_data[str(ray_id)]['class']
    
    cls = ray_point_closest_to_lm[str(ray_id)]['point_label'] # class of the closes point - landmark pair

    x_cls, y_cls = [], [] # same class 
    x_nocls, y_nocls = [], [] # different class
    for i in range(len(x)):
        if lbls[i] == cls:
            x_cls.append(x[i])
            y_cls.append(y[i])
        else:
            x_nocls.append(x[i])
            y_nocls.append(y[i])
   
    if ax is None:
        fig, ax = plt.subplots()
    ax.set_xticks(list(range(0,30,2)))
    ax.set_xlim(0,40)

    ax.stem(x_cls, y_cls, basefmt="g")
    if len(x_nocls) > 0:
        ax.stem(x_nocls, y_nocls, 'c', markerfmt='co') # those, that don't match will be in different color
    plt.xlabel('Depth along ray; $\\lambda$ [m]')
    plt.ylabel('Weight')

    ax.axvline(x=ray_point_closest_to_lm[str(ray_id)]['point_ray_distance'], color='red', linestyle='--')
    # ax.axvline(x=ray_point_closest_to_lm[str(ray_id)]['landmark_projected_coord'], color='magenta', linestyle='--')

    plt.show()
    print(ray_point_closest_to_lm[str(ray_id)])
