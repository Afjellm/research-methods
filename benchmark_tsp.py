import random

import numpy as np

from tsp_problem import run_tsp_algorithms
import pandas as pd
from math import radians, sin, cos, sqrt, atan2
from random import uniform
from shapely.geometry import Point
import geopandas as gpd
from shapely.lib import unary_union


def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

# Step 2: Generate clustered locations in Vienna
def generate_clustered_and_random_austria_locations(count_clustered, count_random, cluster_stddev=0.1):
    """
    Generate clustered and random location points within Austria.

    Args:
        count_clustered (int): Number of clustered points to generate.
        count_random (int): Number of scattered random points to generate.
        cluster_stddev (float): Standard deviation for clustered points.
        austria_geojson_path (str): Path to the Austria GeoJSON file.

    Returns:
        list of tuples: (name, latitude, longitude)
    """

    # Load Austria geometry
    austria_gdf = gpd.read_file("austria_districts.geojson")
    austria_union = unary_union(austria_gdf.geometry)  # Ensure it's a single GeometryCollection or MultiPolygon

    # Define clusters
    cluster_centers = [
        ("Cluster_West", 47.26, 11.39),  # Innsbruck
        ("Cluster_North", 48.31, 14.29),  # Linz
        ("Cluster_Center", 47.81, 13.04),  # Salzburg
        ("Cluster_South", 46.62, 14.31),  # Klagenfurt
        ("Cluster_East", 48.21, 16.37),  # Vienna
        ("Cluster_SE", 47.07, 15.44),  # Graz
    ]

    locations = []

    # Clustered points
    for i in range(count_clustered):
        cluster_name, base_lat, base_lon = random.choice(cluster_centers)
        lat = round(random.gauss(base_lat, cluster_stddev), 6)
        lon = round(random.gauss(base_lon, cluster_stddev), 6)
        name = f"{cluster_name}_Point_{i + 1}"
        locations.append((name, lat, lon))

    # Random points within bounding box, filtered by Austria geometry
    random_points = []
    attempts = 0
    max_attempts = count_random * 20  # safety limit

    while len(random_points) < count_random and attempts < max_attempts:
        lat = round(random.uniform(46.3, 49.1), 6)
        lon = round(random.uniform(9.5, 17.2), 6)
        point = Point(lon, lat)
        if austria_union.contains(point).any():
            name = f"Random_Point_{len(random_points) + 1}"
            random_points.append((name, lat, lon))
        attempts += 1

    return locations + random_points

# Step 3: Compute Haversine distance matrix
def compute_distance_matrix(locations):
    n = len(locations)
    D = np.zeros((n, n))
    for i in range(n):
        _, lat1, lon1 = locations[i]
        for j in range(n):
            _, lat2, lon2 = locations[j]
            D[i][j] = haversine(lat1, lon1, lat2, lon2)
    return D

# Step 4: Apply non-symmetric inflation
def make_asymmetric(D, percent=0.1, asym_factor=1.5):
    n = D.shape[0]
    indices = [(i, j) for i in range(n) for j in range(n) if i != j]
    chosen = random.sample(indices, int(len(indices) * percent))
    for i, j in chosen:
        D[i][j] *= uniform(1, asym_factor)
    return D

# Step 5: Apply river-crossing penalty
def crosses_river(loc1, loc2, river_lon=16.38):
    return (loc1[2] < river_lon and loc2[2] > river_lon) or (loc1[2] > river_lon and loc2[2] < river_lon)

def apply_river_penalty(locations, D, penalty=2.5):
    n = len(locations)
    for i in range(n):
        for j in range(n):
            if i != j and crosses_river(locations[i], locations[j]):
                D[i][j] *= penalty
    return D

# Step 6: Put it all together
def generate_augmented_tsp_instance(n_locations):
    locations = generate_clustered_and_random_austria_locations(n_locations, n_locations / 5)
    D = compute_distance_matrix(locations)
    D = make_asymmetric(D, percent=0.85, asym_factor=10)
    D = apply_river_penalty(locations, D, penalty=2.5)
    return locations, D


def run_multiple_tsp_trials(problem_sizes=[50,100,150,200], n_instances=50, runs_per_seed=10):
    records = []

    for size in problem_sizes:
        for trial in range(n_instances):
            seed = 42 + trial
            random.seed(seed)
            # Example usage
            locations, dist_matrix = generate_augmented_tsp_instance(size)
            n = len(locations)

            for seed in range(runs_per_seed):
                random.seed(seed)
                results = run_tsp_algorithms(dist_matrix, n, trial)
                for algorithm, metrics in results.items():
                    records.append({
                        "trial": trial,
                        "run_seed": seed,
                        "problem_size": size,
                        "algorithm": algorithm,
                        "cost_km": float(metrics["Distance (km)"]),
                        "runtime_s": float(metrics["Time (s)"])
                    })

    df = pd.DataFrame(records)
    return df

if __name__ == "__main__":
    df_results = run_multiple_tsp_trials()
    # Save to CSV
    df_results.to_csv("tsp_results.csv", index=False)

    print("Results saved to tsp_results.csv")