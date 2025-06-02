import random
import time
import numpy as np
from math import radians, sin, cos, sqrt, atan2
from random import uniform
from shapely.geometry import Point
import geopandas as gpd
from python_tsp.heuristics import solve_tsp_local_search
from shapely.lib import unary_union


def run_tsp_algorithms(seed, problem_size, plot: bool = False):
    random.seed(seed)

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

    # Example usage
    locations, dist_matrix = generate_augmented_tsp_instance(problem_size)
    dist_matrix_safe = np.where(dist_matrix == 0, 1e-10, dist_matrix)
    n = len(locations)

    if plot:
        from plot_tsp_locations import plot_locations
        plot_locations(locations)
        exit()

    # Route distance
    def route_distance(route):
        return sum(dist_matrix[route[i]][route[(i+1) % len(route)]] for i in range(len(route)))

    # === Nearest Neighbor ===
    def nearest_neighbor():
        unvisited = set(range(1, n))
        route = [0]
        while unvisited:
            last = route[-1]
            next_city = min(unvisited, key=lambda city: dist_matrix[last][city])
            route.append(next_city)
            unvisited.remove(next_city)
        return route

    # === Ant Colony Optimization ===
    class AntColony:
        def __init__(self, dist_matrix, n_ants=50, n_best=5, n_iterations=100, decay=0.1, alpha=1, beta=2):
            self.dist_matrix = dist_matrix
            self.pheromone = np.ones(dist_matrix.shape) / len(dist_matrix)
            self.n_ants = n_ants
            self.n_best = n_best
            self.n_iterations = n_iterations
            self.decay = decay
            self.alpha = alpha
            self.beta = beta
            self.all_inds = range(len(dist_matrix))

        def run(self):
            best_path = None
            best_cost = float("inf")
            for _ in range(self.n_iterations):
                paths = self._all_paths()
                self._spread_pheromone(paths)
                best_iter = min(paths, key=lambda x: x[1])
                if best_iter[1] < best_cost:
                    best_path, best_cost = best_iter
                self.pheromone *= (1 - self.decay)
            return best_path, best_cost

        def _spread_pheromone(self, paths):
            for path, cost in sorted(paths, key=lambda x: x[1])[:self.n_best]:
                for i in range(len(path) - 1):
                    self.pheromone[path[i]][path[i+1]] += 1.0 / self.dist_matrix[path[i]][path[i+1]]

        def _all_paths(self):
            paths = []
            for _ in range(self.n_ants):
                path = self._construct_path(0)
                cost = sum(self.dist_matrix[path[i]][path[i+1]] for i in range(len(path)-1)) + self.dist_matrix[path[-1]][path[0]]
                paths.append((path, cost))
            return paths

        def _construct_path(self, start):
            path = [start]
            visited = set(path)
            while len(path) < len(self.dist_matrix):
                curr = path[-1]
                probs = self._probabilities(curr, visited)
                next_city = np.random.choice(self.all_inds, p=probs)
                path.append(next_city)
                visited.add(next_city)
            return path

        def _probabilities(self, curr, visited):
            pher = self.pheromone[curr]
            dist = self.dist_matrix[curr]
            dist = np.where(dist == 0, 1e-10, dist)
            pher = np.where([i in visited for i in self.all_inds], 0, pher)
            prob = pher**self.alpha * (1.0 / dist)**self.beta
            prob_sum = prob.sum()
            if prob_sum == 0:
                prob = np.ones_like(prob)
                prob_sum = prob.sum()
            return prob / prob_sum


    start_lib = time.time()
    route, cost = solve_tsp_local_search(dist_matrix)
    end_lib = time.time()
    start_aco = time.time()
    aco = AntColony(dist_matrix_safe)
    aco_route, aco_dist = aco.run()
    end_aco = time.time()


    start_nn = time.time()
    nn_route = nearest_neighbor()
    end_nn = time.time()
    nn_dist = route_distance(nn_route)


    # === Runtime Summary ===
    print(f"Nearest Neighbor:       {nn_dist:.2f} km, Time: {end_nn - start_nn:.2f}s")
    print(f"Ant Colony Optimization:{aco_dist:.2f} km, Time: {end_aco - start_aco:.2f}s")
    print(f"Lib Optimization:{aco_dist:.2f} km, Time: {end_lib - start_lib:.2f}s")

    return {
        "Nearest Neighbor": {
            "Distance (km)": round(nn_dist, 2),
            "Time (s)": round(end_nn - start_nn, 2)
        },
        "Ant Colony Optimization": {
            "Distance (km)": round(aco_dist, 2),
            "Time (s)": round(end_aco - start_aco, 2)
        },
        "Lib Optimization": {
            "Distance (km)": round(aco_dist, 2),
            "Time (s)": round(end_lib - start_lib, 2)
        }
    }