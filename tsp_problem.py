import random
import time
import numpy as np
from python_tsp.heuristics import solve_tsp_simulated_annealing
import lkh


def run_tsp_algorithms(distance_matrix, n, seed):

    dist_matrix = distance_matrix
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
    route, cost = solve_tsp_simulated_annealing(dist_matrix)
    end_lib = time.time()
    start_aco = time.time()
    dist_matrix_safe = np.where(dist_matrix == 0, 1e-10, dist_matrix)
    aco = AntColony(dist_matrix_safe)
    aco_route, aco_dist = aco.run()
    end_aco = time.time()


    start_nn = time.time()
    nn_route = nearest_neighbor()
    end_nn = time.time()
    nn_dist = route_distance(nn_route)

    def write_tsplib(distance_matrix, filename="problem.tsp"):
        n = len(distance_matrix)
        with open(filename, "w") as f:
            f.write("NAME: ATSP\n")
            f.write("TYPE: ATSP\n")
            f.write(f"DIMENSION: {n}\n")
            f.write("EDGE_WEIGHT_TYPE: EXPLICIT\n")
            f.write("EDGE_WEIGHT_FORMAT: FULL_MATRIX\n")
            f.write("EDGE_WEIGHT_SECTION\n")
            for row in distance_matrix:
                f.write(" ".join(str(int(x)) for x in row) + "\n")
            f.write("EOF\n")

    write_tsplib(dist_matrix)
    problem = lkh.LKHProblem.load("problem.tsp")

    start_lkh = time.time()
    route = lkh.solve(solver='E:/TU/Sem/research-methods/lkh/LKH-3.exe', problem=problem, runs=20, seed=seed)
    end_lkh = time.time()
    tour = [city - 1 for city in route[0]]

    def compute_tour_cost(tour, distance_matrix):
        cost = 0
        for i in range(len(tour)):
            from_node = tour[i]
            to_node = tour[(i + 1) % len(tour)]  # wrap around to start
            cost += distance_matrix[from_node][to_node]
        return cost

    cost_lkh = compute_tour_cost(tour, dist_matrix)

    # === Runtime Summary ===
    print(f"Nearest Neighbor:       {nn_dist:.2f} km, Time: {end_nn - start_nn:.2f}s")
    print(f"Ant Colony Optimization:{aco_dist:.2f} km, Time: {end_aco - start_aco:.2f}s")
    print(f"Lib Optimization:{cost:.2f} km, Time: {end_lib - start_lib:.2f}s")
    print(f"LKH Optimization:{cost_lkh:.2f} km, Time: {end_lkh - start_lkh:.2f}s")

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
            "Distance (km)": round(cost, 2),
            "Time (s)": round(end_lib - start_lib, 2)
        },
        "LKH Algorithm": {
            "Distance (km)": round(cost_lkh, 2),
            "Time (s)": round(end_lkh - start_lkh, 2)
        }
    }