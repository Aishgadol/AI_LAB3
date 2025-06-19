"""
MSH (Multi-Stage Heuristic) for Ackley function optimization.
Phase 1: Partition the search space into regions (clusters)
Phase 2: Optimize within each region using local search/metaheuristics
"""

import numpy as np
import random
import time 
import math
LOWER_BOUND = -32.768
UPPER_BOUND = 32.768
DIM = 10  

def f(x):
    # Ackley function value
    a, b, c = 20, 0.2, 2*np.pi
    d = len(x)
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(c * x))
    term1 = -a * np.exp(-b * np.sqrt(sum1 / d))
    term2 = -np.exp(sum2 / d)
    return term1 + term2 + a + np.exp(1)

def N(s, dim=10, lower_bound=LOWER_BOUND, upper_bound=UPPER_BOUND):
    # Generate neighbors for Ackley function optimization
    neighbors = []
    neighbors_size = 200
    operators = ["single_dim", "multi_dim", "gaussian", "origin_pull", "reflect"]
    current_value = f(s)
    base_step_size = 0.1
    if current_value < 5.0:
        adaptive_step = base_step_size * 0.5
    elif current_value < 10.0:
        adaptive_step = base_step_size
    else:
        adaptive_step = base_step_size * 2.0
    while len(neighbors) < neighbors_size:
        operator = random.choice(operators)
        neighbor = np.copy(s)
        match operator:
            case "single_dim":
                idx = random.randint(0, dim-1)
                perturbation = np.random.uniform(-adaptive_step, adaptive_step)
                neighbor[idx] += perturbation
            case "multi_dim":
                indices = random.sample(range(dim), random.randint(2, dim//2))
                for idx in indices:
                    perturbation = np.random.uniform(-adaptive_step, adaptive_step)
                    neighbor[idx] += perturbation
            case "gaussian":
                noise = np.random.normal(0, adaptive_step, dim)
                neighbor += noise
            case "origin_pull":
                pull_strength = random.uniform(0.01, 0.1)
                neighbor = neighbor * (1 - pull_strength)
            case "reflect":
                idx = random.randint(0, dim-1)
                neighbor[idx] = -neighbor[idx]
        neighbor = np.clip(neighbor, lower_bound, upper_bound)
        if not np.array_equal(neighbor, s):
            neighbors.append(neighbor)
    if random.random() < 0.05:
        for _ in range(5):
            random_solution = generate_initial_solution(dim, lower_bound, upper_bound)
            neighbors.append(random_solution)
    return neighbors

def L(s, lower_bound=LOWER_BOUND, upper_bound=UPPER_BOUND):
    return np.all((s >= lower_bound) & (s <= upper_bound))

def S(neighbors):
    legal_neighbors = [n for n in neighbors if L(n)]
    return min(legal_neighbors, key=lambda x: f(x)) if legal_neighbors else None

def LocalSearchStep(s):
    improved = True
    max_trials = 10
    current_solution = np.copy(s)
    best_value = f(current_solution)
    while improved and max_trials > 0:
        max_trials -= 1
        neighbors = N(current_solution)
        best_neighbor = S(neighbors)
        if best_neighbor is not None and f(best_neighbor) < best_value:
            current_solution = np.copy(best_neighbor)
            best_value = f(best_neighbor)
        else:
            improved = False
    return current_solution

def generate_new_solution(s, dim=10, lower_bound=LOWER_BOUND, upper_bound=UPPER_BOUND):
    perturbation = np.random.uniform(-0.1, 0.1, dim)
    new_values = s + perturbation
    new_values = np.clip(new_values, lower_bound, upper_bound)
    return new_values

def generate_initial_solution(dim=10, lower_bound=LOWER_BOUND, upper_bound=UPPER_BOUND):
    return np.random.uniform(lower_bound, upper_bound, dim)

def partition_ackley_space(dim, lower, upper, num_clusters):
    """
    Partition the search space into num_clusters hypercubes (regions).
    Returns a list of (lower_bound, upper_bound) tuples for each region.
    """
    # For simplicity, partition each dimension into k = ceil(num_clusters^(1/dim)) bins
    k = int(np.ceil(num_clusters ** (1/dim)))
    bins = np.linspace(lower, upper, k+1)
    regions = []
    # Generate all combinations of bins for each dimension
    from itertools import product
    for idxs in product(range(k), repeat=dim):
        region_lower = np.array([bins[i] for i in idxs])
        region_upper = np.array([bins[i+1] for i in idxs])
        regions.append((region_lower, region_upper))
        if len(regions) >= num_clusters:
            break
    return regions

def region_based_assign(population, regions):
    """
    Assign each solution to the region (cluster) it falls into.
    Returns dict: cluster_id -> list of solutions
    """
    clusters = {i: [] for i in range(len(regions))}
    for x in population:
        assigned = False
        for i, (lb, ub) in enumerate(regions):
            if np.all(x >= lb) and np.all(x <= ub):
                clusters[i].append(x)
                assigned = True
                break
        if not assigned:
            # Optionally assign to nearest region
            distances = [np.linalg.norm(x - (lb + ub)/2) for (lb, ub) in regions]
            closest = np.argmin(distances)
            clusters[closest].append(x)
    return clusters

def value_function_ackley(x):
    """Assign higher value to solutions closer to the global minimum (origin)."""
    # Lower Ackley value is better, so invert and shift
    return -f(x)

def mkp_assign_ackley(population, num_clusters):
    """
    Assign solutions to clusters (vehicles) using a greedy multi-knapsack style heuristic.
    Returns a dict: cluster_id -> list of solutions.
    """
    clusters = {i: [] for i in range(num_clusters)}
    # Sort by value (best first)
    sorted_pop = sorted(population, key=value_function_ackley, reverse=True)
    for idx, sol in enumerate(sorted_pop):
        clusters[idx % num_clusters].append(sol)
    return clusters

def msh_ackley_solver(dim=10, lower=-32.768, upper=32.768, num_clusters=8,
                      meta_heuristic='local', time_limit=60, max_searches=1000, seed=None, pop_size=80, refinement_stages=2):
    """
    Multi-Stage Heuristic for Ackley function (true analog to CVRP MSH):
    1. Generate a population of initial solutions
    2. Assign solutions to clusters using a value function (MKP-style)
    3. Optimize each cluster using a metaheuristic
    4. Repeat assignment and optimization for several stages
    5. Return the best solution found
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    # Phase 1: Partition space
    regions = partition_ackley_space(dim, lower, upper, num_clusters)

    # Phase 2: Generate initial population
    population = [generate_initial_solution(dim, lower, upper) for _ in range(pop_size)]
    best_solution = None
    best_value = float('inf')

    for stage in range(refinement_stages):

        # Phase 3: Assign population to regions (clusters)
        clusters = region_based_assign(population, regions)

        # Phase 4: Optimize each region's solutions
        new_population = []
        for cluster_id, sols in clusters.items():
            for s in sols:
                if meta_heuristic == 'local':
                    s_opt = LocalSearchStep(s)
                else:
                    s_opt = s
                new_population.append(s_opt)
                if L(s_opt) and f(s_opt) < best_value:
                    best_solution = np.copy(s_opt)
                    best_value = f(s_opt)

        # Phase 5: Prepare for next stage
        population = new_population

    return best_solution, best_value