
# ALNS.py
# nocaps adaptive large neighborhood search for CVRP with improved fidelity
# (fixed nearest_neighbors_CVRP signature in shaw_removal,
#  reminder: call solver with 'min_veh' as 4th argument, not 'n_nodes')

import random
import copy
import numpy as np
import time
from utils import (
    cost_function,  # computes route cost given coordinates
    distance_function,  # computes euclidean distance: distance_function(x1, y1, x2, y2)
    validate_routes_capacity,  # checks that each route's demand <= capacity
    validate_routes_cities,  # checks that each customer appears exactly once
    Ackley_score,
    generate_initial_solution_ackley,
    nearest_neighbors_CVRP  # signature: nearest_neighbors_CVRP(coordinates, num_neighbors)
)

# default hyperparameters
ALNS_ITERATIONS_DEFAULT = 1000
REMOVE_FRACTION_BOUNDS_DEFAULT = (0.1, 0.3)
REMOVE_FRACTION_MAX = 0.5
REMOVE_FRACTION_ADAPT_STEP = 0.05
EXTRA_TWOOPT_PROBABILITY_DEFAULT = 0.2
WEIGHT_DECAY_DEFAULT = 0.8
WEIGHT_UPDATE_INTERVAL_DEFAULT = 100
KNN_NEIGHBORS_DEFAULT = 50
REGRET_K_DEFAULT = 2
TEMPERATURE_INITIAL = 1.0
TEMPERATURE_EPSILON = 1e-6
STAGNATION_THRESHOLD = 50  # iterations without improvement before adapting removal fraction
VEHICLE_PENALTY = 1e6  # large penalty for exceeding vehicle count


def nearest_neighbor_initial_solution(coordinates, demands, vehicle_capacity, num_vehicles):
    '''constructs an initial solution using greedy nearest-neighbor insertion'''
    total_customers = len(coordinates) - 1
    unvisited = set(range(2, total_customers + 2))
    solution = []
    while unvisited:
        route = [1]  # start at depot (node 1)
        load = 0
        current = 1
        while True:
            nearest = None
            nearest_dist = float('inf')
            for cust in unvisited:
                if load + demands[cust] <= vehicle_capacity:
                    x1, y1 = coordinates[current]
                    x2, y2 = coordinates[cust]
                    dist = distance_function(x1, y1, x2, y2)
                    if dist < nearest_dist:
                        nearest_dist = dist
                        nearest = cust
            if nearest is None:
                break
            route.append(nearest)
            load += demands[nearest]
            unvisited.remove(nearest)
            current = nearest
        route.append(1)  # return to depot
        solution.append(route)
        if len(solution) >= num_vehicles:
            break
    for cust in list(unvisited):
        solution.append([1, cust, 1])
        unvisited.remove(cust)
    return solution


def shaw_removal(routes, coordinates, removal_count, knn_n=KNN_NEIGHBORS_DEFAULT):
    '''removes a set of related customers based on Shaw heuristic (fixed signature)'''
    # collect all customer IDs (excluding depot=1)
    all_customers = [cust for route in routes for cust in route if cust != 1]
    if not all_customers:
        return routes, []
    removed = []
    # pick a random seed customer to start removal
    seed = random.choice(all_customers)
    removed.append(seed)

    # --- FIXED: first get the full k-neighbor dict, then look up seed's neighbors ---
    if knn_n > len(all_customers):
        knn_n = len(all_customers)
    neighbor_dict = nearest_neighbors_CVRP(coordinates, knn_n)
    # neighbor_dict is a dict: {node_id: [closest_k neighbors of node_id], ...}
    # now get seed's actual neighbors:
    knn_list = neighbor_dict.get(seed, [])

    # remove from knn_list until we reach removal_count
    for neighbor in knn_list:
        if len(removed) >= removal_count:
            break
        if neighbor not in removed and neighbor != 1:
            removed.append(neighbor)

    remaining = set(all_customers) - set(removed)
    while len(removed) < removal_count and remaining:
        choice = random.choice(list(remaining))
        removed.append(choice)
        remaining.remove(choice)

    partial = []
    for route in routes:
        new_route = [node for node in route if node not in removed]
        if len(new_route) >= 2:
            if new_route[0] != 1:
                new_route.insert(0, 1)
            if new_route[-1] != 1:
                new_route.append(1)
            partial.append(new_route)
    return partial, removed


def random_removal(routes, removal_count):
    '''removes a random set of customers from the current solution'''
    all_customers = [cust for route in routes for cust in route if cust != 1]
    removal_count = min(removal_count, len(all_customers))
    removed = random.sample(all_customers, removal_count)
    partial = []
    for route in routes:
        new_route = [node for node in route if node not in removed]
        if len(new_route) >= 2:
            if new_route[0] != 1:
                new_route.insert(0, 1)
            if new_route[-1] != 1:
                new_route.append(1)
            partial.append(new_route)
    return partial, removed


def worst_distance_removal(routes, coordinates, removal_count):
    '''iteratively removes customers whose removal gives worst detour improvement'''
    current_routes = copy.deepcopy(routes)
    removed = []
    for _ in range(removal_count):
        best_gain = 0
        best_cust = None
        for route in current_routes:
            for idx in range(1, len(route) - 1):
                cust = route[idx]
                prev_node = route[idx - 1]
                next_node = route[idx + 1]
                x_p, y_p = coordinates[prev_node]
                x_c, y_c = coordinates[cust]
                x_n, y_n = coordinates[next_node]
                cost_with = distance_function(x_p, y_p, x_c, y_c) + distance_function(x_c, y_c, x_n, y_n)
                cost_without = distance_function(x_p, y_p, x_n, y_n)
                gain = cost_with - cost_without
                if gain > best_gain:
                    best_gain = gain
                    best_cust = cust
        if best_cust is None:
            break
        new_routes = []
        for route in current_routes:
            if best_cust in route:
                new_route = [node for node in route if node != best_cust]
                if len(new_route) >= 2:
                    if new_route[0] != 1:
                        new_route.insert(0, 1)
                    if new_route[-1] != 1:
                        new_route.append(1)
                    new_routes.append(new_route)
            else:
                new_routes.append(route)
        current_routes = new_routes
        removed.append(best_cust)
    return current_routes, removed


def greedy_insertion(routes, removed, coordinates, demands, vehicle_capacity):
    '''inserts removed customers using cheapest feasible insertion'''
    partial = copy.deepcopy(routes)
    for cust in removed:
        best_cost = float('inf')
        best_route_idx = None
        best_position = None
        for r_idx, route in enumerate(partial):
            load = sum(demands[node] for node in route if node != 1)
            if load + demands[cust] > vehicle_capacity:
                continue
            for pos in range(1, len(route)):
                prev_node = route[pos - 1]
                next_node = route[pos]
                x_p, y_p = coordinates[prev_node]
                x_c, y_c = coordinates[cust]
                x_n, y_n = coordinates[next_node]
                cost_delta = (
                        distance_function(x_p, y_p, x_c, y_c) +
                        distance_function(x_c, y_c, x_n, y_n) -
                        distance_function(x_p, y_p, x_n, y_n)
                )
                if cost_delta < best_cost:
                    best_cost = cost_delta
                    best_route_idx = r_idx
                    best_position = pos
        if best_route_idx is not None:
            partial[best_route_idx].insert(best_position, cust)
        else:
            partial.append([1, cust, 1])
    return partial


def greedy_insertion_with_penalty(routes, removed, coordinates, demands, vehicle_capacity, num_vehicles):
    '''same as greedy_insertion but penalizes solutions exceeding vehicle count'''
    partial = copy.deepcopy(routes)
    for cust in removed:
        best_cost = float('inf')
        best_route_idx = None
        best_position = None
        for r_idx, route in enumerate(partial):
            load = sum(demands[node] for node in route if node != 1)
            if load + demands[cust] > vehicle_capacity:
                continue
            for pos in range(1, len(route)):
                prev_node = route[pos - 1]
                next_node = route[pos]
                x_p, y_p = coordinates[prev_node]
                x_c, y_c = coordinates[cust]
                x_n, y_n = coordinates[next_node]
                cost_delta = (
                        distance_function(x_p, y_p, x_c, y_c) +
                        distance_function(x_c, y_c, x_n, y_n) -
                        distance_function(x_p, y_p, x_n, y_n)
                )
                if cost_delta < best_cost:
                    best_cost = cost_delta
                    best_route_idx = r_idx
                    best_position = pos
        if best_route_idx is not None:
            partial[best_route_idx].insert(best_position, cust)
        else:
            partial.append([1, cust, 1])
    return partial


def regret_insertion(routes, removed, coordinates, demands, vehicle_capacity, k=REGRET_K_DEFAULT):
    '''inserts removed customers using regret-k heuristic'''
    partial = copy.deepcopy(routes)
    unplaced = removed.copy()
    while unplaced:
        best_cust = None
        best_regret = -float('inf')
        best_insert = None
        for cust in unplaced:
            insertion_costs = []
            for r_idx, route in enumerate(partial):
                load = sum(demands[node] for node in route if node != 1)
                if load + demands[cust] > vehicle_capacity:
                    continue
                for pos in range(1, len(route)):
                    prev_node = route[pos - 1]
                    next_node = route[pos]
                    x_p, y_p = coordinates[prev_node]
                    x_c, y_c = coordinates[cust]
                    x_n, y_n = coordinates[next_node]
                    cost_delta = (
                            distance_function(x_p, y_p, x_c, y_c) +
                            distance_function(x_c, y_c, x_n, y_n) -
                            distance_function(x_p, y_p, x_n, y_n)
                    )
                    insertion_costs.append((cost_delta, r_idx, pos))
            if not insertion_costs:
                regret = VEHICLE_PENALTY
                if regret > best_regret:
                    best_regret = regret
                    best_cust = cust
                    best_insert = None
            else:
                insertion_costs.sort(key=lambda x: x[0])
                if len(insertion_costs) >= k:
                    regret = insertion_costs[k - 1][0] - insertion_costs[0][0]
                else:
                    regret = sum([c[0] for c in insertion_costs]) - len(insertion_costs) * insertion_costs[0][0]
                if regret > best_regret:
                    best_regret = regret
                    best_cust = cust
                    best_insert = insertion_costs[0]
        if best_insert:
            _, r_idx, pos = best_insert
            partial[r_idx].insert(pos, best_cust)
        else:
            partial.append([1, best_cust, 1])
        unplaced.remove(best_cust)
    return partial


def two_opt_route(route, coordinates, demands, vehicle_capacity, max_rounds=3):
    '''performs intra-route 2-opt while ensuring capacity constraints'''
    best_route = route.copy()
    best_cost = cost_function(best_route, coordinates)
    for _ in range(max_rounds):
        improved = False
        n = len(best_route)
        for i in range(1, n - 2):
            for j in range(i + 1, n - 1):
                new_route = best_route[:i] + best_route[i:j + 1][::-1] + best_route[j + 1:]
                load = sum(demands[node] for node in new_route if node != 1)
                if load > vehicle_capacity:
                    continue
                new_cost = cost_function(new_route, coordinates)
                if new_cost < best_cost:
                    best_route = new_route
                    best_cost = new_cost
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return best_route


def relocate_move(solution, coordinates, demands, vehicle_capacity):
    '''attempts to relocate a single customer to another route if cost improves'''
    best_solution = copy.deepcopy(solution)
    current_cost = sum(cost_function(route, coordinates) for route in best_solution) \
                   + VEHICLE_PENALTY * max(0, len(best_solution) - num_vehicles_global)
    for r_from in range(len(solution)):
        for idx in range(1, len(solution[r_from]) - 1):
            cust = solution[r_from][idx]
            for r_to in range(len(solution)):
                if r_to == r_from:
                    continue
                load_to = sum(demands[node] for node in solution[r_to] if node != 1)
                if load_to + demands[cust] > vehicle_capacity:
                    continue
                for pos in range(1, len(solution[r_to])):
                    new_sol = copy.deepcopy(solution)
                    new_sol[r_from].remove(cust)
                    new_sol[r_to].insert(pos, cust)
                    if len(new_sol[r_from]) < 3:
                        new_sol.pop(r_from)
                    total_cost = sum(cost_function(route, coordinates) for route in new_sol)
                    total_cost += VEHICLE_PENALTY * max(0, len(new_sol) - num_vehicles_global)
                    if total_cost < current_cost:
                        best_solution = new_sol
                        current_cost = total_cost
    return best_solution


def swap_move(solution, coordinates, demands, vehicle_capacity):
    '''attempts to swap two customers from different routes if cost improves'''
    best_solution = copy.deepcopy(solution)
    current_cost = sum(cost_function(route, coordinates) for route in best_solution) \
                   + VEHICLE_PENALTY * max(0, len(best_solution) - num_vehicles_global)
    for r1 in range(len(solution)):
        for r2 in range(r1 + 1, len(solution)):
            for idx1 in range(1, len(solution[r1]) - 1):
                for idx2 in range(1, len(solution[r2]) - 1):
                    cust1 = solution[r1][idx1]
                    cust2 = solution[r2][idx2]
                    load_r1 = sum(demands[node] for node in solution[r1] if node != 1) - demands[cust1] + demands[cust2]
                    load_r2 = sum(demands[node] for node in solution[r2] if node != 1) - demands[cust2] + demands[cust1]
                    if load_r1 > vehicle_capacity or load_r2 > vehicle_capacity:
                        continue
                    new_sol = copy.deepcopy(solution)
                    new_sol[r1][idx1], new_sol[r2][idx2] = cust2, cust1
                    total_cost = sum(cost_function(route, coordinates) for route in new_sol)
                    total_cost += VEHICLE_PENALTY * max(0, len(new_sol) - num_vehicles_global)
                    if total_cost < current_cost:
                        best_solution = new_sol
                        current_cost = total_cost
    return best_solution


def accept_solution(current_cost, candidate_cost, temperature):
    '''simulated annealing acceptance criterion'''
    if candidate_cost < current_cost:
        return True
    diff = candidate_cost - current_cost
    prob = np.exp(-diff / max(temperature, TEMPERATURE_EPSILON))
    return random.random() < prob


def alns_metaheuristic_solver(
        coordinates,
        demands,
        vehicle_capacity,
        num_vehicles,  # <<<<<< must pass 'min_vehicles' here, not 'n_nodes'
        iterations=ALNS_ITERATIONS_DEFAULT,
        remove_bounds=REMOVE_FRACTION_BOUNDS_DEFAULT,
        twoopt_prob=EXTRA_TWOOPT_PROBABILITY_DEFAULT,
        knn_n=KNN_NEIGHBORS_DEFAULT,
        random_seed=None,
        time_limit=None  # optional time limit in seconds
):
    '''main improved ALNS solver with decoupled operator pools and constraint enforcement

    :returns: (solution, cost, metrics) where metrics track best and current
              cost per iteration, acceptance rate and time per iteration
    '''
    start_time = time.time()

    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)

    # BUILD INITIAL SOLUTION—nearest neighbor or MSH.
    current_solution = nearest_neighbor_initial_solution(coordinates, demands, vehicle_capacity, num_vehicles)
    # (If you'd rather start from MSH, comment out the line above and uncomment below)
    # current_solution, _ = MSH_alg(coordinates, demands, vehicle_capacity, num_vehicles)

    # ensure coverage (no duplicates or misses). Capacity may exceed temporarily—penalty applies.
    assert validate_routes_cities(current_solution, coordinates, demands, vehicle_capacity), \
        'initial solution missing or duplicating cities'

    def total_cost_with_penalty(solution):
        cost = sum(cost_function(route, coordinates) for route in solution)
        extra = max(0, len(solution) - num_vehicles)
        return cost + VEHICLE_PENALTY * extra

    current_cost = total_cost_with_penalty(current_solution)
    best_solution = copy.deepcopy(current_solution)
    best_cost = current_cost

    # metrics init
    best_cost_per_iter = []
    current_cost_per_iter = []
    accept_rate_per_iter = []
    time_per_iter = []
    temperature_history = []
    max_frac_history = []
    destroy_weights_history = []
    repair_weights_history = []
    destroy_prob_history = []
    repair_prob_history = []

    # operator pools (destroy and repair separated)
    destroy_ops = ['random_removal', 'worst_distance_removal', 'shaw_removal']
    repair_ops = ['greedy_insertion', 'regret_insertion', 'greedy_insertion_with_penalty']

    destroy_weights = [1.0] * len(destroy_ops)
    destroy_scores = [0.0] * len(destroy_ops)
    destroy_usage = [0] * len(destroy_ops)

    repair_weights = [1.0] * len(repair_ops)
    repair_scores = [0.0] * len(repair_ops)
    repair_usage = [0] * len(repair_ops)

    min_frac, max_frac = remove_bounds
    last_improve_iter = 0
    total_customers = len(coordinates) - 1

    global num_vehicles_global
    num_vehicles_global = num_vehicles
    t0 = time.time()
    print(
        f'\nStarting ALNS with {iterations} iterations, with exploration of {KNN_NEIGHBORS_DEFAULT} neighbors per iteration.\n')
    for iteration in range(iterations):
        t_start = time.time()

        # adapt removal fraction if stagnated
        if iteration - last_improve_iter > STAGNATION_THRESHOLD:
            max_frac = min(REMOVE_FRACTION_MAX, max_frac + REMOVE_FRACTION_ADAPT_STEP)
        else:
            max_frac = max(remove_bounds[1], max_frac - REMOVE_FRACTION_ADAPT_STEP / 2)

        max_frac_history.append(max_frac)

        # SELECT DESTROY operator
        total_d_weight = sum(destroy_weights)
        d_probs = [w / total_d_weight for w in destroy_weights]
        # record destroy ops pools
        destroy_weights_history.append(destroy_weights.copy())
        destroy_prob_history.append(d_probs.copy())
        d_idx = random.choices(range(len(destroy_ops)), weights=d_probs)[0]
        destroy_usage[d_idx] += 1

        # SELECT REPAIR operator
        total_r_weight = sum(repair_weights)
        r_probs = [w / total_r_weight for w in repair_weights]
        # record repair ops pools
        repair_weights_history.append(repair_weights.copy())
        repair_prob_history.append(r_probs.copy())
        r_idx = random.choices(range(len(repair_ops)), weights=r_probs)[0]
        repair_usage[r_idx] += 1

        # DETERMINE how many customers to remove
        min_remove = max(1, int(min_frac * total_customers))
        max_remove = max(min_remove, int(max_frac * total_customers))
        removal_count = random.randint(min_remove, max_remove)

        # DESTRUCTION
        if destroy_ops[d_idx] == 'random_removal':
            partial_routes, removed = random_removal(current_solution, removal_count)
        elif destroy_ops[d_idx] == 'worst_distance_removal':
            partial_routes, removed = worst_distance_removal(current_solution, coordinates, removal_count)
        else:  # shaw_removal
            partial_routes, removed = shaw_removal(current_solution, coordinates, removal_count, knn_n)

        # REPAIR
        if repair_ops[r_idx] == 'greedy_insertion':
            candidate = greedy_insertion(partial_routes, removed, coordinates, demands, vehicle_capacity)
        elif repair_ops[r_idx] == 'regret_insertion':
            candidate = regret_insertion(partial_routes, removed, coordinates, demands, vehicle_capacity,
                                         k=REGRET_K_DEFAULT)
        else:  # greedy_insertion_with_penalty
            candidate = greedy_insertion_with_penalty(partial_routes, removed, coordinates, demands, vehicle_capacity,
                                                      num_vehicles)

        # LOCAL SEARCH: 2-opt on each route
        for r_i in range(len(candidate)):
            if random.random() < twoopt_prob:
                candidate[r_i] = two_opt_route(candidate[r_i], coordinates, demands, vehicle_capacity)
        # occasionally try relocate/swap
        if random.random() < 0.1:
            candidate = relocate_move(candidate, coordinates, demands, vehicle_capacity)
        if random.random() < 0.1:
            candidate = swap_move(candidate, coordinates, demands, vehicle_capacity)

        # COMPUTE cost + fleet penalty
        candidate_cost = total_cost_with_penalty(candidate)

        # SA ACCEPTANCE
        temperature = TEMPERATURE_INITIAL * (1 - iteration / iterations)
        # record temp
        temperature_history.append(temperature)
        accepted = accept_solution(current_cost, candidate_cost, temperature)
        if accepted:
            current_solution = candidate
            current_cost = candidate_cost
            if candidate_cost < best_cost:
                best_solution = copy.deepcopy(candidate)
                best_cost = candidate_cost
                last_improve_iter = iteration
                destroy_scores[d_idx] += 2.0
                repair_scores[r_idx] += 2.0
            else:
                destroy_scores[d_idx] += 1.0
                repair_scores[r_idx] += 1.0
        else:
            destroy_scores[d_idx] += 0.5
            repair_scores[r_idx] += 0.5

        # record metrics for this iteration
        best_cost_per_iter.append(best_cost)
        current_cost_per_iter.append(current_cost)
        accept_rate_per_iter.append(1 if accepted else 0)
        time_per_iter.append(time.time() - t_start)

        # PERIODIC WEIGHT UPDATE
        if (iteration + 1) % WEIGHT_UPDATE_INTERVAL_DEFAULT == 0:
            # destroy weights update
            for i in range(len(destroy_weights)):
                if destroy_usage[i] > 0:
                    perf = destroy_scores[i] / destroy_usage[i]
                    destroy_weights[i] = (
                            WEIGHT_DECAY_DEFAULT * destroy_weights[i]
                            + (1 - WEIGHT_DECAY_DEFAULT) * perf
                    )
                destroy_scores[i] = 0.0
                destroy_usage[i] = 0
            # repair weights update
            for i in range(len(repair_weights)):
                if repair_usage[i] > 0:
                    perf = repair_scores[i] / repair_usage[i]
                    repair_weights[i] = (
                            WEIGHT_DECAY_DEFAULT * repair_weights[i]
                            + (1 - WEIGHT_DECAY_DEFAULT) * perf
                    )
                repair_scores[i] = 0.0
                repair_usage[i] = 0
        total_elapsed = time.time() - t0
        print(
            f'[ALNS] iteration = {iteration + 1}/{iterations} | best = {best_cost:.2f} | accepted = {accepted} | time = {total_elapsed:.2f}s')
        time_check = time.time() - start_time
        if time_limit is not None and time_check >= time_limit:
            print(f'[ALNS] Time limit reached: {time_limit} seconds.')
            break
    # FINAL FEASIBILITY CHECK
    assert validate_routes_cities(best_solution, coordinates, demands, vehicle_capacity), \
        'best solution missing or duplicating cities'
    assert validate_routes_capacity(best_solution, coordinates, demands, vehicle_capacity), \
        'best solution violates capacity'

    metrics = {
        'best_cost_per_iter': best_cost_per_iter,
        'current_cost_per_iter': current_cost_per_iter,
        'accept_rate_per_iter': accept_rate_per_iter,
        'computation_time_per_step': time_per_iter
    }
    # ---------------------------------------------------------------------
    # plot all dynamic parameters before exiting
    # ---------------------------------------------------------------------
    import matplotlib.pyplot as plt
    actual_iters = len(best_cost_per_iter)
    iters = list(range(1, actual_iters + 1))
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    # 1) cost metrics
    axes[0].plot(iters, best_cost_per_iter, label='best cost')
    axes[0].plot(iters, current_cost_per_iter, label='current cost')
    axes[0].set_ylabel('cost')
    axes[0].set_title('ALNS Cost Metrics')
    axes[0].legend(loc='best')

    # 2) acceptance rate & timing
    axes[1].plot(iters, accept_rate_per_iter, label='accept rate')
    axes[1].plot(iters, time_per_iter, label='time per iter (s)')
    axes[1].set_ylabel('rate / seconds')
    axes[1].set_title('Acceptance Rate & Timing')
    axes[1].legend(loc='best')

    # 3) temperature & removal fraction
    axes[2].plot(iters, temperature_history, label='temperature')
    axes[2].plot(iters, max_frac_history, label='max removal frac')
    axes[2].set_ylabel('value')
    axes[2].set_title('Temperature Schedule & Removal Fraction')
    axes[2].legend(loc='best')
    plt.tight_layout()
    plt.show()

    return best_solution, best_cost, metrics


import random
import copy
import numpy as np

from utils import (
    cost_function,
    distance_function,
    validate_routes_capacity,
    validate_routes_cities,
    nearest_neighbors_CVRP,
    Ackley_score,
    generate_initial_solution_ackley
)

# ─── (all existing CVRP‐specific ALNS code here) ─────────────────────────────

import matplotlib.pyplot as plt
# ─── Updated Ackley-mode ALNS (ruin & rebuild) ────────────────────────────────
# def alns_ackley_solver(
#     dim=10,
#     lower=-32.768,
#     upper=32.768,
#     iterations=ALNS_ITERATIONS_DEFAULT,
#     remove_bounds=REMOVE_FRACTION_BOUNDS_DEFAULT,
#     k=KNN_NEIGHBORS_DEFAULT,
#     seed=None
# ):
#     """ALNS for the d-dimensional Ackley function
#
#     :returns: (vector, value, metrics) tracking best/current value, acceptance
#               rate and time per iteration
#     """
#
#     # Optional seeding for reproducibility
#     if seed is not None:
#         random.seed(seed)
#         np.random.seed(seed)
#
#     # 1) Initialize current solution as a Python list
#     #    (convert the NumPy array to a list immediately)
#     current = generate_initial_solution_ackley(None, dim=dim, lower_bound=lower, upper_bound=upper).tolist()
#     best = current.copy()
#     best_val = Ackley_score(np.array(best))
#
#     best_cost_per_iter = []
#     current_cost_per_iter = []
#     accept_rate_per_iter = []
#     time_per_iter = []
#
#     for it in range(iterations):
#         t_start = time.time()
#         # Decide how many coords to “remove”
#         frac = random.uniform(*remove_bounds)
#         num_remove = max(1, int(frac * dim))
#         idxs = random.sample(range(dim), num_remove)
#
#         # Make a “partial” copy as a Python list, set removed coords to None
#         partial = current.copy()
#         old_vals = {i: partial[i] for i in idxs}
#         for i in idxs:
#             partial[i] = None
#
#         # ── REPAIR STEP: for each removed index i, choose best among k random values ──
#         for i in idxs:
#             local_best_val = float("inf")
#             local_best_coord = None
#
#             for _ in range(k):
#                 candidate_val = random.uniform(lower, upper)
#                 trial = partial.copy()
#                 trial[i] = candidate_val
#                 # fill in any remaining None’s with their old values
#                 for j in idxs:
#                     if trial[j] is None:
#                         trial[j] = old_vals[j]
#                 v = Ackley_score(np.array(trial))
#                 if v < local_best_val:
#                     local_best_val = v
#                     local_best_coord = candidate_val
#
#             partial[i] = local_best_coord
#
#         # Evaluate the fully repaired solution
#         val = Ackley_score(np.array(partial))
#         if val < best_val:
#             best, best_val = partial.copy(), val
#             acc = 1
#         else:
#             acc = 0
#
#         current = partial.copy()
#         best_cost_per_iter.append(best_val)
#         current_cost_per_iter.append(val)
#         accept_rate_per_iter.append(acc)
#         time_per_iter.append(time.time() - t_start)
#
#     metrics = {
#         'best_cost_per_iter': best_cost_per_iter,
#         'current_cost_per_iter': current_cost_per_iter,
#         'accept_rate_per_iter': accept_rate_per_iter,
#         'computation_time_per_step': time_per_iter
#     }
#
#     return best, best_val, metrics

import random
import copy
import numpy as np
import matplotlib.pyplot as plt

from utils import (
    Ackley_score, generate_initial_solution_ackley
)


# ALNS for Ackley, tracking removal dynamics and acceptance rate in one plot

def alns_ackley_solver(
        dim=10,
        lower=-32.768,
        upper=32.768,
        iterations=ALNS_ITERATIONS_DEFAULT,
        remove_bounds=REMOVE_FRACTION_BOUNDS_DEFAULT,
        k=KNN_NEIGHBORS_DEFAULT,
        seed=None
):
    """ALNS for the d-dimensional Ackley function

    :returns: (vector, value, metrics) with histories for
              removal frac, num_remove, and acceptance flag
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # initialize solution
    current = generate_initial_solution_ackley(
        None, dim=dim, lower_bound=lower, upper_bound=upper
    ).tolist()
    best = current.copy()
    best_val = Ackley_score(np.array(best))

    # history lists
    frac_history = []
    num_remove_history = []
    accept_rate_per_iter = []

    # main loop
    for it in range(iterations):
        # ruin: remove a fraction of coords
        frac = random.uniform(*remove_bounds)
        num_remove = max(1, int(frac * dim))
        frac_history.append(frac)
        num_remove_history.append(num_remove)

        # create partial solution
        idxs = random.sample(range(dim), num_remove)
        partial = current.copy()
        old_vals = {i: partial[i] for i in idxs}
        for i in idxs:
            partial[i] = None

        # repair: try k candidates
        for i in idxs:
            best_loc_val = float('inf')
            best_loc_coord = None
            for _ in range(k):
                cand = random.uniform(lower, upper)
                trial = partial.copy()
                trial[i] = cand
                for j in idxs:
                    if trial[j] is None:
                        trial[j] = old_vals[j]
                v = Ackley_score(np.array(trial))
                if v < best_loc_val:
                    best_loc_val = v
                    best_loc_coord = cand
            partial[i] = best_loc_coord

        # evaluate & accept
        val = Ackley_score(np.array(partial))
        accepted = 1 if val < best_val else 0
        if accepted:
            best = partial.copy()
            best_val = val
        current = partial.copy()
        accept_rate_per_iter.append(accepted)

    # prepare metrics
    metrics = {
        'frac_history': frac_history,
        'num_remove_history': num_remove_history,
        'accept_rate_per_iter': accept_rate_per_iter
    }

    # plot all dynamics in one figure
    gens = list(range(1, iterations + 1))
    plt.figure(figsize=(12, 8))
    plt.plot(gens, frac_history, label='removal fraction')
    plt.plot(gens, num_remove_history, label='num removed')
    plt.plot(gens, accept_rate_per_iter, label='acceptance flag')
    plt.xlabel('iteration')
    plt.ylabel('value')
    plt.title('ALNS Ackley dynamic parameters over iterations')
    plt.legend(loc='upper right', fontsize='small')
    plt.show()

    return best, best_val, metrics
