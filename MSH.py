"""
Two-Phase Multi-Knapsack Problem (MKP) based heuristic for CVRP.
Phase 1: Cluster customers using MKP to assign customers to vehicles
Phase 2: Route optimization using TSP within each cluster
"""

#-------------------Imports-------------------------------
from utils import *
import numpy as np
import os
from pulp import *
import math
import time


#--------------------Functions-------------------------------

def calculate_customer_values(coordinates,demands, capacity,  depot_idx=1):
    """
    Calculate the value (priority) of each customer based on:
    1. Distance from depot
    2. Average distance to nearest neighbors

    Lower values are better (higher priority) as we want to minimize distance
    """
    values = {}
    depot_coords = coordinates[depot_idx]

    # Pre-calculate all distances for efficiency
    distances = {}
    for i in coordinates:
        for j in coordinates:
            if i != j:
                distances[(i, j)] = distance_function(
                    coordinates[i][0], coordinates[i][1],
                    coordinates[j][0], coordinates[j][1]
                )

    for customer_id in coordinates:
        if customer_id == depot_idx:
            continue

        # Distance from depot
        dist_from_depot = distances[(depot_idx, customer_id)]

        # Calculate potential savings for this customer
        savings = {}
        for other_id in coordinates:
            if other_id != customer_id and other_id != depot_idx:
                # Clarke-Wright savings: d(0,i) + d(0,j) - d(i,j)
                savings[other_id] = distances[(depot_idx, customer_id)] + \
                                    distances[(depot_idx, other_id)] - \
                                    distances[(customer_id, other_id)]

        # Top 5 savings neighbors
        top_savings = sorted(savings.items(), key=lambda x: x[1], reverse=True)[:5]
        avg_savings = sum(s for _, s in top_savings) / len(top_savings) if top_savings else 0

        # Consider demand as well - prefer customers with higher demand relative to capacity
        demand_ratio = demands[customer_id] / capacity

        # Final value incorporates both distance and demand considerations
        values[customer_id] = (0.5 * avg_savings + 0.3 * (1/dist_from_depot) + 0.2 * demand_ratio) * 100

    return values

def solve_mkp(coordinates, demands, capacity, num_vehicles):
    """
    Enhanced MKP solver with spatial clustering initialization and multi-neighborhood search.
    Uses existing utils functions for neighborhood operations.
    """
    customers = [i for i in coordinates.keys() if i != 1]
    depot = coordinates[1]  # Depot coordinates

    # Step 1: Better initialization - spatial seeds + greedy assignment
    # Initialize clusters with spatially distributed seeds
    clusters = {j: [1] for j in range(num_vehicles)}
    loads = [0] * num_vehicles

    # Create initial seeds spread around the depot
    for j in range(num_vehicles):
        angle = 2 * math.pi * j / num_vehicles
        radius = 50  # Search radius from depot
        center_x = depot[0] + radius * math.cos(angle)
        center_y = depot[1] + radius * math.sin(angle)

        # Find closest customer to this seed point
        if customers:
            seed_customer = min(customers, key=lambda c:
                                (coordinates[c][0] - center_x)**2 +
                                (coordinates[c][1] - center_y)**2)

            clusters[j].append(seed_customer)
            loads[j] += demands[seed_customer]
            customers.remove(seed_customer)

    # Greedy assignment of remaining customers by demand and proximity
    customers_sorted = sorted(customers, key=lambda i: -demands[i])
    for customer in customers_sorted:
        # Calculate cost-to-insert for each vehicle
        best_j = None
        best_score = float('inf')

        for j in range(num_vehicles):
            # Skip if capacity would be exceeded
            if loads[j] + demands[customer] > capacity:
                continue

            # Calculate average distance to existing customers in cluster
            if len(clusters[j]) > 1:  # If cluster already has customers beyond depot
                avg_dist = sum(distance_function(
                    coordinates[customer][0], coordinates[customer][1],
                    coordinates[c][0], coordinates[c][1]
                ) for c in clusters[j][1:]) / (len(clusters[j]) - 1)
            else:
                # Distance to depot only
                avg_dist = distance_function(
                    coordinates[customer][0], coordinates[customer][1],
                    depot[0], depot[1]
                )

            # Score based on distance and remaining capacity (lower is better)
            remaining_capacity = capacity - loads[j]
            score = 0.7 * avg_dist - 0.3 * remaining_capacity

            if score < best_score:
                best_score = score
                best_j = j

        # Assign customer to best vehicle or least loaded if none found
        if best_j is not None:
            clusters[best_j].append(customer)
            loads[best_j] += demands[customer]
        else:
            # If no vehicle can take this customer, assign to the least loaded
            best_j = min(range(num_vehicles), key=lambda j: loads[j])
            clusters[best_j].append(customer)
            loads[best_j] += demands[customer]

    # Step 2: Advanced multi-neighborhood search
    max_attempts = 200
    no_improvement_limit = 50
    attempts = 0
    no_improvement = 0

    # Calculate initial solution quality
    quality = sum(sum(distance_function(
        coordinates[c1][0], coordinates[c1][1],
        coordinates[c2][0], coordinates[c2][1]
    ) for c1 in clusters[j] for c2 in clusters[j] if c1 != c2) for j in range(num_vehicles))


    while attempts < max_attempts and no_improvement < no_improvement_limit:
        attempts += 1
        old_quality = quality
        improved = False

        # 1. Try relocate moves
        for i in range(num_vehicles):
            for j in range(num_vehicles):
                if i == j:
                    continue

                # Try relocating each customer from cluster i to j
                for customer in clusters[i][1:]:  # Skip depot
                    # Check if capacity allows the move
                    if loads[j] + demands[customer] <= capacity:
                        # Perform relocate move
                        clusters[i].remove(customer)
                        clusters[j].append(customer)
                        loads[i] -= demands[customer]
                        loads[j] += demands[customer]

                        # Calculate new solution quality
                        new_quality = sum(sum(distance_function(
                            coordinates[c1][0], coordinates[c1][1],
                            coordinates[c2][0], coordinates[c2][1]
                        ) for c1 in clusters[k] for c2 in clusters[k] if c1 != c2) for k in range(num_vehicles))

                        if new_quality < quality:
                            # Accept the improvement
                            quality = new_quality
                            improved = True
                            break
                        else:
                            # Revert the move
                            clusters[j].remove(customer)
                            clusters[i].append(customer)
                            loads[j] -= demands[customer]
                            loads[i] += demands[customer]

                if improved:
                    break
            if improved:
                break

        # 2. Try swap moves if no relocate move improved the solution
        if not improved:
            for i in range(num_vehicles):
                for j in range(i+1, num_vehicles):
                    for c1 in clusters[i][1:]:  # Skip depot
                        for c2 in clusters[j][1:]:  # Skip depot
                            # Check if swap maintains capacity constraints
                            if (loads[i] - demands[c1] + demands[c2] <= capacity and
                                loads[j] - demands[c2] + demands[c1] <= capacity):

                                # Perform swap
                                clusters[i].remove(c1)
                                clusters[j].remove(c2)
                                clusters[i].append(c2)
                                clusters[j].append(c1)
                                loads[i] = loads[i] - demands[c1] + demands[c2]
                                loads[j] = loads[j] - demands[c2] + demands[c1]

                                # Calculate new solution quality
                                new_quality = sum(sum(distance_function(
                                    coordinates[c1][0], coordinates[c1][1],
                                    coordinates[c2][0], coordinates[c2][1]
                                ) for c1 in clusters[k] for c2 in clusters[k] if c1 != c2) for k in range(num_vehicles))

                                if new_quality < quality:
                                    # Accept the improvement
                                    quality = new_quality
                                    improved = True
                                    break
                                else:
                                    # Revert the swap
                                    clusters[i].remove(c2)
                                    clusters[j].remove(c1)
                                    clusters[i].append(c1)
                                    clusters[j].append(c2)
                                    loads[i] = loads[i] - demands[c2] + demands[c1]
                                    loads[j] = loads[j] - demands[c1] + demands[c2]

                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break

        # Update improvement counter
        if quality < old_quality:
            no_improvement = 0

        else:
            no_improvement += 1



    # Filter out empty clusters
    return {j: cluster for j, cluster in clusters.items() if len(cluster) > 1}

"""def solve_mkp(coordinates, demands, capacity, num_vehicles):
    
    #Solve the Multi-Knapsack Problem to assign customers to vehicles.
    #Each vehicle is treated as a knapsack with capacity constraints.
    #The objective is to maximize the value (priority) of assigned customers.
    
    customer_values = calculate_customer_values(coordinates, demands, capacity)
    # Create the MKP model
    model = LpProblem("CVRP_MKP", LpMaximize)
    
    # Decision variables
    customers = [i for i in coordinates.keys() if i != 1]
    vehicles = range(num_vehicles)
    
    x = {}
    for i in customers:
        for j in vehicles:
            x[i, j] = LpVariable(f"x_{i}_{j}", cat=LpBinary)
    
    # Objective function with geographic compactness
    model += lpSum(customer_values[i] * x[i, j] for i in customers for j in vehicles)
    
    # Customer assignment constraint
    for i in customers:
        model += lpSum(x[i, j] for j in vehicles) == 1
    
    # Capacity constraint
    for j in vehicles:
        model += lpSum(demands[i] * x[i, j] for i in customers) <= capacity
    
    # Add geographic proximity incentives
    depot = coordinates[1]
    vehicle_centers = {}
    for j in vehicles:
        angle = 2 * math.pi * j / num_vehicles
        radius = 50  # Adjust based on your coordinate scale
        vehicle_centers[j] = (
            depot[0] + radius * math.cos(angle),
            depot[1] + radius * math.sin(angle)
        )
    
    # Add soft constraints to encourage customers to be assigned to geographically proximate vehicles
    proximity_vars = {}
    for i in customers:
        for j in vehicles:
            # Calculate distance from customer to vehicle's ideal center
            dist = math.sqrt((coordinates[i][0] - vehicle_centers[j][0])**2 + 
                            (coordinates[i][1] - vehicle_centers[j][1])**2)
            # Normalized proximity score (closer is better)
            proximity_vars[(i, j)] = LpVariable(f"prox_{i}_{j}", lowBound=0, upBound=1, cat=LpContinuous)
            # Link proximity variables to assignment variables
            model += proximity_vars[(i, j)] <= x[i, j]
            model += proximity_vars[(i, j)] <= 1 - dist/200  # Normalize based on your distance scale
    
    # Add proximity to objective with small weight
    model += 0.2 * lpSum(proximity_vars.values())
    # Solve using bundled CBC
    if getattr(sys, 'frozen', False):
    # PyInstaller EXE: use sys._MEIPASS
        cbc_path = os.path.join(sys._MEIPASS, "solvers", "cbc.exe")
        cbc_path
    else:
        # Script mode: use relative to this file
        cbc_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "solvers", "cbc.exe")
        
    if not os.path.isfile(cbc_path):
        raise FileNotFoundError(f"CBC solver not found at: {cbc_path}")
    model.solve(PULP_CBC_CMD(path=cbc_path, msg=False))
    print("--------------------------------------------------------------------")
    # Extract clusters
    clusters = {j: [1] for j in vehicles}
    for i in customers:
        for j in vehicles:
            if x[i, j].value() > 0.5:
                clusters[j].append(i)
    
    # Filter out empty clusters
    return {j: cluster for j, cluster in clusters.items() if len(cluster) > 1}
"""
def two_opt(route, coords):
    """Apply 2-opt optimization to improve a TSP route."""
    best = route.copy()
    improved = True
    while improved:
        improved = False
        for i in range(1, len(best) - 2):
            for j in range(i + 1, len(best)):
                if j - i == 1:
                    continue  # Skip adjacent nodes
                new_route = best[:i] + best[i:j][::-1] + best[j:]
                if cost_function(new_route, coords) < cost_function(best, coords):
                    best = new_route
                    improved = True
    return best


def insertion_heuristic_tsp(cluster, coordinates, depot_idx=1):
    """
    Solve TSP using Cheapest Insertion heuristic for a given cluster.

    This method starts with a subtour containing only the depot, then iteratively
    inserts the remaining nodes at the position that minimizes the insertion cost.

    Parameters:
    - cluster: List of node indices in the cluster
    - coordinates: Dictionary mapping node indices to (x, y) coordinates
    - depot_idx: Index of the depot node (default: 1)

    Returns:
    - route: Optimized TSP route starting and ending at the depot
    """
    # Start with a subtour containing only the depot
    route = [depot_idx, depot_idx]

    # Get unvisited nodes
    unvisited = [node for node in cluster if node != depot_idx]

    # Continue until all nodes are visited
    while unvisited:
        # Find the best (node, position) combination
        best_cost_increase = float('inf')
        best_node = None
        best_pos = None

        for node in unvisited:
            # Try inserting the node at each position in the current route
            for pos in range(1, len(route)):
                # Calculate the cost increase
                prev_node = route[pos-1]
                next_node = route[pos]

                # Current cost between prev_node and next_node
                current_cost = distance_function(
                    coordinates[prev_node][0], coordinates[prev_node][1],
                    coordinates[next_node][0], coordinates[next_node][1]
                )

                # New cost with node inserted between prev_node and next_node
                new_cost = distance_function(
                    coordinates[prev_node][0], coordinates[prev_node][1],
                    coordinates[node][0], coordinates[node][1]
                ) + distance_function(
                    coordinates[node][0], coordinates[node][1],
                    coordinates[next_node][0], coordinates[next_node][1]
                )

                # Cost increase
                cost_increase = new_cost - current_cost

                # Update best insertion if this is better
                if cost_increase < best_cost_increase:
                    best_cost_increase = cost_increase
                    best_node = node
                    best_pos = pos

        # Insert the best node at the best position
        route.insert(best_pos, best_node)
        unvisited.remove(best_node)

    return route

def nearest_neighbor_tsp(cluster, coordinates, depot_idx=1):
    """
    Solve TSP using Nearest Neighbor heuristic for a given cluster.
    """
    route = [depot_idx]  # Start from depot
    unvisited = cluster.copy()
    unvisited.remove(depot_idx)

    while unvisited:
        current = route[-1]
        nearest = min(unvisited, key=lambda x: distance_function(
            coordinates[current][0], coordinates[current][1],
            coordinates[x][0], coordinates[x][1]
        ))
        route.append(nearest)
        unvisited.remove(nearest)

    route.append(depot_idx)  # Return to depot
    return route

def christofides_approximation(cluster, coordinates, depot_idx=1):
    """
    Approximate TSP using a simplified version of Christofides algorithm.

    This is a simplified implementation that:
    1. Builds a minimum spanning tree (MST)
    2. Identifies odd-degree vertices in the MST
    3. Adds minimum-weight matching edges to make all vertices even degree
    4. Constructs an Eulerian circuit
    5. Converts to a Hamiltonian circuit by shortcutting

    Parameters:
    - cluster: List of node indices in the cluster
    - coordinates: Dictionary mapping node indices to (x, y) coordinates
    - depot_idx: Index of the depot node (default: 1)

    Returns:
    - route: Approximate TSP route starting and ending at the depot
    """
    # Calculate distance matrix
    n = len(cluster)
    dist_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i != j:
                dist_matrix[i][j] = distance_function(
                    coordinates[cluster[i]][0], coordinates[cluster[i]][1],
                    coordinates[cluster[j]][0], coordinates[cluster[j]][1]
                )

    # Step 1: Build MST using Prim's algorithm
    mst_edges = []
    visited = [0]  # Start with the first node (which should be depot)

    while len(visited) < n:
        min_edge = None
        min_cost = float('inf')

        for i in visited:
            for j in range(n):
                if j not in visited and dist_matrix[i][j] < min_cost:
                    min_cost = dist_matrix[i][j]
                    min_edge = (i, j)

        mst_edges.append(min_edge)
        visited.append(min_edge[1])

    # Step 2: Find vertices with odd degree in the MST
    degrees = [0] * n
    for i, j in mst_edges:
        degrees[i] += 1
        degrees[j] += 1

    odd_vertices = [i for i in range(n) if degrees[i] % 2 == 1]

    # Step 3: Find minimum-weight perfect matching for odd vertices
    # (Simplified approach - just match odd vertices in a greedy manner)
    matching_edges = []
    while odd_vertices:
        i = odd_vertices.pop(0)
        min_cost = float('inf')
        best_j = None

        for j in odd_vertices:
            if dist_matrix[i][j] < min_cost:
                min_cost = dist_matrix[i][j]
                best_j = j

        if best_j is not None:
            matching_edges.append((i, best_j))
            odd_vertices.remove(best_j)

    # Step 4: Combine MST and matching edges to form a multigraph
    all_edges = mst_edges + matching_edges

    # Step 5: Form an Eulerian circuit (simplified approach)
    # Create adjacency list
    adj_list = [[] for _ in range(n)]
    for i, j in all_edges:
        adj_list[i].append(j)
        adj_list[j].append(i)

    # Start with depot node (index 0)
    eulerian_path = []

    def find_eulerian_path(node):
        while adj_list[node]:
            next_node = adj_list[node][0]
            # Remove the edge
            adj_list[node].remove(next_node)
            adj_list[next_node].remove(node)
            # Recurse
            find_eulerian_path(next_node)
        eulerian_path.append(node)

    find_eulerian_path(0)
    eulerian_path.reverse()  # Correct order

    # Step 6: Shortcut the Eulerian path to get a Hamiltonian circuit
    # Keep track of visited nodes
    visited = set()
    hamiltonian_path = []

    for node in eulerian_path:
        if node not in visited:
            hamiltonian_path.append(node)
            visited.add(node)

    # Close the circuit
    hamiltonian_path.append(0)

    # Convert indices back to actual node IDs
    route = [cluster[i] for i in hamiltonian_path]

    return route

def solve_tsp_for_clusters(clusters, coordinates):
    """
    Solve TSP for each cluster to find optimal routes.
    Uses nearest neighbor heuristic followed by 2-opt optimization.
    """
    routes = []

    for cluster_id, cluster in clusters.items():
        if len(cluster) <= 3:  # For very small clusters, simple approach is sufficient
            route = cluster + [cluster[0]]  # Complete the cycle
            routes.append(route)
            continue

        # Try multiple TSP approaches and select the best
        candidate_routes = []

        # 1. Nearest neighbor
        nn_route = nearest_neighbor_tsp(cluster, coordinates)
        nn_route = two_opt(nn_route, coordinates)
        candidate_routes.append(nn_route)

        # 2. Insertion heuristic
        insert_route = insertion_heuristic_tsp(cluster, coordinates)
        insert_route = two_opt(insert_route, coordinates)
        candidate_routes.append(insert_route)

        # 3. Christofides algorithm approximation for small clusters
        if len(cluster) < 30:  # Limit for computational reasons
            christ_route = christofides_approximation(cluster, coordinates)
            christ_route = two_opt(christ_route, coordinates)
            candidate_routes.append(christ_route)

        # Select the best route
        best_route = min(candidate_routes, key=lambda r: cost_function(r, coordinates))
        routes.append(best_route)

    return routes

# ---------------------------------------------------------------------------
# helper to split routes exceeding capacity ---------------------------------
# ---------------------------------------------------------------------------
def enforce_capacity(routes, demands, capacity, depot=1):
    """Split routes greedily so that each respects the capacity bound."""
    fixed = []
    for rt in routes:
        cur = [depot]
        load = 0
        for node in rt[1:]:
            if node == depot:
                continue
            demand = demands[node]
            if load + demand > capacity:
                cur.append(depot)
                if len(cur) > 2:
                    fixed.append(cur)
                cur = [depot, node]
                load = demand
            else:
                cur.append(node)
                load += demand
        if cur[-1] != depot:
            cur.append(depot)
        if len(cur) > 2:
            fixed.append(cur)
    return fixed

#--------------------Two-Phase MKP-Based CVRP Heuristic-------------------------------
def MSH_alg(coordinates, demands, capacity, num_nodes, time_limit=60):
    """
    Two-Phase MKP-Based CVRP Heuristic

    Phase 1: Clustering - Assign customers to vehicles using MKP
    Phase 2: Routing - Solve TSP for each cluster

    Parameters:
    - coordinates: Dictionary mapping node indices to (x, y) coordinates
    - demands: Dictionary mapping node indices to demand values
    - capacity: Vehicle capacity
    - num_nodes: Total number of nodes including depot

    Returns:
    - optimized_routes: List of routes (each route is a list of nodes)
    - total_cost: Total cost of all routes
    """
    time_start = time.time()
    # Determine number of vehicles needed (simple estimation)
    total_demand = sum(demands.values()) - demands[1]  # Exclude depot
    num_vehicles = math.ceil(total_demand / capacity)

    # Phase 1: Clustering using MKP
    clusters = solve_mkp(coordinates, demands, capacity, num_vehicles)
    if time.time() - time_start > time_limit:
        print("[WARN] MKP clustering exceeded time limit, returning empty clusters")
        return {}, 0

    for cluster_id, cluster in clusters.items():
        cluster_demand = sum(demands[i] for i in cluster if i != 1)

    # Phase 2: Routing using TSP for each cluster
    routes = solve_tsp_for_clusters(clusters, coordinates)
    if time.time() - time_start > time_limit:
        print("[WARN] TSP routing exceeded time limit, returning empty routes")
        return [], 0

    # Validate and calculate total cost
    total_cost = 0
    for route in routes:
        route_cost = cost_function(route, coordinates)
        total_cost += route_cost

        # Validate route capacity
        route_demand = sum(demands[i] for i in route if i != 1)

    # Final validation & repair if necessary
    if not validate_routes_capacity(routes, coordinates, demands, capacity):
        print("[WARN] capacity exceeded – repairing routes")
        routes = enforce_capacity(routes, demands, capacity)
    if not validate_routes_cities(routes, coordinates, demands, capacity):
        print("[WARN] missing cities detected – repairing")
        routes = enforce_capacity(routes, demands, capacity)

    # recompute total cost after any repair
    total_cost = sum(cost_function(r, coordinates) for r in routes)

    return routes, total_cost

