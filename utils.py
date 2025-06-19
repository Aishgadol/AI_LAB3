""" Common utility functions for the project. """


#--------------- Import Libraries ---------------
import numpy as np
import random
import copy

#--------------- Function Definitions ---------------

"""------------- Distance and Cost Functions ----------------"""

def distance_function(x1, y1, x2, y2): 
    """Calculate the EUC_2D between two points."""
    return round(np.sqrt((x1-x2)**2 + (y1-y2)**2))

def cost_function(route, coordinates):
    """Calculate the total cost of a route."""
    total_cost = 0
    for i in range(len(route) - 1):
        x1, y1 = coordinates[route[i]]
        x2, y2 = coordinates[route[i + 1]]
        total_cost += distance_function(x1, y1, x2, y2)
    return total_cost


"""----------------- Validation Functions ----------------"""

def validate_routes_capacity(routes, coordinates, demands, capacity):
    """Validate the routes against the capacity and demand constraints."""
    
    for route in routes:
        total_demand = sum(demands[node] for node in route)
        if total_demand > capacity: #cacpacity check
            return False
    
    return True


def validate_routes_cities(routes, coordinates, demands, capacity):
    """Validate that each city is visited exactly once."""
    # Skip depot in validation
    depot = 1
    
    # Get all cities (excluding depot)
    all_cities = set(coordinates.keys())
    all_cities.remove(depot)
    
    # Track visited cities
    visited_cities = set()
    
    # Check for duplicates
    for route in routes:
        for city in route:
            if city != depot:  # Ignore depot
                if city in visited_cities:
                    return False
                visited_cities.add(city)
    
    # Check if all cities are visited
    if visited_cities != all_cities:
        missing = all_cities - visited_cities
        return False
    
    return True


""" ------------------ Initialization and Nearest Neighbors ----------------- """

def nearest_neighbors_CVRP(coordinates, num_neighbors=1):
    """Find the nearest neighbors for each point in the coordinates dict."""
    from sklearn.neighbors import NearestNeighbors
    
    node_ids = list(coordinates.keys())
    coords_list = [coordinates[nid] for nid in node_ids]
    
    nbrs = NearestNeighbors(n_neighbors=num_neighbors, algorithm='ball_tree').fit(coords_list)
    distances, indices = nbrs.kneighbors(coords_list)

    # Build the neighbor dict (skip self )
    neighbors_dict = {}
    for i, nid in enumerate(node_ids):
        neighbor_indices = indices[i][1:]  # skip self
        neighbors = [node_ids[n] for n in neighbor_indices]
        neighbors_dict[nid] = neighbors
    
    return neighbors_dict

def most_connected_node(nbrs,visited):
    #find the node with the most connections
    max_connections = 0
    most_connected = None
    for node, neighbors in nbrs.items():
        if len(neighbors) > max_connections and visited[node] == False:
            max_connections = len(neighbors)
            for neighbor in neighbors: # check if the neighbor is already visited
                # if the neighbor is already visited, we cannot count it
                if visited[neighbor] == True:
                    max_connections -= 1
            most_connected = node
    return most_connected

def initialize_routes(coordinates, demands, capacity,num_nodes,nbrs):
    """Initialize routes based on the nearest neighbors heuristic."""
    
    routes = []
    visited = {i: False for i in coordinates}
    dead_end = False
    
    depot = 1 #depot is the first node
    visited[depot] = True   
    route = [depot]
    num_nodes -= 1
        
    #Stage 1 : start with the most connected node (Greedy)
    first_city = most_connected_node(nbrs,visited)
    route.append(first_city)
    remaining_capacity = capacity - demands[first_city]
    visited[first_city] = True
    num_nodes -= 1
    
    #Stage 2 : add the next city to the route - Nearest Neighbors (Greedy)
    while num_nodes > 0:
        current_city = route[-1]
        #check if the route is full or if we are at a dead end
        if remaining_capacity <= 0 or dead_end:
            route.append(depot) # return to depot
            routes.append(route)
            
            route = [depot] # start a new route from the depot
            first_city = most_connected_node(nbrs,visited)
            if first_city is None:
                break
            route.append(first_city)
            remaining_capacity = capacity - demands[first_city]
            visited[first_city] = True
            num_nodes -= 1
            dead_end = False
            
            
        #find the next city to add to the route
        next_city = None
        for neighbor in nbrs[current_city]:
            if not visited[neighbor] and demands[neighbor] <= remaining_capacity:
                next_city = neighbor
                break
        if next_city is not None:
            route.append(next_city)
            remaining_capacity -= demands[next_city]
            visited[next_city] = True
            num_nodes -= 1
            dead_end = False
        else: 
            dead_end = True
            
    route.append(depot) # return to depot
    routes.append(route) # add the last route
    
    return routes


"""------------------ Mutation Operators -----------------"""
    
def relocate(routes, demands, capacity , coordinates):
    """Relocate a node from one route to another."""
    
    new_routes = copy.deepcopy(routes)  # Avoid modifying the original routes
    
    # Randomly select a route and a node to relocate
    route_from = random.choice(new_routes)
    if len(route_from) <= 2:  # Cannot relocate if only depot is present
        return routes
    
    new_routes.remove(route_from)  # Remove the selected route from the list
    
    node_to_relocate = random.choice(route_from[1:-1])  # Exclude depot and last node
    
    # Find a suitable route to relocate the node
    suitable_routes = [r for r in new_routes if sum(demands[n] for n in r) + demands[node_to_relocate] <= capacity]
    
    if not suitable_routes:
        return routes  # No suitable route found
    
    route_from.remove(node_to_relocate)
    if len(route_from) > 2 : # If there are still nodes other than depot, add it back to the new routes
        new_routes.append(route_from)  # Add the modified route back to the list
    
    route_to = random.choice(suitable_routes) # Randomly select a suitable route to insert the node
    new_routes.remove(route_to)  # Remove the selected route to insert the node
    
    # Insert the node into the new route at a random position
    insert_position = random.randint(1, len(route_to) - 1)  # Exclude depot
    route_to.insert(insert_position, node_to_relocate)
    
    new_routes.append(route_to)  # Add the modified route back to the list
    
    return new_routes

def swap(routes, demands, capacity):
    """Swap two nodes between two routes."""
    
    new_routes = copy.deepcopy(routes)  # Avoid modifying the original routes
    
    # Randomly select two different routes
    route_from = random.choice(new_routes)
    route_to = random.choice([r for r in new_routes if r != route_from])
    
    if len(route_from) <= 2 or len(route_to) <= 2:  # Cannot swap if only depot is present
        return routes
    
    new_routes.remove(route_from)  # Remove the selected route from the list
    new_routes.remove(route_to)  # Remove the selected route from the list
    
    # Randomly select nodes to swap
    node_from = random.choice(route_from[1:-1])  # Exclude depot and last node
    node_to = random.choice(route_to[1:-1])  # Exclude depot and last node
    
    # Check if the swap maintains capacity constraints
    if (sum(demands[n] for n in route_from) - demands[node_from] + demands[node_to] > capacity or
        sum(demands[n] for n in route_to) - demands[node_to] + demands[node_from] > capacity):
        return routes  # Swap not valid, return unchanged routes
    
    # Perform the swap
    route_from.remove(node_from)
    route_to.remove(node_to)
    
    route_from.insert(len(route_from) - 1, node_to)  # Insert at the end before depot
    route_to.insert(len(route_to) - 1, node_from)  # Insert at the end before depot
    
    new_routes.append(route_from)  # Add the modified route back to the list
    new_routes.append(route_to)  # Add the modified route back to the list
            
    return new_routes

def cross_exchange(routes, demands, capacity):
    """ Exchange subsequences of customers between two routes (while
        respecting capacity constraints."""
        
    new_routes = copy.deepcopy(routes)  # Avoid modifying the original routes
        
    # Randomly select two different routes
    route_from = random.choice(routes)
    route_to = random.choice([r for r in routes if r != route_from])
    
    if len(route_from) <= 2 or len(route_to) <= 2:  # Cannot cross-exchange if only depot is present
        return routes

    new_routes.remove(route_from)  # Remove the selected route from the list
    new_routes.remove(route_to)  # Remove the selected route from the list
    
    # Randomly select nodes to exchange
    start_from_1 = random.randint(1, len(route_from) - 2)  # Exclude depot and last node
    end_from_1 = random.randint(start_from_1 + 1, len(route_from) - 1)  # Ensure at least one node is selected
    start_from_2 = random.randint(1, len(route_to) - 2)  # Exclude depot and last node
    end_from_2 = random.randint(start_from_2 + 1, len(route_to) - 1)  # Ensure at least one node is selected
    
    # Extract the subsequences
    subseq_from_1 = route_from[start_from_1:end_from_1] 
    subseq_from_2 = route_to[start_from_2:end_from_2]
    # Check if the exchange maintains capacity constraints
    if (sum(demands[n] for n in route_from) - sum(demands[n] for n in subseq_from_1) + sum(demands[n] for n in subseq_from_2) > capacity or
        sum(demands[n] for n in route_to) - sum(demands[n] for n in subseq_from_2) + sum(demands[n] for n in subseq_from_1) > capacity):
        return routes  # Exchange not valid, return unchanged routes
    
    # Perform the exchange
    route_from = route_from[:start_from_1] + subseq_from_2 + route_from[end_from_1:]
    route_to = route_to[:start_from_2] + subseq_from_1 + route_to[end_from_2:]
    
    new_routes.append(route_from)  # Add the modified route back to the list
    new_routes.append(route_to)  # Add the modified route back to the list
    
    return new_routes
        
def two_opt(route, coordinates, demands , capacity):
    """Two-Opt: Reverse the order of customers within a route"""
    best_route = route
    best_cost = cost_function(route, coordinates)
    
    for i in range(1, len(route) - 2):  # Exclude depot and last node
        for j in range(i + 1, len(route) - 1):  # Exclude depot and last node
            if j - i == 1:  # No change if adjacent nodes
                continue
            
            new_route = copy.deepcopy(route)
            new_route[i:j] = reversed(new_route[i:j])  # Reverse the segment
            
            if validate_routes_capacity([new_route], coordinates, demands, capacity):
                new_cost = cost_function(new_route, coordinates)
                if new_cost < best_cost:
                    best_route = new_route
                    best_cost = new_cost
    
    return best_route

#slightly diff version for two opt route

def two_opt_route(route, coordinates, max_rounds=3):
    # apply 2-opt local search on a single route
    best_route = route[:]
    best_cost = cost_function(best_route, coordinates)
    n = len(best_route)
    for _ in range(max_rounds):
        improved = False
        for i in range(1, n-2):
            for j in range(i+1, n-1):
                new_route = best_route[:i] + best_route[i:j][::-1] + best_route[j:]
                new_cost = cost_function(new_route, coordinates)
                if new_cost < best_cost:
                    best_route, best_cost = new_route, new_cost
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return best_route



#--------------- Ackley Function ----------------

def Ackley_score(x):
    """Calculate Ackley function value"""
    a, b, c = 20, 0.2, 2*np.pi
    d = len(x)
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(c * x))
    
    term1 = -a * np.exp(-b * np.sqrt(sum1 / d))
    term2 = -np.exp(sum2 / d)
    
    return term1 + term2 + a + np.exp(1)

def generate_initial_solution_ackley(s, dim=10, lower_bound=-32.768, upper_bound=32.768):
    """Generate an initial solution within the bounds"""
    return np.random.uniform(lower_bound,upper_bound,dim)