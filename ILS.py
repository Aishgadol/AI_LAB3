"""ILS (Iterated Local Search) algorithm for solving the Capacitated Vehicle Routing Problem (CVRP).
based on meta heuristics such as Tabu Search, Ant Colony Optimization (ACO), and Simulated Annealing."""


#-------------------Imports-------------------------------
from utils import * # from utils import functions
from MSH import MSH_alg
import time
import concurrent.futures
import multiprocessing
import sys
import random
import copy
import numpy as np

#--------------------Functions for ILS-------------------------------
def f(solution, coordinates): #heuristic function
    total_distance = 0
    for route in solution:
        total_distance += cost_function(route, coordinates)
    return total_distance

def N(s,coordinates,demands,capacity):
    neighbors = []
    depot = 1
    neighbors_size = 300
    red_flag = {'relocate': 0, 'swap': 0, 'cross-exchange': 0, 'two-opt': 0}  # Track number of times each operator fails
    raise_flag = 10
    operators = ["relocate", "swap", "cross-exchange","two-opt"]  # List of operators to use

    while neighbors_size > 0:
        new_solution = None
        neighbors_size -= 1
        for operator in operators:
            if red_flag[operator] > raise_flag:
                # If an operator has failed too many times, skip it
                if operator == "relocate":
                    raise ValueError("Relocation operator failed too many times. Check the input data.")
                operators.remove(operator)
                red_flag[operator] = 0

        operator = random.choice(operators)  # Randomly select an operator

        match operator:
            case "relocate":
                new_solution = relocate(s, demands, capacity, coordinates)  # Apply relocation operator
            case "swap":
                new_solution = swap(s, demands, capacity)
            case "cross-exchange":
                new_solution = cross_exchange(s, demands, capacity)
            case "two-opt":
                route = random.choice(s)  # Select a random route
                if len(route) <= 2:
                    continue
                new_route = two_opt(route, coordinates, demands , capacity)  # Apply two-opt to the selected route
                new_solution = copy.deepcopy(s)
                new_solution[s.index(route)] = new_route
            case _:
                raise ValueError(f"Unknown operator: {operator}")

        if new_solution == s:
            continue
        cap = validate_routes_capacity(new_solution, coordinates, demands, capacity)
        cities = validate_routes_cities(new_solution, coordinates, demands, capacity)
        if cap and cities:  # Check if the new solution is valid
            neighbors.append(new_solution)
        else:
            red_flag[operator] += 1  # Increment the failure count for the operator



    return neighbors



def L(s, coordinates, demands, capacity): # check if the solution is legal
    cap = validate_routes_capacity(s, coordinates, demands, capacity)
    routes = validate_routes_cities(s, coordinates, demands, capacity)
    return cap and routes

def S(nbrs, coordinates, demands, capacity): # select the best legal neighbor
    legal_neighbors = [n for n in nbrs if L(n, coordinates, demands, capacity)]
    return min(legal_neighbors, key=lambda x: f(x, coordinates)) if legal_neighbors else None

def LocalSearchStep(s,coordinates, demands, capacity):
    improved = True
    max_trials = 10
    new_s =  copy.deepcopy(s)
    while improved and max_trials > 0:
        max_trials -= 1
        neighbors = N(s,coordinates,demands,capacity) # neighbors of the current solution
        s_prime = S(neighbors, coordinates, demands, capacity) # select the best legal neighbor
        if s_prime and f(s_prime, coordinates) < f(s, coordinates): # check if the new solution is better
            new_s = copy.deepcopy(s_prime)
        else:
            improved = False
    return new_s  # Return the best solution found in this local search step

def generate_new_solution(s,coordinates, demands, capacity):
    operators = ["relocate", "swap", "cross-exchange", "two-opt"]  # List of operators to use
    for _ in range(20):  # Try 20 times to generate a new solution
        operator = random.choice(operators)  # Randomly select an operator
        if operator == "relocate":
            new_s = relocate(s, demands, capacity, coordinates)
        elif operator == "swap":
            new_s = swap(s, demands, capacity)
        elif operator == "two-opt":
            route = random.choice(s)
            if len(route) <= 2:
                return s
            new_route = two_opt(route, coordinates, demands, capacity)  # Apply two-opt to the selected route
            new_s = copy.deepcopy(s)
            new_s[s.index(route)] = new_route
        elif operator == "cross-exchange":
            new_s = cross_exchange(s, demands, capacity)  # Apply cross-exchange operator

        cap = validate_routes_capacity(new_s, coordinates, demands, capacity)
        cities = validate_routes_cities(new_s, coordinates, demands, capacity)
        if cap and cities:  # Check if the new solution is valid
            return new_s
    return s

def generate_initial_solution(coordinates, demands, capacity, num_nodes):
    nbrs = nearest_neighbors_CVRP(coordinates, num_neighbors=num_nodes)
    routes = initialize_routes(coordinates, demands, capacity,num_nodes,nbrs)

    return routes  # Return the initial solution as a list of routes

def remove_empty_routes(routes):
    """Remove empty routes from the solution."""
    return [route for route in routes if len(route) > 2]  # Keep only routes with at least one customer (depot + customer)

def start_end_depot(routes):
    """Ensure all routes start and end at the depot."""
    for route in routes:
        if route[0] != 1:  # If the first node is not the depot
            route.insert(0, 1)  # Add depot at the start
        if route[-1] != 1:  # If the last node is not the depot
            route.append(1)  # Add depot at the end
    return routes


#-------------Functions for Simulated Annealing--------------------

def update_temperature(T, alpha, iteration , max_iter, reheating=False, Tmin=1e-3):
    """Update the temperature for simulated annealing."""
    if reheating:
        # Reheat by increasing temperature
        return T * 3.0  # Triple the temperature to escape local optima
    else:
        # Standard geometric cooling
        return max(T * alpha, Tmin)

def metropolis_step(T,s,coordinates, demands, capacity):
    """Perform a Metropolis step for simulated annealing."""
    neighbors = N(s, coordinates, demands, capacity)
    legal_neighbors = [n for n in neighbors if L(n, coordinates, demands, capacity)]

    if not legal_neighbors:
        return s

    n = random.choice(legal_neighbors)  # Select a random legal neighbor
    delta_f = f(s, coordinates) - f(n, coordinates)  # Calculate cost difference
    if delta_f > 0:
        # If the neighbor is better, accept it
        return n
    else:
        # accept worse neighbors with a probability based on temperature
        exponent = -delta_f / T if T > 0 else 0
        exponent = np.clip(exponent, -700, 700)  # Prevent overflow
        probability = np.exp(exponent)
        if random.random() < probability:
            return n
    return s

def calculate_initial_temperature(s, coordinates, demands, capacity):
    """
    Calculate initial temperature so that worse moves have ~50% acceptance probability
    at the start of the search.
    """
    # Sample cost differences from random neighbor moves
    cost_diffs = []
    current_cost = f(s, coordinates)

    # Generate sample neighbors and calculate cost differences
    for _ in range(10):
        neighbors = N(s, coordinates, demands, capacity)
        legal_neighbors = [n for n in neighbors if L(n, coordinates, demands, capacity)]
        if legal_neighbors:
            neighbor = random.choice(legal_neighbors)
            neighbor_cost = f(neighbor, coordinates)
            cost_diff = neighbor_cost - current_cost
            if cost_diff > 0:  # Only consider worse moves
                cost_diffs.append(cost_diff)

    # Calculate initial temperature based on average cost difference
    if cost_diffs:
        avg_diff = sum(cost_diffs) / len(cost_diffs)
        # Set initial T so that probability of accepting avg_diff is 0.5
        # Using p = exp(-diff/T) = 0.5 => T = -diff/ln(0.5)
        return avg_diff / 0.693  # ln(0.5) ≈ -0.693
    else:
        return 10.0  # Default fallback




#--------------------Ant Colony Optimization (ACO) Functions-------------------------------
def aco_main(inital_solution, coordinates, demands, capacity, num_nodes, max_searches, params, time_limit,update_parameters=False):
    """Main function for Ant Colony Optimization (ACO) for CVRP."""
    #parameters for ACO

    time_Start = time.time()
    nAnts = max(10, int(0.3 * num_nodes))  # Number of ants, at least 10 or 30% of nodes


    nVars = len(coordinates) + 1  # Number of variables (nodes)
    nEpochs = max(1,  max_searches//nAnts)  # Ensure at least one epoch
    # Extract parameters from params dictionary
    alpha = params.get('alpha', 1.0)  # Pheromone importance
    beta = params.get('beta', 2.0)    # Heuristic importance
    rho = params.get('rho', 0.1)      # Pheromone evaporation rate
    Q = params.get('Q', 1.0)          # Pheromone deposit factor
    Q = 1.0 * np.sqrt(num_nodes) / 10  # Scales with problem size
    # Initialize pheromone and distance matrices
    pheromone = np.ones((nVars, nVars)) * 0.1  # Pheromone matrix with small initial values
    heuristic = np.zeros((nVars, nVars))

    # Heuristic information (inverse of distance)
    for i in range(1,nVars):
        for j in range(1,nVars):
            if i != j:
                x1, y1 = coordinates[i]
                x2, y2 = coordinates[j]
                dist = distance_function(x1, y1, x2, y2)
                heuristic[i][j] = 1.0 / dist if dist > 0 else 0


    #initial population of ants
    ants = []
    for _ in range(nAnts):
        #s = generate_initial_solution(coordinates, demands, capacity, num_nodes)
        ants.append(generate_new_solution(inital_solution, coordinates, demands, capacity)) # Use the initial solution as the starting point for all ants

    #evaluate initial population
    best_solution = min(ants, key=lambda x: f(x, coordinates))  # Best solution found by ants
    best_cost = f(best_solution, coordinates)  # Cost of the best solution

    # Store metrics for analysis
    metrics = {
    'best_costs': [],
    'avg_costs': [],
    'worst_costs': [],
    'parameters': {'alpha': [alpha], 'beta': [beta], 'rho': [rho], 'Q': [Q]}
    }


    #main ACO loop
    iteration_without_improvement = 0  # Counter for iterations without improvement
    for epoch in range(nEpochs):
        new_ants = []
        #construct new ant population solutions
        for ant in range(nAnts):
            ant_s = ant_solution(pheromone, heuristic, coordinates, demands, capacity, alpha, beta)  # Generate a solution for the ant
            # Introduce randomness in path selection
            if random.random() < 0.2:  # 20% chance to select a random path
                ant_s = generate_new_solution(ant_s, coordinates, demands, capacity)
                ant_s = LocalSearchStep(ant_s, coordinates, demands, capacity)  # Apply local search to the new solution

            if L(ant_s, coordinates, demands, capacity):
                ant_s = start_end_depot(ant_s)  # Ensure all routes start and end at the depot
                ant_s = remove_empty_routes(ant_s)  # Remove empty routes if they exist
                new_ants.append(ant_s)


        # If no new ants were generated, use the previous ants
        if not new_ants:
            new_ants = ants


        # randomly  reset pheromone trails
        if iteration_without_improvement > 20:  # If stuck for 20 epochs
            if random.random() < 0.2:  # 20% chance to reset pheromone trails
                #pheromone = np.ones((nVars, nVars)) * 0.1  # Reset pheromone to a low value
                # Reset only partially - keep some memory
                reset_factor = 0.2 + 0.3 * (iteration_without_improvement / 20)  # 0.2-0.5 based on stagnation
                pheromone = pheromone * (1-reset_factor) + np.ones((nVars, nVars)) * 0.1 * reset_factor
                iteration_without_improvement = 0  # Reset counter after reset



        #update pheromone trails
        pheromone = update_pheromone(pheromone, new_ants, coordinates, rho,Q )


        # update global best ant solution
        current_best = min(new_ants, key=lambda x: f(x, coordinates))
        current_best_cost = f(current_best, coordinates)



        if current_best_cost < best_cost:
            best_solution = copy.deepcopy(current_best)
            best_cost = current_best_cost
            # Exploitation phase - focus more on the pheromone
            alpha = min(3.0, alpha * 1.1)  # Increase pheromone importance
            beta = max(1.0, beta * 0.95)   # Decrease heuristic importance
            Q = min(5.0, Q * 1.05)  # Increase pheromone deposit factor
            rho = max(0.01, rho * 0.95)  # Decrease pheromone evaporation rate
            iteration_without_improvement = 0 # Reset counter if improvement is found
        else:
            iteration_without_improvement += 1

        # If no improvement, adjust parameters for exploration
        if iteration_without_improvement == max(5, nEpochs // 10):  # Update parameters when stuck for 10% of epochs
            # Exploration phase - focus more on the heuristic
            alpha = max(0.5, alpha * 0.9)  # Reduce pheromone importance
            beta = min(5.0, beta * 1.1)    # Increase heuristic importance
            Q = max(0.5, Q * 0.95)  # Decrease pheromone deposit factor
            # For large problems, use faster evaporation
            if num_nodes > 50:
                rho = min(0.2, rho * 1.5)  # Increase evaporation rate
            else:
                rho = min(0.5, rho * 1.05)  # Increase pheromone evaporation rate

        #Encourage ants to explore new paths by using memory
        if epoch % 5 == 0:  # Every 5 epochs, encourage exploration
            for i in range(nAnts):
                # Only perturb if the solution is not the current best
                if not np.array_equal(new_ants[i], current_best):
                    new_ants[i] = generate_new_solution(new_ants[i], coordinates, demands, capacity)
                    new_ants[i] = start_end_depot(new_ants[i])  # Ensure all routes start and end at the depot
                    new_ants[i] = remove_empty_routes(new_ants[i])  # Remove empty routes after perturbation


        ants = new_ants  # Update ants for the next epoch

        # Store parameters and costs for analysis
        metrics['parameters']['alpha'].append(alpha)
        metrics['parameters']['beta'].append(beta)
        metrics['parameters']['rho'].append(rho)
        metrics['parameters']['Q'].append(Q)
        metrics['best_costs'].append(current_best_cost)
        metrics['avg_costs'].append(np.mean([f(ant, coordinates) for ant in new_ants]))
        metrics['worst_costs'].append(np.max([f(ant, coordinates) for ant in new_ants]))

        if time.time() - time_Start > time_limit:
            break

    return best_solution , metrics  # Return the best solution and metrics for analysis


def roulette_wheel_selection(probabilities):
    """
    Select a node using roulette wheel selection based on probabilities
    """
    rand_val = random.random()
    cumulative_prob = 0

    for node, prob in probabilities.items():
        cumulative_prob += prob
        if rand_val <= cumulative_prob:
            return node

    # Fallback: return random node if something goes wrong
    return random.choice(list(probabilities.keys()))



def ant_solution(pheromone, heuristic, coordinates, demands, capacity, alpha=1.0, beta=2.0):
    """Generate a solution for one ant using pheromone and heuristic information."""

    num_nodes = len(coordinates) + 1
    depot = 1
    unvisited = set(range(1,num_nodes))
    unvisited.remove(depot)

    solution = []

    while unvisited:
        route = [depot]
        current_load = 0
        current_node = depot

        while unvisited:
            # Calculate probabilities for next node selection
            probabilities = {}
            denominator = 0

            # first calculate denominator sum
            for next_node in unvisited:
                if current_load + demands[next_node] <= capacity:
                    # Calculate probability based on pheromone and heuristic
                    tau = pheromone[current_node][next_node]
                    eta = heuristic[current_node][next_node]
                    denominator += (tau ** alpha) * (eta ** beta)

            # calculate probabilities for each feasible next node
            for next_node in unvisited:
                if current_load + demands[next_node] <= capacity:
                    tau = pheromone[current_node][next_node]
                    eta = heuristic[current_node][next_node]
                    probabilities[next_node] = ((tau ** alpha) * (eta ** beta)) / denominator if denominator > 0 else 0


            if not probabilities:
                break  # No feasible moves, end this route

            # Select next node using roulette wheel selection
            next_node = roulette_wheel_selection(probabilities)

            # Add node to route
            route.append(next_node)
            current_load += demands[next_node]
            current_node = next_node
            unvisited.remove(next_node)

        route.append(depot)  # Return to depot

        solution.append(route)

    if not L(solution, coordinates, demands, capacity):
        print("debug ACO: Generated solution is not legal. Retrying...")
        raise ValueError("Generated solution is not legal. Please check the input data.")

    if random.random() < 0.15:  # 15% chance for diversity-enhancing mutation
        # Apply a small perturbation to encourage diversity
        routes_to_modify = random.randint(1, max(1, len(solution)//2))
        for _ in range(routes_to_modify):
            if solution:
                route_idx = random.randint(0, len(solution)-1)
                route = solution[route_idx]
                if len(route) > 3:  # Only modify routes with at least one customer
                    # Shuffle a portion of the route (keeping depot)
                    mid_start = random.randint(1, len(route)-2)
                    mid_end = random.randint(mid_start, len(route)-2)
                    mid_section = route[mid_start:mid_end+1]
                    random.shuffle(mid_section)
                    solution[route_idx] = route[:mid_start] + mid_section + route[mid_end+1:]

    return solution


def update_pheromone(pheromone, ants, coordinates, rho=0.1, Q=1.0):
    """update  pheromone trails based on the ants' solutions."""
    num_nodes = pheromone.shape[0]

    # calculate delta tau for all edges
    delta_tau = np.zeros_like(pheromone)
    for solution in ants:
        solution_length = sum(cost_function(route, coordinates) for route in solution)
        if solution_length > 0: # Avoid division by zero
            for route in solution:
                for i in range(len(route) - 1):
                    from_node = route[i]
                    to_node = route[i + 1]
                    delta_tau[from_node][to_node] += Q / solution_length
                    delta_tau[to_node][from_node] += Q / solution_length
    # update pheromone using the formula τij(t+1) = (1-ρ)τij(t) + ρΔτij(t)
    pheromone = (1-rho) * pheromone + rho * delta_tau

    #ensure minimum pheromone level
    pheromone = np.maximum(pheromone, 0.01)  # Avoid zero pheromone levels
    return pheromone




#-------------------- Tabu Search Functions-------------------------------
def Local_search_tabu(s,tabu,n, coordinates, demands, capacity):
    """Perform a local search step for Tabu Search."""
    max_trials = 10
    new_s =  copy.deepcopy(s)
    current_cost = f(new_s, coordinates)

    for _ in range(max_trials):
        neighbors = N(new_s,coordinates,demands,capacity) # neighbors of the current solution

        non_tabu_neighbors = []
        for neighbor in neighbors:
            if not L(neighbor, coordinates, demands, capacity):
                continue
            # Create a hashable representation of the solution
            neighbor_key = solution_to_hashable(neighbor)
            neighbor_cost = f(neighbor, coordinates)

            if neighbor_key in tabu:
                # In tabu list - only add if it satisfies aspiration criterion
                if neighbor_cost < current_cost:
                    non_tabu_neighbors.append(neighbor)
            else:
                # Not in tabu list - always add
                non_tabu_neighbors.append(neighbor)

        if not non_tabu_neighbors:
            neighbors = [n for n in neighbors if L(n, coordinates, demands, capacity)]  # Fallback to all legal neighbors
            s_prime = S(neighbors, coordinates, demands, capacity) # select the best legal neighbor
        else:
            neighbors = non_tabu_neighbors
            s_prime = S(neighbors, coordinates, demands, capacity) # select the best legal neighbor


        if s_prime and f(s_prime, coordinates) < current_cost: # check if the new solution is better
            new_s = copy.deepcopy(s_prime)
            new_cost = f(new_s, coordinates)
            delta_cost = current_cost - new_cost  # Positive value indicates improvement

            # Add the selected solution to tabu list
            solution_key = solution_to_hashable(s_prime)
            tabu[solution_key] = {
                'cost': new_cost,
                'delta_cost': delta_cost,
                'tenure': n,
                'generation': 0
            }

            current_cost = new_cost


    return new_s,tabu  # Return the best solution found in this local search step

def solution_to_hashable(solution):
    """Convert a CVRP solution (list of routes) to a hashable representation."""
    # Sort each route to ensure consistent representation
    sorted_solution = []
    for route in solution:
        # Convert route to tuple (hashable)
        sorted_solution.append(tuple(route))

    # Sort routes by first customer to ensure consistent ordering
    sorted_solution.sort()

    # Return as a tuple of tuples (fully hashable)
    return tuple(sorted_solution)

def hashable_to_solution(hashable_solution):
    """Convert a hashable solution back to the original list of routes."""
    # Convert each tuple back to a list
    return [list(route) for route in hashable_solution]  # Convert each route back to a list

#--------------------Meta Heuristics-------------------------------
def tabu_search(coordinates, demands, capacity, num_nodes, time_limit, max_searches):
    import time
    start_time = time.time()

    use_msh = False  # Flag to indicate if MSH was used
    s = generate_initial_solution(coordinates, demands, capacity, num_nodes)
    if num_nodes <=80:  # Apply MSH algorithm for small problems
        msh_s ,best_cost = MSH_alg(coordinates, demands, capacity, num_nodes)  # Apply MSH algorithm to improve initial solution
        if f(msh_s,coordinates)  < f(s, coordinates) and L(msh_s, coordinates, demands, capacity):
            for route in msh_s:
                if route[-1] != 1:  # Ensure all routes end with the depot
                    raise ValueError("msh  routes must end with the depot (node 1).")
            print("Using MSH solution as initial solution.")
            use_msh = True
            s = msh_s

    if not L(s, coordinates, demands, capacity):
        print("Initial solution is legal.")
        print(s)
        raise ValueError("Initial solution is not legal. Please check the input data.")
    best_solution = LocalSearchStep(s,coordinates, demands, capacity)
    best_cost = f(best_solution, coordinates)

    # Dynamic upper bound based on problem size
    max_tenure = min(50, max(20, int(np.sqrt(num_nodes) * 3)))
    # Scale initial tenure with problem size
    n = min(max(7, int(np.sqrt(num_nodes) * 1.5)), 20)

    solutions = [best_cost] # Store costs for analysis
    parametes = [n] # Store tabu tenure for analysis

    tabu = {} # Use a node hash as a tabu-list
    solution_key = solution_to_hashable(best_solution)  # Convert solution to hashable form
    tabu[solution_key] = {
        'cost' : best_cost, # cost of the best solution
        'delta_cost' : f(s,coordinates)- best_cost , # delta cost of the best solution
        'tenure' : n, # initial tabu tenure
        'generation': 0  # Generation of the solution
    }

    iteration_without_improvement = 0

    for iteration in range(max_searches):
        parametes.append(n)  # Store the current tabu tenure for analysis

        s, tabu = Local_search_tabu(s,tabu,n, coordinates, demands, capacity)
        s = start_end_depot(s)  # Ensure all routes start and end at the depot
        s = remove_empty_routes(s)  # Remove empty routes after local search if they exist
        if L(s, coordinates, demands, capacity):
            solutions.append(f(s, coordinates))  # Store the cost of the current solution
            # Check if the new solution is better than the best found so far
            if f(s, coordinates) < f(best_solution, coordinates):
                #print(f"debug TS: Iteration {iteration}, new best solution found with cost {f(s, coordinates)}")
                best_solution = copy.deepcopy(s)
                iteration_without_improvement = 0

                # Increase tenure for better solutions
                n = min(max_tenure, n + 2)
                for key in tabu:
                    tabu[key]['tenure'] = n

            else:
                iteration_without_improvement += 1



        found = False
        # Generate new starting solution occasionally
        if iteration_without_improvement > 10:
            max_attempts = 20 # Number of attempts to find a new solution
            make_a_choice = ["Intensification", "Diversification"]
            choice = random.choice(make_a_choice)  # Randomly choose between intensification and diversification
            choice = "Diversification"  if not tabu else choice
            if choice == "Intensification":
                new_key = min(tabu.keys(), key=lambda k: tabu[k]['cost'])  # Fallback to the best solution in tabu list
                new_s = hashable_to_solution(new_key)  # Convert back to list of routes
                s = new_s  # Update current solution with the fallback solution
                if random.random() < 0.5:
                    s = best_solution  # 50% chance to use the best solution found so far

            else:
                # Diversification - generate a new solution
                n = 10 # Reset tabu tenure for diversification
                for _ in range(max_attempts):  # Ensure we generate a new solution
                    new_s = generate_new_solution(s,coordinates, demands, capacity)  # Generate a new solution
                    new_key = solution_to_hashable(new_s)
                    if new_key not in tabu:
                        found = True
                        s = new_s  # Update current solution with the new one
                        break
                if not found:
                    s = new_s #even if tabu use the new solution to continue the search

            s = start_end_depot(s)  # Ensure all routes start and end at the depot
            s = remove_empty_routes(s)  # Remove empty routes after local search if there are any
            # decrease tenure for all items in tabu list
            n = max(1, n - 1) # Decrease tabu tenure if no improvement is found
            for key in tabu:
                tabu[key]['tenure'] = n

        # Decrease tenure for all items in tabu list
        keys_to_remove = []
        for key in tabu:
            tabu[key]['generation'] += 1
            if tabu[key]['tenure'] <= tabu[key]['generation']:  # Check if the tenure has expired
                keys_to_remove.append(key)

        # Remove expired entries
        for key in keys_to_remove:
            tabu.pop(key)

        if time.time() - start_time > time_limit: # time limit check
            print("Time limit exceeded.")
            break


    metrics = {
        'best_costs': solutions,
        'parameters': parametes
    }  # Store costs and parameters for analysis

    return best_solution, metrics  # Return the best solution and the list of costs found


def aco(coordinates, demands, capacity, num_nodes, time_limit, max_searches):
    import time
    start_time = time.time()

    # Initial solution
    current_solution = generate_initial_solution(coordinates, demands, capacity, num_nodes)
    if num_nodes <=80:
        current_solution_msh, _ = MSH_alg(coordinates, demands, capacity, num_nodes) # Apply MSH algorithm to improve initial solution
        if f(current_solution_msh,coordinates) < f(current_solution, coordinates) and L(current_solution_msh, coordinates, demands, capacity):
            print("Using MSH solution as initial solution.")
            current_solution = current_solution_msh
    current_solution = start_end_depot(current_solution)  # Ensure all routes start and end at the depot
    current_solution =remove_empty_routes(current_solution)  # Remove empty routes after MSH if they exist
    current_solution = LocalSearchStep(current_solution,coordinates, demands, capacity)
    best_solution = copy.deepcopy(current_solution)
    best_cost = f(best_solution, coordinates)

    # Dynamic parameters
    alpha = 1.0  # Pheromone importance
    beta = 2.0   # Heuristic importance
    rho = 0.05    # Evaporation rate
    Q = 2.0      # Pheromone deposit factor

    solutions = [best_cost]  # Store costs for analysis
    metrics_all = {
        'best_costs': [],
        'avg_costs': [],
        'worst_costs': [],
        'parameters': { 'alpha': [], 'beta': [], 'rho': [], 'Q': [] },
        'solutions' : solutions
    }

    # For tracking improvement
    iterations_without_improvement = 0
    max_iterations_without_improvement = 1
    divide = min(5, max(3, num_nodes // 50))  # Between 3-5 runs
    original_divide = divide  # Store original divide for later use
    run_searches = max_searches // divide  # Divide searches evenly across runs
    i = 0
    while i < divide:
        i += 1

        # Adjust parameters based on search progress
        update_parameters = False
        if iterations_without_improvement > max_iterations_without_improvement:
            update_parameters = True
            """# Increase exploration if stuck in local optimum
            alpha = max(0.5, alpha * 0.9)  # Reduce pheromone importance
            beta = min(5.0, beta * 1.1)    # Increase heuristic importance
            rho = min(0.5, rho * 1.1)      # Increase evaporation rate
            Q = max(0.5, Q * 0.95)  # Decrease pheromone deposit factor"""


        # Run ACO with current parameters
        aco_params = {
            'alpha': alpha,
            'beta': beta,
            'rho': rho,
            'Q': Q
        }



        # Perturbation using ACO
        perturbed_solution, metrics = aco_main(current_solution,coordinates, demands, capacity, num_nodes, run_searches , aco_params,time_limit,update_parameters)

        metrics_all['best_costs'].extend(metrics['best_costs'])
        metrics_all['avg_costs'].extend(metrics['avg_costs'])
        metrics_all['worst_costs'].extend(metrics['worst_costs'])
        for key in metrics['parameters']:
            metrics_all['parameters'][key].extend(metrics['parameters'][key])

        # Local search
        for _ in range(20):  # Perform local search on the perturbed solution
            improved_solution = LocalSearchStep(perturbed_solution, coordinates, demands, capacity)
            perturbed_solution = copy.deepcopy(improved_solution)  # Update perturbed solution

        improved_solution = start_end_depot(improved_solution)  # Ensure all routes start and end at the depot
        improved_solution = remove_empty_routes(improved_solution)  # Remove empty routes after local search
        # Acceptance criterion (accept if better)
        if L(improved_solution, coordinates, demands, capacity):
            current_cost = f(improved_solution, coordinates)
            solutions.append(current_cost)  # Store cost for analysis
            current_solution = copy.deepcopy(improved_solution)
            if current_cost < best_cost:
                best_solution = copy.deepcopy(improved_solution)
                best_cost = current_cost
                #current_solution = copy.deepcopy(improved_solution)

                iterations_without_improvement = 0

                """# Exploitation phase - focus more on the pheromone
                alpha = min(3.0, alpha * 1.1)  # Increase pheromone importance
                beta = max(1.0, beta * 0.95)   # Decrease heuristic importance
                # When finding good solutions, reduce evaporation to preserve good trails
                rho = max(0.05, rho * 0.9)
                Q = min(2.0, Q * 1.05)  # Increase pheromone deposit factor"""
            else:
                iterations_without_improvement += 1

        if i== divide: # last iteration
            if time.time() - start_time < time_limit : # if there is time  and not stuck in local optimum
                # add more runs
                divide += 1
                run_searches = max(max_searches // divide, max_searches // original_divide)  # Adjust runs per search


        if time.time() - start_time > time_limit: # time limit check
            print("Time limit exceeded.")
            break

    metrics_all['solutions'] = solutions  # Add solutions to metrics for analysis

    return best_solution , metrics_all  # Return the best solution and the list of costs found



def simulated_annealing(coordinates, demands, capacity, num_nodes, time_limit, max_searches):
    import time
    start_time = time.time()

    s = generate_initial_solution(coordinates, demands, capacity, num_nodes)
    use_msh = False  # Flag to indicate if MSH was used
    if num_nodes <= 80:
        msh_s ,best_cost= MSH_alg(coordinates, demands, capacity, num_nodes)  # Apply MSH algorithm to improve initial solution
        if f(msh_s, coordinates) < f(s, coordinates) and L(msh_s, coordinates, demands, capacity):
            use_msh = True
            print("Using MSH solution as initial solution.")
            s = msh_s
    s = start_end_depot(s)  # Ensure all routes start and end at the depot
    s = remove_empty_routes(s)  # Remove empty routes after MSH if they exist
    best_solution = LocalSearchStep(s,coordinates, demands, capacity)
    best_cost = f(best_solution, coordinates)

    T = calculate_initial_temperature(s,coordinates, demands, capacity)  # Initial temperature
    alpha = 0.95  # Cooling rate
    iterations_without_improvement = 0

    costs = [f(best_solution, coordinates)]  # Store costs for analysis
    parameters = {
        'T': [T],
        'alpha': [alpha],
    }

    for iteration in range(max_searches):
        parameters['T'].append(T)  # Store temperature for analysis
        parameters['alpha'].append(alpha) # Store cooling rate for analysis

        reheating = False

        # Reinitialize solution if stuck for too long (different initial solution)
        if iterations_without_improvement > 30:
            #s = generate_initial_solution(coordinates, demands, capacity, num_nodes)  # Reinitialize solution
            if use_msh:
                s ,_= MSH_alg(coordinates, demands, capacity, num_nodes)  # Apply MSH algorithm to improve initial solution
            else:
                s = generate_initial_solution(coordinates, demands, capacity, num_nodes)  # Generate a new initial solution

            s = start_end_depot(s)  # Ensure all routes start and end at the depot
            s = remove_empty_routes(s)
            s = LocalSearchStep(s,coordinates, demands, capacity)  # Apply local search to the new solution
            T = calculate_initial_temperature(s, coordinates, demands, capacity)  # Reset temperature
            alpha = 0.95  # Reset cooling rate
            iterations_without_improvement = 0

        # If stuck for too long, reheat the temperature
        if iterations_without_improvement > 20:  # If stuck, reheat
            reheating = True
            alpha = min(0.99, alpha * 1.05)  # Increase cooling rate if no improvement

        # Perform local search and metropolis step
        s = metropolis_step(T, s, coordinates, demands, capacity)
        s = LocalSearchStep(s,coordinates, demands, capacity)

        s = start_end_depot(s)  # Ensure all routes start and end at the depot
        s = remove_empty_routes(s)  # Remove empty routes after local search if there are any
        if L(s, coordinates, demands, capacity):
            current_cost = f(s, coordinates)
            costs.append(current_cost)  # Store cost for analysis
            if current_cost <best_cost: # accept new solution if it is better
                #print(f"debug SA: Found a better solution with cost {current_cost} at iteration {iteration}.")
                best_solution = copy.deepcopy(s)
                best_cost = current_cost
                iterations_without_improvement = 0  # Reset counter
                alpha = max(0.1, alpha * 0.95)  # Decrease cooling rate if improvement found
            else:
                iterations_without_improvement += 1
                #alpha = min(0.99, alpha * 1.05)  # Increase cooling rate if no improvement

        T = update_temperature(T, alpha, iteration,max_searches,reheating)  # Update temperature

        if iteration % 10 == 0:  # initialize new solution every 10 iterations
            #s = generate_initial_solution(coordinates, demands, capacity, num_nodes)  # Reinitialize solution
            if use_msh:
                s ,_= MSH_alg(coordinates, demands, capacity, num_nodes)  # Apply MSH algorithm to improve initial solution
                s = start_end_depot(s)  # Ensure all routes start and end at the depot
                s = remove_empty_routes(s)  # Remove empty routes after local search if there are any
            else:
                s = generate_initial_solution(coordinates, demands, capacity, num_nodes)
                s = start_end_depot(s)  # Ensure all routes start and end at the depot
                s = remove_empty_routes(s)  # Remove empty routes after local search if there are any
            for _ in range(20):  # Try to improve the solution a few times
                s = generate_new_solution(s, coordinates, demands, capacity)  # Generate a new solution
                new_s = LocalSearchStep(s, coordinates, demands, capacity)  # Apply local search
                if f(new_s, coordinates) < f(s, coordinates):  # If the new solution is better
                    s = new_s
                    s = start_end_depot(s)  # Ensure all routes start and end at the depot
                    s = remove_empty_routes(s)  # Remove empty routes after local search if there are any
        if iteration % 100 == 0:  #
            s = best_solution  # Reset to best solution every 100 iterations




        if time.time() - start_time > time_limit: # time limit check
            print("Time limit exceeded.")
            break

    metrics = {
        'best_costs': costs,
        'parameters': parameters
    }  # Store costs and parameters for analysis

    return best_solution, metrics # Return the best solution and the list of costs found




#-------------------Algorithm-------------------------------
def ILS_normal(coordinates, demands, capacity, num_nodes ,max_searches = 1000):

    # ILS
    s = generate_initial_solution(coordinates, demands, capacity,num_nodes)
    if not L(s, coordinates, demands, capacity):
        print("Initial solution is legal.")
        print(s)
        raise ValueError("Initial solution is not legal. Please check the input data.")
    best_solution = s

    for _ in range(max_searches):
        s = LocalSearchStep(s,coordinates, demands, capacity)
        if L(s, coordinates, demands, capacity):
            print("Local search found a legal solution.")
            if f(s, coordinates) < f(best_solution, coordinates):
                print("Found a better solution.")
                best_solution = s
        s = generate_new_solution(s,coordinates, demands, capacity)
    return best_solution

def ILS(coordinates, demands, capacity, num_nodes, meta_heuristic, time_limit ,max_searches = 1000):
    """
    Iterated Local Search (ILS) for CVRP using specified meta-heuristic.
    returns the best solution found and its cost and metrics.
    """
    match meta_heuristic:
        case "tabu_search":
            routes, metrics = tabu_search(coordinates, demands, capacity, num_nodes, time_limit, max_searches)
        case "aco":
            routes, metrics = aco(coordinates, demands, capacity, num_nodes, time_limit, max_searches)
        case "simulated_annealing":
            routes, metrics = simulated_annealing(coordinates, demands, capacity, num_nodes, time_limit, max_searches)
        case "parallel_hybrid_ils":
            routes, cost, metrics = parallel_hybrid_ils(coordinates, demands, capacity, num_nodes, time_limit, max_searches)
            return routes, cost ,metrics # Return both routes and cost for parallel hybrid ILS
        case _:
            raise ValueError("Invalid meta-heuristic. Choose from 'tabu_search', 'aco', or 'simulated_annealing'.")

    if not L(routes, coordinates, demands, capacity):
        print("Final solution is legal.")
        print(routes)
        raise ValueError("Final solution is not legal. Please check the input data.")

    return routes , f(routes, coordinates) ,metrics # Return the best solution and its cost





def parallel_hybrid_ils(coordinates, demands, capacity, num_nodes, time_limit, max_searches=1000):
    """
    Parallel hybrid ILS that runs multiple meta-heuristics simultaneously.
    """

    # Fix for PyInstaller frozen executables
    try:
        if hasattr(sys, 'frozen'):
            multiprocessing.freeze_support()
            print("Frozen EXE detected, using sequential fallback.")
            return ils_sequence(coordinates, demands, capacity, num_nodes, time_limit, max_searches)
        # Force spawn method for compatibility
        print("Setting multiprocessing start method to 'spawn'.")
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        # Method may already be set
        print("Multiprocessing start method already set or not supported in this environment.")
        return ils_sequence(coordinates, demands, capacity, num_nodes, time_limit, max_searches)

    start_time = time.time()
    start_cpu = time.process_time()

    print(f"Starting parallel hybrid ILS with time limit: {time_limit} seconds and max searches: {max_searches}")
    # Initialize solution
    initial_solution = generate_initial_solution(coordinates, demands, capacity, num_nodes)
    best_solution = copy.deepcopy(initial_solution)
    best_cost = f(best_solution, coordinates)
    
    # Define algorithm functions with consistent interface
    algorithms = {
        "simulated_annealing": simulated_annealing,
        "aco": aco,
        "tabu_search": tabu_search
    }
    
    # Divide resources among algorithms
    time_alloc = time_limit * 0.95  # Leave 5% buffer for process overhead
    algo_time_limit = time_alloc # Time limit for each algorithm
    algo_searches = max_searches 
    
    # Run algorithms in parallel
    results = []
    metrics_dict = {}
    #with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(
                    algo_func, coordinates, demands, capacity, num_nodes, algo_time_limit, algo_searches
                ): algo_name
                for algo_name, algo_func in algorithms.items()
            }
            
            for future in concurrent.futures.as_completed(futures):
                algo_name = futures[future]
                try:
                    result = future.result()
                    # Support both (solution, metrics) and (solution, cost) return types
                    if isinstance(result, tuple) and len(result) == 2:
                        solution, metrics = result
                    else:
                        solution, metrics = result, {}
                    if L(solution, coordinates, demands, capacity):
                        solution = remove_empty_routes(solution)  # Clean up empty routes if any
                        cost = f(solution, coordinates)
                        results.append((solution, cost, algo_name))
                        metrics_dict[algo_name] = metrics
                    
                    run_time = time.time() - start_time
                    cpu_time = time.process_time() - start_cpu
                    print(f"{algo_name} completed | Elapsed  time: {run_time} | CPU time: {cpu_time} | remaining time: {time_limit - (run_time):.1f} seconds")
                        
                except Exception as e:
                    print(f"{algo_name} generated an exception: {e}")
    except Exception as e:
        print(f"Process pool failed: {e}")
        print("Falling back to sequential execution")
        # Fallback to sequential execution
        results = []
        for algo_name, algo_func in algorithms.items():
            try:
                algo_time_limit = time_limit / 3  # Divide time limit among algorithms
                solution, metrics = algo_func(coordinates, demands, capacity, 
                                            num_nodes, algo_time_limit, algo_searches)
                results.append((solution, f(solution, coordinates), algo_name))
            except Exception as e:
                print(f"{algo_name} failed: {e}")
    
    # Find best solution from parallel runs
    if results:
        results.sort(key=lambda x: x[1])  # Sort by cost
        best_parallel_solution, best_parallel_cost, best_algo = results[0]
        
        if best_parallel_cost < best_cost:
            best_solution = best_parallel_solution
            best_cost = best_parallel_cost
            print(f"Best solution found by {best_algo} with cost {best_cost}")
    
    # Use remaining time for hybrid optimization
    remaining_time = max(0, time_limit - (time.time() - start_time))
    if remaining_time > 0:
        print(f"Using remaining {remaining_time:.1f} seconds for hybrid optimization")
        # Take top solutions from each algorithm and combine/improve
        solutions_to_combine = [result[0] for result in results[:3]]
        if not solutions_to_combine:
            solutions_to_combine = [best_solution]
        
        
        # Apply local search to best solutions
        for solution in solutions_to_combine:
            improved = LocalSearchStep(solution, coordinates, demands, capacity)
            if L(improved, coordinates, demands, capacity):
                improved_cost = f(improved, coordinates)
                if improved_cost < best_cost:
                    best_solution = improved
                    best_cost = improved_cost
                    print(f"Hybrid optimization improved solution to {best_cost}")
    
    metrics_dict['best_costs'] = {algo: cost for _, cost, algo in results}
    
    return best_solution, best_cost,  metrics_dict

def ils_sequence(coordinates, demands, capacity, num_nodes, time_limit, max_searches=1000):
    """
    ILS that runs multiple meta-heuristics subsequently """
    
    print("Running ILS in sequential mode due to environment limitations.")
    
    start_time = time.time()
    start_cpu = time.process_time()

    
    # Initialize solution
    initial_solution = generate_initial_solution(coordinates, demands, capacity, num_nodes)
    best_solution = copy.deepcopy(initial_solution)
    best_cost = f(best_solution, coordinates)
    
    # Define algorithm functions with consistent interface
    algorithms = {
        "simulated_annealing": simulated_annealing,
        "aco": aco,
        "tabu_search": tabu_search
    }
    
    # Divide resources among algorithms
    algo_searches = max_searches 
    
    # Run algorithms in sequentially
    metrics_dict = {}
    results = []
    for algo_name, algo_func in algorithms.items():
        try:
            
            algo_time_limit = time_limit // 3  # Divide time limit among algorithms
            solution, metrics = algo_func(coordinates, demands, capacity, 
                                        num_nodes, algo_time_limit, algo_searches)
            results.append((solution, f(solution, coordinates), algo_name))
            metrics_dict[algo_name] = metrics
            run_time = time.time() - start_time
            cpu_time = time.process_time() - start_cpu
            print(f"{algo_name} completed | cost { f(solution, coordinates)} | Elapsed  time: {run_time} | CPU time: {cpu_time} | remaining time: {time_limit - (run_time):.1f} seconds")
                        
        except Exception as e:
            print(f"{algo_name} failed: {e}")
    
    # Find best solution from parallel runs
    if results:
        results.sort(key=lambda x: x[1])  # Sort by cost
        best_parallel_solution, best_parallel_cost, best_algo = results[0]
        
        if best_parallel_cost < best_cost:
            best_solution = best_parallel_solution
            best_cost = best_parallel_cost
            print(f"Best solution found by {best_algo} with cost {best_cost}")
    
    # Use remaining time for hybrid optimization
    remaining_time = max(0, time_limit - (time.time() - start_time))
    if remaining_time > 0:
        print(f"Using remaining {remaining_time:.1f} seconds for hybrid optimization")
        # Take top solutions from each algorithm and combine/improve
        solutions_to_combine = [result[0] for result in results[:3]]
        if not solutions_to_combine:
            solutions_to_combine = [best_solution]
        
        
        # Apply local search to best solutions
        for solution in solutions_to_combine:
            improved = LocalSearchStep(solution, coordinates, demands, capacity)
            if L(improved, coordinates, demands, capacity):
                improved_cost = f(improved, coordinates)
                if improved_cost < best_cost:
                    best_solution = improved
                    best_cost = improved_cost
                    print(f"Hybrid optimization improved solution to {best_cost}")
    
    metrics_dict['best_costs'] = {algo: cost for _, cost, algo in results}
    
    return best_solution, best_cost,  metrics_dict