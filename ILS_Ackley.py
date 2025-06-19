
"""ILS (Iterated Local Search) algorithm for solving the Capacitated Vehicle Routing Problem (CVRP).
based on meta heuristics such as Tabu Search, Ant Colony Optimization (ACO), and Simulated Annealing."""

# -------------------Imports-------------------------------
from utils import *  # from utils import functions
from MSH_Ackley import msh_ackley_solver
import time

LOWER_BOUND = -32.768
UPPER_BOUND = 32.768
DIM = 10


def f(x):
    # Ackley function value
    a, b, c = 20, 0.2, 2 * np.pi
    d = len(x)
    sum1 = np.sum(x ** 2)
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
                idx = random.randint(0, dim - 1)
                perturbation = np.random.uniform(-adaptive_step, adaptive_step)
                neighbor[idx] += perturbation
            case "multi_dim":
                indices = random.sample(range(dim), random.randint(2, dim // 2))
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
                idx = random.randint(0, dim - 1)
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


# -------------------- Simulated Annealing Functions -------------------------------

def update_temperature(T, alpha, iteration, max_iter, reheating=False, Tmin=1e-3):
    if reheating:
        return T * 3.0
    else:
        return max(T * alpha, Tmin)


def metropolis_step(T, s):
    neighbors = N(s)
    legal_neighbors = [n for n in neighbors if L(n)]
    if not legal_neighbors:
        return s
    n = random.choice(legal_neighbors)
    delta_f = f(s) - f(n)
    if delta_f > 0:
        return n
    else:
        exponent = -delta_f / T if T > 0 else 0
        exponent = np.clip(exponent, -700, 700)
        probability = np.exp(exponent)
        if random.random() < probability:
            return n
    return s


def calculate_initial_temperature(s, dim=10, lower_bound=LOWER_BOUND, upper_bound=UPPER_BOUND):
    cost_diffs = []
    current_cost = f(s)
    for _ in range(10):
        neighbors = N(s, dim, lower_bound, upper_bound)
        legal_neighbors = [n for n in neighbors if L(n, lower_bound, upper_bound)]
        if legal_neighbors:
            neighbor = random.choice(legal_neighbors)
            neighbor_cost = f(neighbor)
            cost_diff = neighbor_cost - current_cost
            if cost_diff > 0:
                cost_diffs.append(cost_diff)
    if cost_diffs:
        avg_diff = sum(cost_diffs) / len(cost_diffs)
        return avg_diff / 0.693
    else:
        return 10.0


# -------------------- Ant Colony Optimization (ACO) Functions Ackley -------------------------------
def ant_solution(pheromone, heuristic, dim, lower_bound, upper_bound, alpha=1.0, beta=2.0):
    """Generate a solution for one ant using pheromone and heuristic information for Ackley function."""

    num_bins = pheromone.shape[1]
    solution = np.zeros(dim)

    # Generate value for each dimension
    for d in range(dim):
        # Calculate probabilities for selecting each bin
        probabilities = {}
        denominator = 0

        # First calculate denominator sum
        for bin_idx in range(num_bins):
            tau = pheromone[d][bin_idx]
            eta = heuristic[d][bin_idx]
            denominator += (tau ** alpha) * (eta ** beta)

        # Calculate probabilities for each bin
        for bin_idx in range(num_bins):
            tau = pheromone[d][bin_idx]
            eta = heuristic[d][bin_idx]
            probabilities[bin_idx] = ((tau ** alpha) * (
                        eta ** beta)) / denominator if denominator > 0 else 1.0 / num_bins

        # Select bin using roulette wheel selection
        selected_bin = roulette_wheel_selection(probabilities)

        # Convert bin to continuous value
        bin_size = (upper_bound - lower_bound) / num_bins
        min_val = lower_bound + selected_bin * bin_size
        max_val = min_val + bin_size

        # Set value with some noise
        solution[d] = random.uniform(min_val, max_val)

    # Add diversity-enhancing mutation (equivalent to route shuffling in CVRP)
    if random.random() < 0.15:  # 15% chance for diversity-enhancing mutation
        # Apply a small perturbation to encourage diversity
        dims_to_modify = random.randint(1, max(1, dim // 2))
        indices = random.sample(range(dim), dims_to_modify)

        for idx in indices:
            # Apply a larger perturbation to selected dimensions
            perturbation = np.random.normal(0, (upper_bound - lower_bound) * 0.1)
            solution[idx] += perturbation

        # Ensure solution stays within bounds
        solution = np.clip(solution, lower_bound, upper_bound)

    return solution


def roulette_wheel_selection(probabilities):
    """
    Select a bin using roulette wheel selection based on probabilities
    """
    rand_val = random.random()
    cumulative_prob = 0

    for bin_idx, prob in probabilities.items():
        cumulative_prob += prob
        if rand_val <= cumulative_prob:
            return bin_idx

    # Fallback: return random bin if something goes wrong
    return random.choice(list(probabilities.keys()))


def update_pheromone(pheromone, solutions, dim, lower_bound, upper_bound, rho=0.1, Q=1.0):
    """Update pheromone trails based on the ants' solutions for Ackley function."""
    num_bins = pheromone.shape[1]
    bin_size = (upper_bound - lower_bound) / num_bins

    # Calculate delta tau for all bins
    delta_tau = np.zeros_like(pheromone)
    for solution in solutions:
        solution_quality = f(solution)
        if solution_quality > 0:  # Avoid division by zero
            solution_deposit = Q / solution_quality  # Better solutions get more pheromone

            # Deposit pheromone in bins corresponding to solution values
            for d in range(dim):
                val = solution[d]
                bin_idx = int((val - lower_bound) / bin_size)
                bin_idx = max(0, min(bin_idx, num_bins - 1))  # Ensure it's in bounds
                delta_tau[d][bin_idx] += solution_deposit

    # Update pheromone using the formula τij(t+1) = (1-ρ)τij(t) + ρΔτij(t)
    pheromone = (1 - rho) * pheromone + rho * delta_tau

    # Ensure minimum pheromone level
    pheromone = np.maximum(pheromone, 0.01)  # Avoid zero pheromone levels

    return pheromone


def aco_main_ackley(inital_solution, max_searches, params, dim=10, upper_bound=UPPER_BOUND, lower_bound=LOWER_BOUND):
    """Main function for Ant Colony Optimization (ACO) for Ackley function."""
    # Parameters for ACO
    nAnts = 40  # Number of ants
    num_bins = 20  # Number of bins per dimension
    nEpochs = max_searches // nAnts  # Number of epochs
    # ACO parameters
    # Extract parameters from params dictionary
    alpha = params.get('alpha', 1.0)  # Pheromone importance
    beta = params.get('beta', 2.0)  # Heuristic importance
    rho = params.get('rho', 0.1)  # Pheromone evaporation rate
    Q = params.get('Q', 1.0)  # Pheromone deposit factor

    # Initialize pheromone matrix
    pheromone = np.ones((dim, num_bins)) * 0.1  # Initial pheromone levels

    # Initialize heuristic information (closeness to origin is better for Ackley)
    heuristic = np.zeros((dim, num_bins))
    for d in range(dim):
        for b in range(num_bins):
            bin_center = lower_bound + (b + 0.5) * ((upper_bound - lower_bound) / num_bins)
            # Closer to 0 is better for Ackley
            heuristic[d][b] = 1.0 / (1.0 + abs(bin_center))

    # Initial population of ants
    ants = []
    for _ in range(nAnts):
        # s = generate_initial_solution(dim, lower_bound, upper_bound)
        s = generate_new_solution(inital_solution, dim, lower_bound, upper_bound)
        s = LocalSearchStep(s)  # Improve initial solution
        ants.append(s)

    # Evaluate initial population
    best_solution = min(ants, key=lambda x: f(x))  # Best solution found by ants
    best_value = f(best_solution)  # Value of the best solution

    # Main ACO loop
    iteration_without_improvement = 0  # Track stagnation
    for epoch in range(nEpochs):
        new_ants = []

        # Construct new ant population solutions
        for ant in range(nAnts):
            ant_s = ant_solution(pheromone, heuristic, dim, lower_bound, upper_bound, alpha, beta)
            # Introduce randomness in path selection
            if random.random() < 0.2:  # 20% chance to select a random path
                ant_s = generate_new_solution(ant_s, dim, lower_bound, upper_bound)

            if L(ant_s):
                # Apply local search to improve solution
                ant_s = LocalSearchStep(ant_s)
                new_ants.append(ant_s)

        if not new_ants:
            new_ants = ants  # If no new solutions, keep the old ones

        # Encourage ants to explore new paths by using memory
        if epoch % 5 == 0:  # Every 5 epochs, encourage exploration
            for i in range(len(new_ants)):
                new_ants[i] = generate_new_solution(new_ants[i], dim, lower_bound, upper_bound)

        # Randomly reset pheromone trails
        if random.random() < 0.2:  # 20% chance to reset pheromone trails
            # Reset only partially - keep some memory
            reset_factor = 0.2 + 0.3 * (iteration_without_improvement / 20)  # 0.2-0.5 based on stagnation
            pheromone = pheromone * (1 - reset_factor) + np.ones_like(pheromone) * 0.1 * reset_factor

        # Implement pheromone smoothing to level path differences
        if epoch % 3 == 0:  # Apply smoothing every 3 epochs for better performance
            # For Ackley function, normalize across bins for each dimension
            for d in range(dim):
                row_sum = np.sum(pheromone[d])
                if row_sum > 0:
                    pheromone[d] = pheromone[d] / row_sum
            pheromone = np.clip(pheromone, 0.1, None)  # Ensure minimum pheromone level

        # Update pheromone trails
        pheromone = update_pheromone(pheromone, new_ants, dim, lower_bound, upper_bound, rho, Q)

        # Update global best ant solution
        current_best = min(new_ants, key=lambda x: f(x))
        current_best_value = f(current_best)

        if current_best_value < best_value:
            best_solution = np.copy(current_best)
            best_value = current_best_value

            # Exploitation phase - focus more on the pheromone
            alpha = min(3.0, alpha * 1.1)  # Increase pheromone importance
            beta = max(1.0, beta * 0.95)  # Decrease heuristic importance
            iteration_without_improvement = 0
        else:
            iteration_without_improvement += 1
            # Exploration phase - focus more on the heuristic
            alpha = max(0.5, alpha * 0.9)  # Reduce pheromone importance
            beta = min(5.0, beta * 1.1)  # Increase heuristic importance

        ants = new_ants  # Update ants for the next epoch

    return best_solution


# -------------------- Tabu Search Functions-------------------------------
def Local_search_tabu(s, tabu, n):
    max_trials = 10
    new_s = np.copy(s)
    current_cost = f(new_s)
    for _ in range(max_trials):
        neighbors = N(new_s)
        non_tabu_neighbors = []
        for neighbor in neighbors:
            if not L(neighbor):
                continue
            neighbor_key = solution_to_region_key(neighbor)
            neighbor_cost = f(neighbor)
            if neighbor_key in tabu:
                if neighbor_cost < current_cost:
                    non_tabu_neighbors.append(neighbor)
            else:
                non_tabu_neighbors.append(neighbor)
        if not non_tabu_neighbors:
            neighbors = [n for n in neighbors if L(n)]
            s_prime = S(neighbors) if neighbors else None
        else:
            neighbors = non_tabu_neighbors
            s_prime = S(neighbors) if neighbors else None
        if s_prime is not None and f(s_prime) < f(new_s):
            new_s = np.copy(s_prime)
            new_cost = f(new_s)
            solution_key = solution_to_region_key(s_prime)
            delta_cost = current_cost - new_cost
            tabu[solution_key] = {
                'cost': new_cost,
                'delta_cost': delta_cost,
                'tenure': n,
                'generation': 0,
                'solution': np.copy(s_prime)
            }
            current_cost = new_cost
    return new_s, tabu


def solution_to_region_key(solution, region_size=0.05):
    current_value = f(solution)
    if current_value < 5.0:
        effective_region_size = region_size * 0.2
    elif current_value < 10.0:
        effective_region_size = region_size * 0.5
    else:
        effective_region_size = region_size
    return tuple(np.round(solution / effective_region_size) * effective_region_size)


def tabu_search(time_limit, max_searches, dim=10, lower_bound=LOWER_BOUND, upper_bound=UPPER_BOUND):
    start_time = time.time()

    s = generate_initial_solution(dim, lower_bound, upper_bound)
    MSH_s, _ = msh_ackley_solver(dim, lower_bound, upper_bound, num_clusters=8)
    if f(s) > f(MSH_s) and L(MSH_s):
        s = MSH_s
    best_solution = LocalSearchStep(s)
    best_cost = f(best_solution)

    tabu = {}
    # Scale tabu tenure with problem dimension
    n = min(max(8, int(dim * 1.2)), 25)
    solution_key = solution_to_region_key(best_solution)
    tabu[solution_key] = {
        'cost': best_cost,
        'delta_cost': f(s) - best_cost,
        'tenure': n,
        'generation': 0,
        'solution': np.copy(best_solution)
    }
    iteration_without_improvement = 0
    solutions = [best_cost]
    for iteration in range(max_searches):
        s, tabu = Local_search_tabu(s, tabu, n)
        if L(s):
            current_cost = f(s)
            solutions.append(current_cost)
            if current_cost < best_cost:
                best_solution = np.copy(s)
                best_cost = current_cost
                iteration_without_improvement = 0
                n = min(20, n + 2)
                for key in tabu:
                    tabu[key]['tenure'] = n
            else:
                iteration_without_improvement += 1
        found = False
        if iteration_without_improvement > 10:
            max_attempts = 10
            make_a_choice = ["Intensification", "Diversification"]
            choice = random.choice(make_a_choice)
            choice = "Diversification" if not tabu else choice
            if choice == "Intensification":
                new_key = min(tabu.keys(), key=lambda k: tabu[k]['cost'])
                new_s = np.array(new_key)
                s = new_s
                if random.random() < 0.5:
                    s = best_solution
            else:
                n = 10  # Reset tenure for diversification
                for _ in range(max_attempts):
                    new_s = generate_new_solution(s, dim, lower_bound, upper_bound)
                    new_key = solution_to_region_key(new_s)
                    if new_key not in tabu:
                        found = True
                        s = new_s
                        break
                if not found:
                    s = new_s
            n = max(1, n - 1)
            for key in tabu:
                tabu[key]['tenure'] = n
        keys_to_remove = []
        for key in tabu:
            tabu[key]['generation'] += 1
            if tabu[key]['tenure'] <= tabu[key]['generation']:
                keys_to_remove.append(key)
        for key in keys_to_remove:
            tabu.pop(key)

        if time.time() - start_time > time_limit:
            print("Time limit exceeded.")
            break
    return best_solution, solutions


def aco(time_limit, max_searches, dim=10, lower_bound=LOWER_BOUND, upper_bound=UPPER_BOUND):
    start_time = time.time()
    """current_solution = generate_initial_solution(dim, lower_bound, upper_bound)
    current_solution = LocalSearchStep(current_solution)"""
    s = generate_initial_solution(dim, lower_bound, upper_bound)
    MSH_s, _ = msh_ackley_solver(dim, lower_bound, upper_bound, num_clusters=8)
    if f(s) > f(MSH_s) and L(MSH_s):
        s = MSH_s
    best_solution = LocalSearchStep(s)
    best_cost = f(best_solution)
    solutions = [best_cost]
    alpha = 1.0
    beta = 2.0
    rho = 0.1
    Q = 1.0
    iterations_without_improvement = 0
    max_iterations_without_improvement = 1
    remaining_searches = max_searches
    for _ in range(10):
        """if iterations_without_improvement > max_iterations_without_improvement:
            alpha = max(0.5, alpha * 0.9)
            beta = min(5.0, beta * 1.1)
            rho = min(0.5, rho * 1.1)
            Q = max(0.1, Q * 0.9)
            iterations_without_improvement = 0"""
        aco_params = {
            'alpha': alpha,
            'beta': beta,
            'rho': rho,
            'Q': Q
        }
        perturbed_solution = aco_main_ackley(best_solution, max_searches // 10, aco_params)
        for _ in range(5):
            improved_solution = LocalSearchStep(perturbed_solution)
            perturbed_solution = np.copy(improved_solution)

        if L(improved_solution):
            current_cost = f(improved_solution)
            solutions.append(current_cost)
            if current_cost < best_cost:
                best_solution = np.copy(improved_solution)
                best_cost = current_cost
                current_solution = improved_solution
                iterations_without_improvement = 0
                """alpha = min(3.0, alpha * 1.1)
                beta = max(1.0, beta * 0.95)
                rho = max(0.05, rho * 0.9)
                Q = min(2.0, Q * 1.05)"""
            else:
                iterations_without_improvement += 1
        remaining_searches -= 1
        if time.time() - start_time > time_limit:
            print("Time limit exceeded.")
            break
    return best_solution, solutions


def simulated_annealing(time_limit, max_searches, dim=10, lower_bound=LOWER_BOUND, upper_bound=UPPER_BOUND):
    start_time = time.time()

    use_msh = False
    s = generate_initial_solution(dim, lower_bound, upper_bound)
    MSH_s, _ = msh_ackley_solver(dim, lower_bound, upper_bound, num_clusters=8)
    if f(s) > f(MSH_s) and L(MSH_s):
        s = MSH_s
        use_msh = True
    best_solution = LocalSearchStep(s)

    T = calculate_initial_temperature(s, dim, lower_bound, upper_bound)
    alpha = 0.95
    iterations_without_improvement = 0

    solutions = [f(best_solution)]

    for iteration in range(max_searches):

        reheating = False

        if iterations_without_improvement > 30:
            if not use_msh:
                s = generate_initial_solution(dim, lower_bound, upper_bound)
            else:
                s, _ = msh_ackley_solver(dim, lower_bound, upper_bound, num_clusters=8)
            best_solution = LocalSearchStep(s)
            T = calculate_initial_temperature(s)
            alpha = 0.95
            iterations_without_improvement = 0

        if iterations_without_improvement > 20:
            reheating = True
            alpha = min(0.99, alpha * 1.05)

        s = metropolis_step(T, s)
        s = LocalSearchStep(s)
        if L(s):
            solutions.append(f(s))
            if f(s) < f(best_solution):
                best_solution = np.copy(s)
                iterations_without_improvement = 0
                alpha = max(0.8, alpha * 0.95)
            else:
                iterations_without_improvement += 1
                # alpha = min(0.99, alpha * 1.05)
        T = update_temperature(T, alpha, iteration, max_searches, reheating=reheating)

        if iteration % 10 == 0:
            if not use_msh:
                s = generate_initial_solution(dim, lower_bound, upper_bound)
            else:
                s, _ = msh_ackley_solver(dim, lower_bound, upper_bound, num_clusters=8)
            for _ in range(5):
                s = generate_new_solution(s, dim, lower_bound, upper_bound)
                new_s = LocalSearchStep(s)
                if f(new_s) < f(s):
                    s = new_s
        if iteration % 100 == 0:
            s = best_solution

        if time.time() - start_time > time_limit:
            print("Time limit exceeded.")
            break

    return best_solution, solutions


# -------------------Algorithm-------------------------------
def ILS_normal(dim=10, lower_bound=LOWER_BOUND, upper_bound=UPPER_BOUND, max_searches=1000):
    # ILS
    s = generate_initial_solution(dim, lower_bound, upper_bound)  # Generate initial solution
    if not L(s):
        print("Initial solution is legal.")
        print(s)
        raise ValueError("Initial solution is not legal. Please check the input data.")
    best_solution = LocalSearchStep(s)  # Perform local search on the initial solution

    for _ in range(max_searches):
        s = LocalSearchStep(s)
        if L(s):
            print("Local search found a legal solution.")
            if f(s) < f(best_solution):  # Check if the new solution is better
                print("Found a better solution.")
                best_solution = s
        s = generate_new_solution(s)
    return best_solution


def ILS_ACKLEY(meta_heuristic, dim=10, time_limit=60, max_searches=1000):
    """
    Iterated Local Search for the Ackley function.
    """

    match meta_heuristic:
        case "tabu_search":
            solution, ackley_scores = tabu_search(time_limit, max_searches, dim, LOWER_BOUND, UPPER_BOUND)
        case "aco":
            solution, ackley_scores = aco(time_limit, max_searches, dim, LOWER_BOUND, UPPER_BOUND)
        case "simulated_annealing":
            solution, ackley_scores = simulated_annealing(time_limit, max_searches, dim, LOWER_BOUND, UPPER_BOUND)
        case "parallel_hybrid":
            solution, ackley_scores = parallel_hybrid_ils_ackley(dim, LOWER_BOUND, UPPER_BOUND, time_limit,
                                                                 max_searches)
        case _:
            solution, ackley_scores = parallel_hybrid_ils_ackley(dim, LOWER_BOUND, UPPER_BOUND, time_limit,
                                                                 max_searches)

    return solution, ackley_scores  # Return the best solution and its scores


def parallel_hybrid_ils_ackley(dim=10, lower_bound=LOWER_BOUND, upper_bound=UPPER_BOUND, time_limit=300,
                               max_searches=1000, seed=None):
    """
    Parallel hybrid ILS that runs multiple meta-heuristics simultaneously for Ackley function optimization.

    Args:
        dim: Dimension of the Ackley function
        lower_bound: Lower bound for variables
        upper_bound: Upper bound for variables
        time_limit: Maximum time in seconds
        max_searches: Maximum number of search iterations
        seed: Random seed for reproducibility

    Returns:
        best_solution: The best solution found
        scores: List of best scores over time
    """
    import concurrent.futures

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    start_time = time.time()
    start_cpu = time.process_time()

    # Initialize solution
    initial_solution = generate_initial_solution(dim, lower_bound, upper_bound)
    initial_solution = LocalSearchStep(initial_solution)
    best_solution = np.copy(initial_solution)
    best_value = f(best_solution)

    # Track scores over time
    scores = [best_value]

    # Define algorithm functions with consistent interface
    algorithms = {
        "simulated_annealing": simulated_annealing,
        "aco": aco,
        "tabu_search": tabu_search
    }

    # Divide resources among algorithms
    algo_time_limit = time_limit
    algo_searches = max_searches

    # Run algorithms in parallel
    results = []
    metrics = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(
                algo_func, algo_time_limit, algo_searches, dim, lower_bound, upper_bound
            ): algo_name
            for algo_name, algo_func in algorithms.items()
        }

        for future in concurrent.futures.as_completed(futures):
            algo_name = futures[future]
            try:
                print(f"Running {algo_name} with time limit {algo_time_limit} seconds and max searches {algo_searches}")
                solution, solutions_alg = future.result()  # Your functions return solution and scores
                if L(solution):
                    value = f(solution)
                    results.append((solution, value, algo_name))
                    print(f"{algo_name} found solution with value {value}")
                    scores.append(value)
                    metrics[algo_name] = solutions_alg

                run_time = time.time() - start_time
                cpu_time = time.process_time() - start_cpu
                print(
                    f"{algo_name} completed successfully.  Elapsed  time: {run_time} | CPU time: {cpu_time} | remaining time: {time_limit - (run_time):.1f} seconds")
            except Exception as e:
                print(f"{algo_name} generated an exception: {e}")

    # Find best solution from parallel runs
    if results:
        results.sort(key=lambda x: x[1])  # Sort by value (lower is better for Ackley)
        best_parallel_solution, best_parallel_value, best_algo = results[0]

        if best_parallel_value < best_value:
            best_solution = np.copy(best_parallel_solution)
            best_value = best_parallel_value
            print(f"Best solution found by {best_algo} with value {best_value}")
            scores.append(best_value)

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
            improved = LocalSearchStep(solution)
            if L(improved):
                improved_value = f(improved)
                if improved_value < best_value:
                    best_solution = np.copy(improved)
                    best_value = improved_value
                    scores.append(best_value)
                    print(f"Hybrid optimization improved solution to {best_value}")

    print(f"CPU time: {time.process_time() - start_cpu:.2f} seconds")
    print(f"Elapsed time: {time.time() - start_time:.2f} seconds")
    # Print final best value
    print(f"Final best solution value: {best_value}")

    return best_solution, metrics


def ils_ackley_solver(dim=10, lower=-32.768, upper=32.768, meta_heuristic="simulated_annealing",
                      time_limit=60, max_searches=1000, seed=None):
    """
    Wrapper function for ILS to match the interface expected by ackley_solver.py

    Parameters:
    - dim: Dimension of the Ackley function
    - lower: Lower bound for variables
    - upper: Upper bound for variables
    - meta_heuristic: Which metaheuristic to use ("simulated_annealing", "tabu_search", "aco",
                                                 "hybrid", or "parallel")
    - time_limit: Maximum time in seconds
    - time_limit: Maximum time in seconds
    - max_searches: Maximum number of search iterations
    - seed: Random seed for reproducibility

    Returns:
    - solution: The best solution found
    - value: The objective function value of the best solution
    """
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    # Run ILS with the specified metaheuristic
    solution, metrics = ILS_ACKLEY(meta_heuristic, dim, time_limit, max_searches)

    # Calculate the final objective value
    value = f(solution)

    return solution, value, metrics