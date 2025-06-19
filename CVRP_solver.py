"""Capacitated Vehicle Routing Problem (CVRP) solver"""

"""minimize the total distance traveled by a fleet of vehicles so that each city is visited 
exavcly once and the total demand of each vehicle does not exceed its capacity."""

#------------------- Imports -------------------------------------
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use 'Agg' backend for non-interactive plotting
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sys
import time
import random
import matplotlib.patheffects as PathEffects

# Importing algorithms
from MSH import MSH_alg # from MSH import MSH algorithm
from ILS import parallel_hybrid_ils  , ILS   # from ILS import ILS algorithm
from plots.plots import *


DISPLAY_PLOTS = False  # or False if you don't want to show plots interactively

#-------------------Input \ Output -------------------------------

# CVRP problem extraction from a file
def parse_vrp_file(filepath):
    with open(filepath, 'r') as file:
        lines = file.readlines()

    num_vehicles = None
    capacity = None
    dimension = None
    node_coords = {}
    demands = {}
    section = None

    for line in lines:
        line = line.strip()
        if line.startswith("COMMENT"):
            if "Min no of trucks:" in line:
                num_vehicles = int(line.split("Min no of trucks:")[1].split(",")[0].strip())
        elif line.startswith("CAPACITY"):
            capacity = int(line.split(":")[1].strip())
        elif line.startswith("DIMENSION"):
            dimension = int(line.split(":")[1].strip())
        elif line == "NODE_COORD_SECTION":
            section = "NODE"
            continue
        elif line == "DEMAND_SECTION":
            section = "DEMAND"
            continue
        elif line == "DEPOT_SECTION" or line == "EOF":
            section = None
            continue

        if section == "NODE":
            parts = line.split()
            if len(parts) == 3:
                node_id = int(parts[0])
                node_coords[node_id] = (int(parts[1]), int(parts[2]))
        elif section == "DEMAND":
            parts = line.split()
            if len(parts) == 2:
                node_id = int(parts[0])
                demands[node_id] = int(parts[1])

    return {
        "num_vehicles": num_vehicles,
        "capacity": capacity,
        "num_nodes": dimension,
        "coordinates": node_coords,
        "demands": demands
    }

def parse_solution_file(filepath):
    """
    Parses a CVRP solution file and returns the routes and total cost.
    Returns:
        routes: list of lists, each list is a route (list of node ids)
        cost: int or float, total cost
    """
    routes = []
    cost = None
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("Route #"):
                # Example: Route #1: 17 20 18 15 12
                parts = line.split(":", 1)
                if len(parts) == 2:
                    route_nodes = [int(x) for x in parts[1].strip().split()]
                    routes.append(route_nodes)
            elif line.startswith("Cost"):
                # Example: Cost 375
                cost = float(line.split()[1])
    return routes, cost

# print solution
def print_cvrp_solution(routes, total_cost):
    for i, route in enumerate(routes, 1):
        route_str = " ".join(str(node) for node in route)
        print(f"Route #{i}: {route_str}")
    print(f"Cost {total_cost}")
def plot_cvrp_solution(coords, routes, total_cost):
    """
    coords: list of (x, y) tuples, index = node id (0 = depot)
    routes: list of lists, each list is a vehicle route including depot, e.g., [0, 2, 3, 0]
    total_cost: total route cost
    """
    plt.figure(figsize=(8, 6))

    # Plot nodes
    for idx, (x, y) in coords.items():
        plt.plot(x, y, 'ko', markersize=6)
        plt.text(x + 1.5, y + 1.5, str(idx), fontsize=9)
        if idx == 1:
            plt.text(x + 1.5, y + 1.5, "Depot", fontsize=9, color='red')

    # Use colormap for route colors
    num_routes = len(routes)
    cmap = plt.colormaps.get_cmap('tab20')
    colors = [cmap(i / num_routes) for i in range(num_routes)]

    #colors = ['green', 'purple', 'blue', 'orange', 'red', 'cyan']

    for i, route in enumerate(routes):
        #color = colors[i % len(colors)]
        color = colors[i]
        for j in range(len(route) - 1):
            a, b = route[j], route[j + 1]
            x1, y1 = coords[a]
            x2, y2 = coords[b]
            plt.plot([x1, x2], [y1, y2], color=color, linewidth=2)
            plt.arrow(x1, y1, (x2 - x1) * 0.8, (y2 - y1) * 0.8, head_width=2, head_length=2, fc=color, ec=color)

    plt.title(f'CVRP Solution : Cost = {total_cost:.2f}', fontsize=14)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.axis('equal')
    plt.show()


def plot_routes_side_by_side(coords, solution_routes, optimal_routes, solution_cost, optimal_cost, depot_id=1, save_path=None):
    """
    Plots the solution routes and optimal routes side by side for comparison.
    coords: dict of node_id -> (x, y)
    solution_routes: list of lists of node ids (your algorithm's solution)
    optimal_routes: list of lists of node ids (optimal/reference solution)
    solution_cost: float, cost of your solution
    optimal_cost: float, cost of the optimal/reference solution
    depot_id: int, id of the depot node (default 1)
    save_path: str or None, if provided saves the plot to this path
    """
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as PathEffects
    cmap = plt.get_cmap('tab20')
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    def _draw(ax, routes, cost, label):
        for i, route in enumerate(routes, 1):
            color = cmap(i % 20)
            xs = [coords[n][0] for n in route]
            ys = [coords[n][1] for n in route]
            ax.plot(xs, ys, '-o', color=color, linewidth=2.5, markeredgecolor='k', label=f"{label} r{i}")
            for n in route:
                x, y = coords[n]
                txt = ax.text(x, y, str(n), fontsize=9, ha='center', va='center', color='black', zorder=6)
                txt.set_path_effects([
                    PathEffects.Stroke(linewidth=2, foreground='white'),
                    PathEffects.Normal()
                ])
        dx, dy = coords[depot_id]
        ax.scatter([dx], [dy], marker='*', s=140, c='red', edgecolor='k', label='depot')
        ax.set_title(f"{label} cost={cost:.1f}")
        ax.set_aspect('equal')
        ax.grid(True, ls=':')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.legend(fontsize=8, loc='upper right')

    _draw(axes[0], optimal_routes, optimal_cost, 'Optimal')
    _draw(axes[1], solution_routes, solution_cost, 'Solution')
    plt.suptitle('CVRP Solution vs Optimal', fontsize=15)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_routes(reference_data: dict,
                    solution_data:  dict,
                    filename:      str,
                    title:         str) -> None:
        """
        quick fallback for side‐by‐side routes: left=suboptimal, right=algo.
        uses same aesthetics as main but in two axes.
        """
        coords      = reference_data['coords']
        demands     = reference_data['demands']
        capacity    = reference_data['capacity']
        fig, axes   = plt.subplots(1, 2, figsize=(18, 8))
        cmap        = plt.get_cmap('tab20')

        def _draw(ax, routes_list, cost_val, label_text):
            # draw each route as colored line+markers
            for idx, route in enumerate(routes_list, start=1):
                col = cmap(idx % 20)
                xs  = [coords[node][0] for node in route]
                ys  = [coords[node][1] for node in route]
                ax.plot(xs, ys, '-o', color=col, linewidth=3.0,
                        alpha=0.9, markeredgecolor='k', label=f'{label_text} r{idx}')
                # label each node number
                for node in route:
                    x_node, y_node = coords[node]
                    txt = ax.text(x_node, y_node, str(node),
                                  fontsize=10, ha='center', va='center', color='black',
                                  zorder=6)
                    txt.set_path_effects([
                        PathEffects.Stroke(linewidth=2, foreground='white'),
                        PathEffects.Normal()
                    ])
            # draw depot if present
            if 1 in coords:
                dx, dy = coords[1]
                ax.scatter([dx],[dy], s=140, marker='*', c='red',
                           edgecolor='k', label='depot', zorder=7)
            ax.set_title(f"{label_text} cost={cost_val:.1f}", fontsize=12)
            ax.set_aspect('equal')
            ax.grid(True, ls=':')
            ax.set_xlabel('x'); ax.set_ylabel('y')
            ax.legend(fontsize=8, loc='upper right')

        _draw(axes[0], reference_data['routes'], reference_data['cost'], 'reference')
        _draw(axes[1], solution_data['routes'],   solution_data['cost'],   'solution')
        plt.suptitle(title, fontsize=14)
        if filename:
            plt.savefig(filename, dpi=150, bbox_inches='tight')
        if DISPLAY_PLOTS:
            plt.show()
        plt.close()
#-------------------Functions-------------------------------
# Distance function
def distance_function(x1, y1, x2, y2):
    """Calculate the Euclidean distance between two points."""
    return np.sqrt((x1-x2)**2 + (y1-y2)**2)

# Ackley function
def ackley_function(x):
    """Find global minimum of the Ackley function with the algorithms implemented in this CVRP solver."""
    a = 20
    b = 0.2
    c = 2 * np.pi
    d = len(x) #10

    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(c * x))

    term1 = -a * np.exp(-b * np.sqrt(sum1 / d))
    term2 = -np.exp(sum2 / d)

    return term1 + term2 + a + np.exp(1)

#-------------------Main function-------------------------------
# Main function to choose the algorithm
def main():
    """
    Choose an algorithm to use for solving the CVRP problem:

    1. MSH (Multi-Stage Heuristic)
    2. ILS (Iterated Local Search):
        a. Tabu Search
        b. Discerte Ant Colony Optimization - ACO
        c. Simulated Annealing
    3. GA (Genetic Algorithm) + Island Model
    4. ALNS (Adaptive Large Neighborhood Search)
    5. Branch and Bound (B&B) based on LDS heuristic

    """
    random.seed(time.time())

    # Choose the algorithm to use
    algorithms = ["MSH","ILS","GA","ALNS","B&B"]
    algorithm_deafult = "ILS"
    meta_heuristic = "tabu_search"  # Default metaheuristic for ILS

    if len(sys.argv) <= 1:
        algorithm = algorithm_deafult
        time_limit = 600  # Default time limit in seconds
    else:
        algorithm = sys.argv[1]
        time_limit = sys.argv[2] if len(sys.argv) > 2 else 60
        time_limit = int(time_limit)  # Convert time to integer
        if algorithm == "ILS" and len(sys.argv) > 3:
            meta_heuristic = sys.argv[3]



    # CVRP problem extraction from a file
    problems = {
        'beginner': ["E-n22-k4", "P-n16-k8"],
        'intermediate': ["A-n32-k5", "A-n80-k10"],
        'advanced': ["M-n200-k17","X-n101-k25"]
    }

    algorithm = "ILS"  # Default algorithm

    problem_set = 'beginner'  # Default problem set

    for i, problem in enumerate(problems[problem_set]):




        print(f"{i+1}. {problem}")

        filepath = f"problems/{problem_set}/{problem}.vrp"
        data = parse_vrp_file(filepath)

        # CVRP parameters
        num_vehicles = data["num_vehicles"]
        capacity = data["capacity"]
        num_nodes = data["num_nodes"]
        coordinates = data["coordinates"]
        demands = data["demands"]
        print("num_nodes:", num_nodes)
        filepath = f"problems/{problem_set}/{problem}.sol"
        data_solution = parse_solution_file(filepath)
        routes_solution, total_cost_solution = data_solution


        # max searches and time limit for each problem set
        if problem_set == 'beginner':
            max_searches = 3000  # Default number of searches for ILS and MSH
            time_limit = 800  # Default time limit in seconds
        elif problem_set == 'intermediate':
            max_searches = 9000
            time_limit = 1800 #1200  # Default time limit in seconds
        elif problem_set == 'advanced':
            max_searches = 20000
            time_limit = 3600

        # paths
        import os
        plot_dir = f"plots/{problem}_{algorithm}"
        os.makedirs(plot_dir, exist_ok=True)  # Create directory for this problem/algorithm
        save_path_routes = f"{plot_dir}/routes.png"
        save_path_performance = f"{plot_dir}/performance.png"
        save_path_parameters_aco = f"{plot_dir}/aco_parameters.png"
        save_path_parameters_sa = f"{plot_dir}/sa_parameters.png"
        save_path_parameters_ts = f"{plot_dir}/ts_parameters.png"
        save_path_aco_stats = f"{plot_dir}/aco_stats.png"
        save_path_comparison = f"{plot_dir}/comparison.png"

        # run the selected algorithm
        start_time = time.time()
        if algorithm == "MSH":
            routes_alg, total_cost = MSH_alg(coordinates, demands, capacity, num_nodes)
            from utils import validate_routes_capacity
            if not validate_routes_capacity(routes_alg,coordinates ,demands, capacity):
                print("Error: Routes do not satisfy capacity constraints.")
            depot = 1

            # Get all cities (excluding depot)
            all_cities = set(coordinates.keys())
            all_cities.remove(depot)

            # Track visited cities
            visited_cities = set()

            # Check for duplicates
            for route in routes_alg:
                for city in route:
                    if city != depot:  # Ignore depot
                        if city in visited_cities:
                            print(f"Error: City {city} is visited more than once.")
                        visited_cities.add(city)

            # Check if all cities are visited
            if visited_cities != all_cities:
                missing = all_cities - visited_cities
                print(f"Error: Not all cities are visited. Missing cities: {missing}")
            for route in routes_alg:
                if route[0] != depot or route[-1] != depot:
                    print(f"Error: Route {route} does not start and end with the depot.")
                if len(route) < 2:
                    print(f"Error: Route {route} has less than 2 nodes (depot + at least one city).")
        elif algorithm == "aco":
            start_time = time.time()
            routes_alg, total_cost, aco_metrics = ILS(coordinates, demands, capacity, num_nodes,"aco" ,time_limit, max_searches)
            end_time = time.time()
            print(f"Time taken for ACO: {end_time - start_time:.2f} seconds")
            """start_time = time.time()
            routes_alg ,total_cost, sa_metrics = ILS(coordinates, demands, capacity, num_nodes,"simulated_annealing" ,time_limit, max_searches)
            end_time = time.time()
            print(f"Time taken for Simulated Annealing: {end_time - start_time:.2f} seconds")"""
        elif algorithm =="tabu_search":
            start_time = time.time()
            routes_alg, total_cost, ts_metrics = ILS(coordinates, demands, capacity, num_nodes,"tabu_search" ,time_limit, max_searches)
            end_time = time.time()
            print(f"Time taken for Tabu Search: {end_time - start_time:.2f} seconds")
        else:
            routes_alg, total_cost, metrics_dict = parallel_hybrid_ils(coordinates, demands, capacity, num_nodes, time_limit, max_searches)
            routes_alg, total_cost, aco_metrics = ILS(coordinates, demands, capacity, num_nodes,"aco" ,time_limit, max_searches)
            routes_alg, total_cost, sa_metrics = ILS(coordinates, demands, capacity, num_nodes,"simulated_annealing" ,time_limit, max_searches)
            routes_alg, total_cost, ts_metrics = ILS(coordinates, demands, capacity, num_nodes,"tabu_search" ,time_limit, max_searches)
        end_time = time.time()
        
        if algorithm == "ILS":
            best_costs = metrics_dict.get("best_costs")  # overall best costs from all algorithms
        
        # Print the results
        print(f"Algorithm: {algorithm}, Metaheuristic: ACO + Simulated Annealing + Tabu Search || max_searches: {max_searches}, time_limit: {time_limit} seconds")
        print(f"Time taken: {end_time - start_time:.2f} seconds")
        if algorithm == "ILS":
            print("Best cost per algorithm:")
            for algo, cost in best_costs.items():
                print(f"{algo}: {cost}")
        print(f"Total cost: {total_cost}")
        print(f"Optimal cost: {total_cost_solution} || Difference: {total_cost - total_cost_solution:.2f}")
        
        # plot the solution routes vs optimal routes
        print("Solution routes vs optimal routes....")
        reference_data = {
            'coords': coordinates,
            'routes': routes_solution,
            'cost': total_cost_solution,
            'demands': demands,
            'capacity': capacity
        }
        solution_data = {
            'coords': coordinates,
            'routes': routes_alg,
            'cost': total_cost,
            'demands': demands,
            'capacity': capacity
        }
        plot_routes(reference_data, solution_data, filename=save_path_routes, title="Routes Comparison")
        
        if algorithm == "ILS":
            # plots metrics
            aco_metrics =  metrics_dict.get("aco")
            sa_metrics = metrics_dict.get("simulated_annealing")
            ts_metrics = metrics_dict.get("tabu_search")
            
            aco_best_costs = aco_metrics.get("solutions", []) if aco_metrics else []
            sa_best_costs = sa_metrics.get("best_costs", []) if sa_metrics else []
            ts_best_costs = ts_metrics.get("best_costs", []) if ts_metrics else []
            
            aco_parameters = aco_metrics.get("parameters", []) if aco_metrics else []
            sa_parameters = sa_metrics.get("parameters", []) if sa_metrics else []
            ts_parameters = ts_metrics.get("parameters", []) if ts_metrics else []
            
            aco_min = aco_metrics.get("best_costs") if aco_metrics else []
            aco_worst = aco_metrics.get("worst_costs") if aco_metrics else []
            aco_avg = aco_metrics.get("avg_costs") if aco_metrics else []
            
            
            print("Metrics plotting...")
            plot_metaheuristic_performance(aco_min, sa_best_costs, ts_best_costs, iterations=None, title="Metaheuristic Performance", ylabel="Cost", xlabel="Iteration", save_path=save_path_performance)
            #plot_metaheuristic_performance(aco_best_costs, sa_best_costs, ts_best_costs, iterations=None, title="Metaheuristic Performance", ylabel="Cost", xlabel="Iteration", save_path=save_path_performance)
            plot_parameters_ils(aco_parameters, aco_best_costs, metaheuristic="aco", save_path=save_path_parameters_aco)
            plot_parameters_ils(sa_parameters, sa_best_costs, metaheuristic="simulated_annealing", save_path=save_path_parameters_sa)
            plot_parameters_ils(ts_parameters, ts_best_costs, metaheuristic="tabu_search", save_path=save_path_parameters_ts)
            plot_aco_line_stats(aco_worst, aco_avg, aco_min, save_path=save_path_aco_stats)
            plot_comparison(total_cost_solution, list(best_costs.values()), iterations=len(best_costs), algorithm_name="ILS", save_path=save_path_comparison)
        
        if algorithm == "aco":
            # plot ACO metrics
            aco_best_costs = aco_metrics.get("best_costs", [])
            aco_worst_costs = aco_metrics.get("worst_costs", [])
            aco_avg_costs = aco_metrics.get("avg_costs", [])
            aco_parameters = aco_metrics.get("parameters", [])
            
            print("Plotting ACO metrics...")
            plot_aco_line_stats(aco_worst_costs, aco_avg_costs, aco_best_costs, save_path=save_path_aco_stats)
            plot_parameters_ils(aco_parameters, aco_best_costs, metaheuristic="aco", save_path=save_path_parameters_aco)
            plot_comparison([total_cost_solution] * len(aco_best_costs), aco_best_costs, iterations=len(aco_best_costs), algorithm_name="ACO", save_path=save_path_comparison)

            
        #print to file
        print("Writing solution to file...")
        with open(f"{plot_dir}/solution.txt", "w") as f:
            f.write("------------------------------------------------------\n")
            f.write(f"Problem: {problem}\n")
            f.write(f"Algorithm: {algorithm}, Metaheuristic: ACO + Simulated Annealing + Tabu Search || max_searches: {max_searches}, time_limit: {time_limit} seconds\n")
            if algorithm == "ILS":
                f.write("Best cost per metahuristic:")
                for algo, cost in best_costs.items():
                    f.write(f"| {algo}: {cost} ")
            f.write("\n")
            f.write(f"Total cost: {total_cost}\n")
            f.write(f"Optimal cost: {total_cost_solution}\n")
            f.write(f"Difference: {total_cost - total_cost_solution:.2f}\n")
            f.write("Solution routes:\n")
            for route in routes_alg:
                f.write(" ".join(str(node) for node in route) + "\n")
            
            f.write("\n")
            f.write("\n")

        
 
if __name__ == "__main__":
    main()