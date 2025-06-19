import matplotlib.pyplot as plt
import numpy as np

def plot_costs(costs, iterations=None, title="Cost over Iterations", ylabel="Cost", xlabel="Iteration", save_path=None, dispay=False):
    plt.figure(figsize=(8, 5))
    if iterations is not None:
        x = list(range(iterations))
        plt.plot(x, costs, label="Cost")
    else:
        plt.plot(costs, label="Cost")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    if dispay:
        plt.show()

def plot_comparison(optimal_value, found_costs, iterations=None, algorithm_name="Algorithm", title="Algorithm vs Optimal", ylabel="Cost", xlabel="Iteration", save_path=None, dispay=False):
    plt.figure(figsize=(8, 5))
    if iterations is not None:
        x = list(range(iterations))
        plt.plot(x, found_costs, label=f"{algorithm_name} Cost")
        plt.hlines(optimal_value, 0, iterations-1, colors='r', linestyles='dashed', label="Optimal Cost")
    else:
        plt.plot(found_costs, label=f"{algorithm_name} Cost")
        plt.hlines(optimal_value, 0, len(found_costs)-1, colors='r', linestyles='dashed', label="Optimal Cost")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    if dispay:
        plt.show()

def plot_fitness_boxplot(max_fitness, avg_fitness, min_fitness, title="Fitness Distribution per Generation", ylabel="Fitness", xlabel="Generation", save_path=None):
    data = [min_fitness, avg_fitness, max_fitness]
    data = np.array(data)
    data = data.T  # shape: (generations, 3)
    plt.figure(figsize=(10, 6))
    plt.boxplot(data, labels=[f"Gen {i+1}" for i in range(data.shape[0])])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_multiple_algorithms(cost_lists, labels, title="Algorithm Comparison", ylabel="Cost", xlabel="Iteration", save_path=None):
    plt.figure(figsize=(10, 6))
    for costs, label in zip(cost_lists, labels):
        plt.plot(costs, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_final_costs_boxplot(cost_lists, labels, title="Final Solution Cost Distribution", ylabel="Final Cost", save_path=None):
    data = [costs[-1] if hasattr(costs, '__iter__') and len(costs) > 0 else costs for costs in cost_lists]
    plt.figure(figsize=(8, 5))
    plt.boxplot(data, labels=labels)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_runtime_bar(runtimes, labels, title="Algorithm Runtime Comparison", ylabel="Runtime (s)", save_path=None):
    plt.figure(figsize=(8, 5))
    plt.bar(labels, runtimes)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, axis='y')
    if save_path:
        plt.savefig(save_path)
    plt.show()




"""
 Nodes vs. Quality (Branch & Bound/LDS)

"""
def plot_nodes_vs_quality(nodes, best_costs, title="Nodes Explored vs Solution Quality", ylabel="Best Cost", xlabel="Nodes", save_path=None):
    plt.figure(figsize=(8, 5))
    plt.scatter(nodes, best_costs, color='blue', label='Best Cost')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.show()
    
"""
Cost vs. Time (Branch & Bound/LDS)
"""
def plot_cost_vs_time(costs, times, title="Cost vs Time", ylabel="Cost", xlabel="Time (s)", save_path=None):
    plt.figure(figsize=(8, 5))
    plt.plot(times, costs, marker='o', color='green', label='Cost over Time')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.show()


"""
 parameters tuning (ILS, ALNS, etc.)"""
def plot_parameter_tuning(parameter_values, costs, title="Parameter Tuning", ylabel="Cost", xlabel="Parameter Value", save_path=None):
    plt.figure(figsize=(8, 5))
    plt.plot(parameter_values, costs, marker='o', color='purple', label='Cost vs Parameter Value')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.show()


""" 
 Migration Effect (GA_with_ISLANDS)

"""
def plot_migration_effect(migration_rates, costs, title="Migration Effect on Cost", ylabel="Cost", xlabel="Migration Rate", save_path=None):
    plt.figure(figsize=(8, 5))
    plt.plot(migration_rates, costs, marker='o', color='cyan', label='Cost vs Migration Rate')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.show()
    
    
"""
ILS

"""
def plot_metaheuristic_performance(costs_aco,costs_sa,costs_ts, iterations=None, title="Metaheuristic Performance", ylabel="Cost", xlabel="Iteration", save_path=None, dispay=False):
    plt.figure(figsize=(10, 6))
    if iterations is not None:
        x = list(range(iterations))
        plt.plot(x, costs_aco, label="ACO Cost")
        plt.plot(x, costs_sa, label="Simulated Annealing Cost")
        plt.plot(x, costs_ts, label="Tabu Search Cost")
    else:
        plt.plot(costs_aco, label="ACO Cost")
        plt.plot(costs_sa, label="Simulated Annealing Cost")
        plt.plot(costs_ts, label="Tabu Search Cost")
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    if dispay:
        plt.show()
    


    
def plot_parameters_ils(parameters, best_costs, metaheuristic="tabu_search", save_path=None, dispay=False):

    fig, ax1 = plt.subplots(figsize=(10, 6))

    if isinstance(parameters, dict):
        if metaheuristic == "aco":
            param_names = ["Alpha", "Beta", "Rho", "Q"]
            keys = ['alpha', 'beta', 'rho', 'Q']
        elif metaheuristic == "simulated_annealing":
            param_names = ["Temperature", "Alpha"]
            keys = ['T', 'alpha']
        else:
            param_names = list(parameters.keys())
            keys = param_names

        for i, key in enumerate(keys):
            if key in parameters:
                ax1.plot(parameters[key], label=param_names[i])
    elif isinstance(parameters, list):
        ax1.plot(parameters, label="Tabu Tenure")
    else:
        ax1.plot(parameters, label="Parameter")

    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Parameter Value")
    ax1.grid(True)
    ax1.legend(loc="upper left")

    if best_costs is not None and len(best_costs) > 0:
        ax2 = ax1.twinx()
        ax2.plot(best_costs, label="Best Cost", color='purple', linestyle='--', alpha=0.6)
        ax2.set_ylabel("Best Cost")
        ax2.legend(loc="upper right")

    plt.title(f"{metaheuristic.capitalize()}: Parameters and Best Cost Over Iterations")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    if dispay:
        plt.show()
    
def plot_aco_line_stats(worst_costs, avg_costs, best_costs, title="ACO Cost Evolution", ylabel="Cost", xlabel="Generation", save_path=None, dispay = False):
    plt.figure(figsize=(14, 6))
    plt.plot(worst_costs, label="Worst Cost", color="#FF9999")
    plt.plot(avg_costs, label="Average Cost", color="#99CCFF")
    plt.plot(best_costs, label="Best Cost", color="#99FF99")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    if dispay:
        plt.show()
    
    
# Example usage:

# costs = [your_cost_list]
# plot_costs(costs)
# optimal_value = your_optimal_value
# found_costs = [your_found_cost_list]
# plot_comparison(optimal_value, found_costs)

#comparison of algorithms (multiple algorithms)
# cost_lists: list of cost lists from different algorithms
# labels: list of algorithm names
#plot_multiple_algorithms(cost_lists, labels, title="Algorithm Comparison")


#Cost vs. Iteration (ILS, ALNS, etc.)
# costs: list of costs per iteration
#plot_costs(costs, title="Cost vs. Iteration")

#Comparison to Optimal (ILS, ALNS, etc.)
#plot_comparison(optimal_value, found_costs, title="Algorithm vs Optimal Cost", algorithm_name="Your Algorithm")


#Migration Effect (GA_with_ISLANDS)
# island_costs: list of lists, each for an island's best cost per generation
# labels: list of island names
#plot_multiple_algorithms(island_costs, labels, title="Migration Effect on Islands")