



import matplotlib.pyplot as plt
import numpy as np

def plot_best_cost_over_iterations(metrics, algo_name="Algorithm", save_path=None):
    """Plot best cost over iterations/epochs."""
    best_costs = metrics.get('best_costs', [])
    plt.figure(figsize=(8, 5))
    plt.plot(best_costs, label=f"{algo_name} Best Cost")
    plt.xlabel("Iteration/Epoch")
    plt.ylabel("Best Cost")
    plt.title(f"Best Cost Over Iterations - {algo_name}")
    plt.grid(True)
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_aco_cost_stats(metrics, save_path=None):
    """Plot best, average, and worst cost for ACO over epochs."""
    best_costs = metrics.get('best_costs', [])
    avg_costs = metrics.get('avg_costs', [])
    worst_costs = metrics.get('worst_costs', [])
    plt.figure(figsize=(8, 5))
    plt.plot(best_costs, label="Best Cost")
    plt.plot(avg_costs, label="Average Cost")
    plt.plot(worst_costs, label="Worst Cost")
    plt.xlabel("Epoch")
    plt.ylabel("Cost")
    plt.title("ACO: Best, Average, and Worst Cost Over Epochs")
    plt.grid(True)
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_parameters_vs_best_cost(metrics, param_names=None, algo_name="ACO", save_path=None):
    """Plot parameter values and best cost over epochs."""
    params = np.array(metrics.get('parameters', []))
    best_costs = metrics.get('best_costs', [])
    if params.size == 0 or len(best_costs) == 0:
        print("No parameter or best cost data to plot.")
        return
    if param_names is None:
        # Default for ACO: alpha, beta, rho, Q
        param_names = [f"Param {i+1}" for i in range(params.shape[1])]
    fig, ax1 = plt.subplots(figsize=(10, 6))
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    for i in range(params.shape[1]):
        ax1.plot(params[:, i], label=param_names[i], color=colors[i % len(colors)])
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Parameter Value")
    ax1.legend(loc="upper left")
    ax1.grid(True)
    # Plot best cost on secondary y-axis
    ax2 = ax1.twinx()
    ax2.plot(best_costs, label="Best Cost", color='tab:purple', linestyle='--')
    ax2.set_ylabel("Best Cost")
    ax2.legend(loc="upper right")
    plt.title(f"{algo_name}: Parameters and Best Cost Over Epochs")
    if save_path:
        plt.savefig(save_path)
    plt.show()