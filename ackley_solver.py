
"""
ackley_solver.py

Dispatcher for minimizing the 10-D Ackley function via:
  • GA (islands)   → ga_ackley_solver
  • ALNS (ruin & rebuild) → alns_ackley_solver
  • BB (random sampling fallback) → bb_ackley_solver

You can run it in two ways:
1) From the command line:
     python ackley_solver.py <METHOD> [<param>=<value> ...]
   where METHOD ∈ {GA, ALNS, BB, ALL}

   Examples:
     python ackley_solver.py GA
     python ackley_solver.py ALNS iterations=1200 k=10
     python ackley_solver.py BB samples=200000
     python ackley_solver.py ALL pop_size=100 gens=200 mut_rate=0.4 samples=50000

2) Directly from an IDE (e.g. PyCharm) without any arguments:
     Just hit “Run” in PyCharm. It will automatically default to METHOD=ALL.
"""

import sys
import time
import threading
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import matplotlib
# matplotlib.use('Agg')  # use non-interactive backend for plots
import matplotlib.pyplot as plt
from GA_with_islands import ga_ackley_solver
from ALNS import alns_ackley_solver
from BranchBound_LDS import bb_ackley_solver
from ILS_Ackley import ils_ackley_solver
from MSH_Ackley import msh_ackley_solver

# ---------------------------------------------------------------------------
# image output settings (all plots saved as .jpg for space efficiency)
# ---------------------------------------------------------------------------
IMAGE_EXT = ".jpg"

DISPLAY_PLOTS = False

if "--no-display" in sys.argv:
    DISPLAY_PLOTS = False
    sys.argv.remove("--no-display")  # remove flag to avoid confusion later
# ---------------------------------------------------------------------------
# prepare timestamped directory for plots
# ---------------------------------------------------------------------------
TIMESTAMP = datetime.now().strftime('%d_%m_%Y_%H_%M_%S')
PLOT_DIR = os.path.join('plots', TIMESTAMP)
os.makedirs(PLOT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# try to import fancier plotting helpers; else fallback to simple ones
# ---------------------------------------------------------------------------
try:
    from plots.plots import plot_line, plot_box, plot_routes, plot_contour
except Exception:
    # fallback line plot
    from matplotlib.lines import Line2D


    def plot_line(x_steps, series_dict, filename, title, xlabel, ylabel, params_text=""):
        """
        simple line plot: each series in series_dict vs x_steps,
        adds grid & legend, saves+shows figure as jpg.
        """
        plt.figure(figsize=(14, 8))
        for label, values in series_dict.items():
            plt.plot(x_steps, values, label=label, linewidth=3.0)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title, fontsize=14)
        plt.grid(True)
        if params_text:
            plt.gcf().text(0.02, 0.95, params_text, ha='left', va='top', fontsize=10)
        plt.legend()
        if filename:
            # default JPEG quality, inferred from .jpg extension
            plt.savefig(filename, dpi=150)
        if DISPLAY_PLOTS:
            plt.show()
        plt.close()


    # fallback box plot
    def plot_box(data, filename, title, params_text=""):
        plt.figure(figsize=(14, 8))
        positions = [i * 1.5 for i in range(1, len(data) + 1)]
        plt.boxplot(
            data,
            positions=positions,
            widths=0.8,
            patch_artist=True,
            boxprops=dict(linewidth=3.0, facecolor="lightblue"),
            medianprops=dict(linewidth=3.0, color="orange"),
            whiskerprops=dict(linewidth=3.0, color="navy"),
            capprops=dict(linewidth=3.0, color="navy"),
            flierprops=dict(marker="o", markersize=4,
                            markerfacecolor="red", markeredgecolor="k")
        )
        plt.xticks(positions)
        plt.title(title)
        if params_text:
            plt.gcf().text(0.02, 0.95, params_text,
                           ha='left', va='top', fontsize=10)
        legend_elems = [
            Line2D([0], [0], color="navy", lw=3.0, label="min/max"),
            Line2D([0], [0], color="orange", lw=3.0, label="median"),
            Line2D([0], [0], marker='o', color='w', label='outlier',
                   markerfacecolor='red', markeredgecolor='k', markersize=6)
        ]
        plt.legend(handles=legend_elems, fontsize=9)
        if filename:
            plt.savefig(filename, dpi=150)
        if DISPLAY_PLOTS:
            plt.show()
        plt.close()


    # placeholder for unsupported route/contour plotting
    def plot_routes(*args, **kwargs):
        pass


    def plot_contour(*args, **kwargs):
        pass

# ---------------------------------------------------------------------------
# valid solver methods
# ---------------------------------------------------------------------------
VALID_METHODS = {"MSH", "ILS", "GA", "ALNS", "BB", "ALL"}


def print_usage():
    """
    print script usage and exit.
    """
    print("Usage: python ackley_solver.py <METHOD> [<param>=<value> ...]")
    print("METHOD ∈", ", ".join(VALID_METHODS))
    print("Optional parameters (with defaults):")
    print("  dim=10       lower=-32.768       upper=32.768   seed=None")
    print("  GA:    pop_size=250, gens=100, mut_rate=0.30")
    print("  ALNS:  iterations=800, remove_low=0.1, remove_high=0.3, k=5")
    print("  BB:    samples=100000, early_stop_tol=None")
    print("\nIf you run this script with no arguments (e.g. via PyCharm),")
    print("it will default to METHOD=ALL with all default parameters.")
    sys.exit(1)


def parse_args(arg_list):
    """
    convert ['pop_size=100','gens=200'] → {'pop_size':100,'gens':200}.
    floats if value contains . or e/E, else ints.
    """
    params = {}
    for token in arg_list:
        if "=" not in token:
            print(f"error: unexpected token '{token}'")
            print_usage()
        key, val_str = token.split("=", 1)
        if any(c in val_str for c in (".", "e", "E")):
            try:
                params[key] = float(val_str)
            except ValueError:
                print(f"error: cannot parse float from '{val_str}'")
                sys.exit(1)
        else:
            try:
                params[key] = int(val_str)
            except ValueError:
                print(f"error: cannot parse int from '{val_str}'")
                sys.exit(1)
    return params


def run_and_report(name: str, solver_func, time_limit=None, **kwargs):
    """
    run solver_func(**kwargs) in background, print progress dots,
    then report best solution, value, elapsed time, and generate plots.
    """
    # print start banner and parameters
    print(f"\n[→] starting {name} with parameters:")
    for k, v in kwargs.items():
        print(f"     {k} = {v}")

    # submit solver to threadpool
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(solver_func, **kwargs)
    start = time.time()

    # progress indicator thread
    def progress_indicator(fut):
        while not fut.done():
            print(".", end="", flush=True)
            for _ in range(10):  # Check more frequently for timeout
                if time_limit and time.time() - start > time_limit:
                    print("\n[!] Time limit reached, stopping execution")
                    fut.cancel()
                    return
                time.sleep(0.1)

    indicator_thread = threading.Thread(target=progress_indicator, args=(future,))
    indicator_thread.daemon = True
    indicator_thread.start()

    # block until solver done
    try:
        sol, val, metrics = future.result()
    except ValueError:
        sol, val = future.result()  # IN case of MSH solver returning only solution and value
        metrics = {}
    elapsed = time.time() - start

    # ensure indicator thread ends
    if indicator_thread.is_alive():
        indicator_thread.join(timeout=0)

    # print results
    coords = sol.tolist() if hasattr(sol, "tolist") else sol
    print(f"\n[✓] {name} completed")
    print(f"    best vector (first 5 coords): {coords[:5]} …")
    print(f"    ackley value: {val:.6f}")
    print(f"    time elapsed: {elapsed:.2f}s\n")

    if name == "ILS":
        for algo, vals in metrics.items():
            if vals:
                x_steps = list(range(1, len(vals) + 1))
                fname = os.path.join(PLOT_DIR, f"{name}_{algo}_ackley_line_{TIMESTAMP}{IMAGE_EXT}")
                plot_line(x_steps, {'best': vals},
                          fname,
                          f"{name} {algo} convergence",
                          'step', 'value', '')
        # Plot all algorithms on one graph for comparison
        comparison_data = {algo: vals for algo, vals in metrics.items() if vals}
        if comparison_data:
            max_len = max(len(vals) for vals in comparison_data.values())
            # Pad shorter lists with their last value for alignment
            for algo in comparison_data:
                if len(comparison_data[algo]) < max_len:
                    last_val = comparison_data[algo][-1]
                    comparison_data[algo] += [last_val] * (max_len - len(comparison_data[algo]))
            x_steps = list(range(1, max_len + 1))
            fname = os.path.join(PLOT_DIR, f"{name}_ackley_comparison_{TIMESTAMP}{IMAGE_EXT}")
            plot_line(x_steps, comparison_data, fname, f"{name} metaheuristics comparison", 'step', 'value', '')

    # convergence plot
    best_vals = (metrics.get('best_value_per_sample') or
                 metrics.get('best_cost_per_iter') or
                 metrics.get('best_fitness_per_gen'))
    if best_vals:
        x_steps = list(range(1, len(best_vals) + 1))
        fname = os.path.join(PLOT_DIR,
                             f"{name}_ackley_line_{TIMESTAMP}{IMAGE_EXT}")
        plot_line(x_steps, {'best': best_vals},
                  fname,
                  f"{name} convergence",
                  'step', 'value', f"time: {elapsed:.2f}s")

    # box plot of fitness series if available
    if metrics.get('worst_fitness_per_gen'):
        # 1) pull out your per-generation stats
        worst = metrics['worst_fitness_per_gen']
        mean = metrics['mean_fitness_per_gen']
        best = metrics['best_fitness_per_gen']

        # 2) decide on 5% bins (20 total)
        total_gens = len(worst)
        bin_size = max(1, total_gens // 20)

        # 3) zip them into (worst,mean,best) tuples per gen
        gen_stats = list(zip(worst, mean, best))

        # 4) group & flatten each 5% chunk
        box_inputs = [
            [v for triple in gen_stats[i:i + bin_size] for v in triple]
            for i in range(0, total_gens, bin_size)
        ]

        # 5) plot
        fname_box = os.path.join(PLOT_DIR,
                                 f"{name}_ackley_box_{TIMESTAMP}{IMAGE_EXT}")
        plot_box(box_inputs,
                 fname_box,
                 f"{name} fitness (5%-of-generation bins)")

    # bounds improvement plot if provided
    if metrics.get('bounds_improvement'):
        imp = metrics['bounds_improvement']
        x = list(range(1, len(imp) + 1))
        fname_bounds = os.path.join(PLOT_DIR,
                                    f"{name}_bounds_{TIMESTAMP}{IMAGE_EXT}")
        plot_line(x, {'improvement': imp},
                  fname_bounds,
                  f"{name} bound improvement",
                  'node', 'delta', f"time: {elapsed:.2f}s")


def main():
    """
    entry point:
      - parse METHOD from CLI or default to ALL
      - parse extra params
      - run requested solver(s) with run_and_report()
    """
    # determine method & extra params
    DISPLAY_PLOTS = True

    if len(sys.argv) < 2:
        method = "ALL"
        extra = {}
    else:
        method = sys.argv[1].upper()
        if method not in VALID_METHODS:
            print(f"error: unknown method '{method}'")
            print_usage()
        extra = parse_args(sys.argv[2:]) if len(sys.argv) > 2 else {}

    # shared defaults
    common = {
        "dim": 10,
        "lower": -32.768,
        "upper": 32.768
    }
    seed = extra.get("seed", None)

    # dispatch logic
    if method == "ALL":
        # GA
        run_and_report(
            "GA", ga_ackley_solver,
            **common,
            pop_size=extra.get("pop_size", 2500),
            gens=extra.get("gens", 250),
            mut_rate=extra.get("mut_rate", 0.35),
            time_limit=extra.get("time_limit", 500),
            seed=seed
        )
        # ALNS
        run_and_report(
            "ALNS", alns_ackley_solver,
            **common,
            iterations=extra.get("iterations", 500),
            remove_bounds=(
                extra.get("remove_low", 0.1),
                extra.get("remove_high", 0.3)
            ),
            k=extra.get("k", 2500),
            time_limit=extra.get("time_limit", 500),
            seed=seed
        )
        # BB
        run_and_report(
            "BB", bb_ackley_solver,
            **common,
            samples=extra.get("samples", 250),
            early_stop_tol=extra.get("early_stop_tol", None),
            time_limit=extra.get("time_limit", 500),
            seed=seed
        )

    elif method == "GA":
        run_and_report(
            "GA", ga_ackley_solver,
            **common,
            pop_size=extra.get("pop_size", 250),
            gens=extra.get("gens", 100),
            mut_rate=extra.get("mut_rate", 0.30),
            time_limit=extra.get("time_limit", 500),
            seed=seed
        )

    elif method == "ALNS":
        run_and_report(
            "ALNS", alns_ackley_solver,
            **common,
            iterations=extra.get("iterations", 800),
            remove_bounds=(
                extra.get("remove_low", 0.1),
                extra.get("remove_high", 0.3)
            ),
            k=extra.get("k", 100),
            time_limit=extra.get("time_limit", 500),
            seed=seed
        )

    elif method == "BB":
        run_and_report(
            "BB", bb_ackley_solver,
            **common,
            samples=extra.get("samples", 100000),
            early_stop_tol=extra.get("early_stop_tol", None),
            time_limit=extra.get("time_limit", 1200),  # optional time limit
            seed=seed
        )
    elif method == "ILS":
        run_and_report(
            "ILS", ils_ackley_solver,
            **common,
            meta_heuristic=extra.get("meta_heuristic", "parallel_hybrid"),  # or "aco", "tabu_search", "parallel_hybrid"
            time_limit=extra.get("time_limit", 1200),
            max_searches=extra.get("max_searches", 6000),
            seed=seed
        )

    elif method == "MSH":
        run_and_report(
            "MSH", msh_ackley_solver,
            **common,
            num_clusters=extra.get("num_clusters", 8),
            time_limit=extra.get("time_limit", 100),
            seed=seed
        )


if __name__ == "__main__":
    main()