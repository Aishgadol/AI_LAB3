
from __future__ import annotations

import sys
import os
import glob
import re
import time
import math
from datetime import datetime
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from matplotlib.lines import Line2D

from cvrp_master_tester import DISPLAY_PLOTS
# ---------------------------------------------------------------------------
# project utilities & solver implementations (left untouched) --------------
# ---------------------------------------------------------------------------
from utils import cost_function, validate_routes_capacity, validate_routes_cities
from MSH import MSH_alg
from ILS import parallel_hybrid_ils
from GA_with_islands import ga_island_model_solver
import ALNS
from ALNS import alns_metaheuristic_solver
import BranchBound_LDS
from BranchBound_LDS import branch_and_bound_lds_solver

# ---------------------------------------------------------------------------
# setup: timestamped output directory ---------------------------------------
# ---------------------------------------------------------------------------
# timestamp for file names & plot folders
TIMESTAMP = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
PLOT_DIR = os.path.join("plots", TIMESTAMP)
# defaults used if running without CLI args (IDE mode)
DEFAULT_METHOD = "ILS"
DEFAULT_VRP_PATH = "./problems/intermediate"  # /A-n32-k5.vrp"
DISPLAY_PLOTS = False  # whether to show plots interactively

IMAGE_EXT = ".jpg"
IMAGE_FORMAT = "jpg"  # default image format for plots
# ────────────────────────────────────────────────────────────────────────────

os.makedirs(PLOT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# fallback simple plotting routines (line, box, routes) --------------------
# ---------------------------------------------------------------------------
try:
    from plots.plots import plot_line, plot_box, plot_routes, plot_contour
except ImportError:
    # fallback line plot
    def plot_line(x_steps: List[int],
                  series_dict: Dict[str, List[float]],
                  filename: str,
                  title: str,
                  xlabel: str,
                  ylabel: str,
                  params_text: str = "") -> None:
        """
        quick line plot fallback: draws each series in series_dict vs x_steps,
        adds grid, legend, optional params text, saves+shows the figure.
        """
        plt.figure(figsize=(14, 8))
        for series_label, series_values in series_dict.items():
            plt.plot(x_steps, series_values, label=series_label, linewidth=3.0)
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.title(title, fontsize=14)
        plt.grid(True)
        if params_text:
            plt.gcf().text(0.02, 0.95, params_text,
                           ha='left', va='top', fontsize=10)
        plt.legend()
        if filename:
            plt.savefig(filename, dpi=150)
        if DISPLAY_PLOTS:
            plt.show()
        plt.close()


    # fallback box‐and‐whisker plot
    def plot_box(data_groups: List[List[float]],
                 filename: str,
                 title: str,
                 params_text: str = "") -> None:
        """
        quick box plot fallback: each group in data_groups is a separate box.
        positions spaced for readability; includes custom legend for outliers.
        """
        plt.figure(figsize=(14, 8))
        positions = [i * 1.5 for i in range(1, len(data_groups) + 1)]
        bp = plt.boxplot(data_groups,
                         positions=positions,
                         widths=0.8,
                         patch_artist=True,
                         boxprops=dict(linewidth=3.0, facecolor='lightblue'),
                         medianprops=dict(linewidth=3.0, color='orange'),
                         whiskerprops=dict(linewidth=3.0, color='navy'),
                         capprops=dict(linewidth=3.0, color='navy'),
                         flierprops=dict(marker='o', markersize=4,
                                         markerfacecolor='red',
                                         markeredgecolor='k'))
        plt.xticks(positions)
        plt.title(title, fontsize=14)
        if params_text:
            plt.gcf().text(0.02, 0.95, params_text, ha='left', va='top', fontsize=10)
        legend_elems = [
            Line2D([0], [0], color='navy', lw=3.0, label='min/max'),
            Line2D([0], [0], color='orange', lw=3.0, label='median'),
            Line2D([0], [0], marker='o', color='w', label='outlier',
                   markerfacecolor='red', markeredgecolor='k', markersize=6)
        ]
        plt.legend(handles=legend_elems, fontsize=9)
        if filename:
            plt.savefig(filename, dpi=150)
        if DISPLAY_PLOTS:
            plt.show()
        plt.close()


    # fallback side‐by‐side route plot
    def plot_routes(reference_data: dict,
                    solution_data: dict,
                    filename: str,
                    title: str) -> None:
        """
        quick fallback for side‐by‐side routes: left=suboptimal, right=algo.
        uses same aesthetics as main but in two axes.
        """
        coords = reference_data['coords']
        demands = reference_data['demands']
        capacity = reference_data['capacity']
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        cmap = plt.get_cmap('tab20')

        def _draw(ax, routes_list, cost_val, label_text):
            # draw each route as colored line+markers
            for idx, route in enumerate(routes_list, start=1):
                col = cmap(idx % 20)
                xs = [coords[node][0] for node in route]
                ys = [coords[node][1] for node in route]
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
                ax.scatter([dx], [dy], s=140, marker='*', c='red',
                           edgecolor='k', label='depot', zorder=7)
            ax.set_title(f"{label_text} cost={cost_val:.1f}", fontsize=12)
            ax.set_aspect('equal')
            ax.grid(True, ls=':')
            ax.set_xlabel('x');
            ax.set_ylabel('y')
            ax.legend(fontsize=8, loc='upper right')

        _draw(axes[0], reference_data['routes'], reference_data['cost'], 'reference')
        _draw(axes[1], solution_data['routes'], solution_data['cost'], 'solution')
        plt.suptitle(title, fontsize=14)
        if filename:
            plt.savefig(filename, dpi=150)  # , bbox_inches='tight')
        if DISPLAY_PLOTS:
            plt.show()
        plt.close()


# ---------------------------------------------------------------------------
# parsing .vrp file (now extracts depot id) ---------------------------------
# ---------------------------------------------------------------------------

def parse_vrp_file(vrp_filepath: str
                   ) -> Tuple[int | None, int, int,
Dict[int, Tuple[float, float]],
Dict[int, int],
int]:
    """
    parse a TSPLIB‐style .vrp file.
    returns:
      - min_number_of_vehicles_hint  (if COMMENT provides it)
      - vehicle_capacity              (CAPACITY)
      - total_number_of_nodes         (DIMENSION)
      - node_coordinates_map          {node_id: (x,y)}
      - demand_quantity_map           {node_id: demand}
      - depot_node_id                 (first ID in DEPOT_SECTION)
    """
    node_coordinates_map: Dict[int, Tuple[float, float]] = {}
    demand_quantity_map: Dict[int, int] = {}
    min_number_of_vehicles_hint = None
    vehicle_capacity = None
    total_number_of_nodes = None
    depot_node_id = None
    current_section = None

    with open(vrp_filepath, 'r') as file_obj:
        for raw_line in file_obj:
            line = raw_line.strip()
            if not line:
                continue

            # extract vehicle hint from COMMENT if present
            if line.startswith('COMMENT') and 'Min no of trucks:' in line:
                try:
                    min_number_of_vehicles_hint = int(
                        line.split('Min no of trucks:')[1].split(',')[0].strip()
                    )
                except ValueError:
                    # ignore unparsable COMMENT
                    pass

            # parse capacity
            elif line.startswith('CAPACITY'):
                try:
                    vehicle_capacity = int(line.split()[-1])
                except ValueError:
                    pass

            # parse node count
            elif line.startswith('DIMENSION'):
                try:
                    total_number_of_nodes = int(line.split()[-1])
                except ValueError:
                    pass

            # section markers
            elif line == 'NODE_COORD_SECTION':
                current_section = 'NODE_COORD'
                continue
            elif line == 'DEMAND_SECTION':
                current_section = 'DEMAND'
                continue
            elif line == 'DEPOT_SECTION':
                current_section = 'DEPOT'
                continue
            elif line == 'EOF':
                current_section = None
                continue

            # parse coordinates
            if current_section == 'NODE_COORD':
                parts = line.split()
                if len(parts) == 3:
                    try:
                        node = int(parts[0])
                        xval = float(parts[1])
                        yval = float(parts[2])
                        node_coordinates_map[node] = (xval, yval)
                    except ValueError:
                        pass

            # parse demands
            elif current_section == 'DEMAND':
                parts = line.split()
                if len(parts) == 2:
                    try:
                        node = int(parts[0])
                        demand = int(parts[1])
                        demand_quantity_map[node] = demand
                    except ValueError:
                        pass

            # parse depot id
            elif current_section == 'DEPOT':
                if line != '-1':
                    try:
                        depot_node_id = int(line)
                    except ValueError:
                        pass
                else:
                    # end of DEPOT_SECTION
                    current_section = None

    # sanity check required fields
    if vehicle_capacity is None or total_number_of_nodes is None or depot_node_id is None:
        raise ValueError(f"'{vrp_filepath}' missing CAPACITY, DIMENSION or DEPOT info")

    # info printout
    print(f"[INFO] parsed '{os.path.basename(vrp_filepath)}' → "
          f"capacity={vehicle_capacity}, nodes={total_number_of_nodes}, "
          f"depot={depot_node_id}, minVehHint={min_number_of_vehicles_hint}")

    return (min_number_of_vehicles_hint,
            vehicle_capacity,
            total_number_of_nodes,
            node_coordinates_map,
            demand_quantity_map,
            depot_node_id)


# ---------------------------------------------------------------------------
# load_reference_solution: id‐increment + depot‐prepend/append --------------
# ---------------------------------------------------------------------------

def load_reference_solution(vrp_filepath: str,
                            coords_map: Dict[int, Tuple[float, float]] | None = None,
                            depot_node_id: int = 1
                            ) -> Tuple[List[List[int]], float | None]:
    """
    load the .sol companion file (same base name), if exists.
    each raw route line is parsed, then **every ID is incremented by 1**,
    then depot_node_id is prepended/appended if missing.
    returns (fixed_routes, cost_value)
    """
    sol_filepath = vrp_filepath.replace('.vrp', '.sol')
    if not os.path.exists(sol_filepath):
        return None, None

    print(f"[INFO] found reference .sol file: '{os.path.basename(sol_filepath)}'")

    route_pattern = re.compile(r"Route\s*(?:#?\d+)?\s*:\s*(.*)", re.IGNORECASE)
    raw_routes = []
    cost_value = None

    # parse routes and cost
    with open(sol_filepath, 'r') as sol_file:
        for raw_line in sol_file:
            line = raw_line.strip()
            match = route_pattern.search(line)
            if match:
                # parse node IDs and increment each by 1
                try:
                    node_list = [int(tok) + 1 for tok in match.group(1).split()]
                    raw_routes.append(node_list)
                except ValueError:
                    # skip if not purely ints
                    pass
            elif line.lower().startswith('cost'):
                try:
                    cost_value = float(line.split()[1])
                except (IndexError, ValueError):
                    # skip unparsable cost
                    pass

    # ensure depot boundaries on each route
    fixed_routes: List[List[int]] = []
    for route in raw_routes:
        if not route:
            continue
        if route[0] != depot_node_id:
            route = [depot_node_id] + route
        if route[-1] != depot_node_id:
            route = route + [depot_node_id]
        fixed_routes.append(route)

    # if coords provided and cost missing, compute it
    if coords_map is not None and cost_value is None:
        cost_value = sum(cost_function(r, coords_map) for r in fixed_routes)

    # print loaded info
    # format cost cleanly (only “n/a” if cost_value is None)
    cost_str = f"{cost_value:.1f}" if cost_value is not None else "n/a"
    print(f"[INFO] reference .sol loaded: {len(fixed_routes)} routes, cost={cost_str}")

    return fixed_routes, cost_value


# ---------------------------------------------------------------------------
# repair_routes: ensures all customers served, capacity respected ----------
# ---------------------------------------------------------------------------

def repair_routes(raw_route_list: List[List[int]],
                  demand_map: Dict[int, int],
                  vehicle_capacity: int,
                  depot_node_id: int = 1
                  ) -> List[List[int]]:
    """
    safety net: ensure no missing customers & capacity not violated.
    starts/ends each route at depot_node_id, removes duplicates,
    then greedily inserts any missing customers.
    """
    all_customers = set(demand_map.keys())
    if depot_node_id in all_customers:
        all_customers.remove(depot_node_id)

    cleaned_routes: List[List[int]] = []
    visited = set()

    # step 1: force each raw route to start & end at depot, remove duplicates
    for raw_route in raw_route_list:
        route = raw_route[:]
        if not route:
            continue
        if route[0] != depot_node_id:
            route.insert(0, depot_node_id)
        if route[-1] != depot_node_id:
            route.append(depot_node_id)

        filtered = [depot_node_id]
        for n in route[1:-1]:
            if n not in visited and n in all_customers:
                filtered.append(n)
                visited.add(n)
        filtered.append(depot_node_id)
        cleaned_routes.append(filtered)

    # step 2: insert missing customers greedily
    missing = all_customers.difference(visited)
    for cust in missing:
        placed = False
        for route in cleaned_routes:
            load = sum(demand_map[node] for node in route if node != depot_node_id)
            if load + demand_map[cust] <= vehicle_capacity:
                route.insert(-1, cust)
                placed = True
                break
        if not placed:
            cleaned_routes.append([depot_node_id, cust, depot_node_id])

    return cleaned_routes


# ---------------------------------------------------------------------------
# print_cvrp_solution: human‐readable solution dump ------------------------
# ---------------------------------------------------------------------------

def print_cvrp_solution(route_list: List[List[int]],
                        total_cost: float) -> None:
    """
    prints each route with arrows and total cost.
    prefix lines with [RESULT].
    """
    print("[RESULT] solution routes:")
    for idx, route in enumerate(route_list, start=1):
        print(f"  route {idx}: {' → '.join(map(str, route))}")
    print(f"  total cost: {total_cost:.2f}")


# ---------------------------------------------------------------------------
# main execution logic (solver dispatch, plotting) – left unchanged --------
# ---------------------------------------------------------------------------
def main():
    """
    entry point:
      - read METHOD + PATH from CLI or use defaults
      - gather .vrp files
      - for each file:
          * parse (.vrp)
          * load_reference_solution()
          * run solver exactly once
          * repair_routes()
          * print & plot solution + reference overlay
          * plot convergence & grouped box if metrics available
    """
    # CLI vs IDE defaults
    time_limit = None
    if len(sys.argv) >= 4:
        method_choice = sys.argv[1].upper()
        vrp_path_input = sys.argv[2]
        time_limit = float(sys.argv[3])
        time_checker = "time_limit"
    elif len(sys.argv) >= 3:
        method_choice = sys.argv[1].upper()
        vrp_path_input = sys.argv[2]
        time_checker = "default"
    else:
        method_choice = DEFAULT_METHOD
        vrp_path_input = DEFAULT_VRP_PATH
        time_checker = "default"
        print(f"[info] no CLI args → using defaults: METHOD={method_choice}, PATH={vrp_path_input}")

    valid_methods = {"MSH", "ILS", "GA", "ALNS", "BB"}
    if method_choice not in valid_methods:
        print(f"[ERROR] unknown method '{method_choice}' – choose from {valid_methods}")
        sys.exit(1)

    # normalize BB alias
    if method_choice == "B&B":
        method_choice = "BB"

    # gather vrp files
    if os.path.isdir(vrp_path_input):
        vrp_file_list = sorted(glob.glob(os.path.join(vrp_path_input, "*.vrp")))
        if not vrp_file_list:
            print(f"[ERROR] no .vrp in directory '{vrp_path_input}'");
            sys.exit(1)
    else:
        if not vrp_path_input.endswith(".vrp") or not os.path.exists(vrp_path_input):
            print(f"[ERROR] invalid .vrp path '{vrp_path_input}'");
            sys.exit(1)
        vrp_file_list = [vrp_path_input]

    print(f"[INFO] found {len(vrp_file_list)} .vrp files in '{vrp_path_input}'")
    # solve each problem file
    for vrp_file in vrp_file_list:
        # extract problem name from path
        prob_name = os.path.basename(vrp_file)

        print(f"\n[INFO] solving '{prob_name}' with method '{method_choice}'")

        # parse VRP
        (vehicles_hint,
         capacity,
         total_nodes,
         coords_map,
         demands_map,
         depot_id) = parse_vrp_file(vrp_file)

        # reference solution
        reference_routes, reference_cost = load_reference_solution(
            vrp_file, coords_map, depot_id)

        # solver dispatch
        start_time = time.time()
        if method_choice == "MSH":
            if time_checker == "default":
                time_limit = 200  # Default time limit in seconds for MSH

            raw_solution, _ = MSH_alg(coords_map, demands_map, capacity, total_nodes, time_limit)
            metrics = {}
        elif method_choice == "ILS":
            # ILS parameters based on problem size

            if total_nodes <= 30:
                max_searches = 3000  # Default number of searches for ILS and MSH
                if time_checker == "default":
                    time_limit = 800  # Default time limit in seconds
            elif total_nodes <= 80:
                max_searches = 9000
                if time_checker == "default":
                    time_limit = 1800  # 1200  # Default time limit in seconds
            else:
                max_searches = 20000
                if time_checker == "default":
                    time_limit = 3600  # Default time limit in seconds

            raw_solution, _, metrics = parallel_hybrid_ils(coords_map, demands_map, capacity, total_nodes, time_limit,
                                                           max_searches)
            # metrics = {}


        elif method_choice == "GA":
            import GA_with_islands
            GA_with_islands.__ga_demands = demands_map
            GA_with_islands.__ga_capacity = capacity
            num_vehicles = vehicles_hint or math.ceil(sum(demands_map.values()) / capacity)
            if time_checker == "default":
                time_limit = None
            raw_solution, _, metrics = ga_island_model_solver(
                coords_map, demands_map, capacity, num_vehicles, time_limit)


        elif method_choice == "ALNS":
            num_vehicles= vehicles_hint or math.ceil(sum(demands_map.values()) / capacity)
            if time_checker == "default":
                time_limit = None
            raw_solution, _, metrics = alns_metaheuristic_solver(
                coords_map, demands_map, capacity, num_vehicles, time_limit=time_limit)


        else:  # BB
            if time_checker == "default":
                time_limit = None
            raw_solution, _, metrics = branch_and_bound_lds_solver(
                coords_map, demands_map, capacity, total_nodes, time_limit)
        elapsed = time.time() - start_time
        print(f"[INFO] {method_choice} completed in {elapsed:.2f}s")

        # repair solution
        final_routes = repair_routes(raw_solution, demands_map, capacity, depot_id)
        total_cost = sum(cost_function(r, coords_map) for r in final_routes)

        # print and per‐instance plot
        print_cvrp_solution(final_routes, total_cost)
        sol_data = {
            'coords': coords_map,
            'routes': final_routes,
            'cost': total_cost,
            'demands': demands_map,
            'capacity': capacity
        }
        if reference_routes:
            ref_data = {
                'coords': coords_map,
                'routes': reference_routes,
                'cost': reference_cost or 0,
                'demands': demands_map,
                'capacity': capacity
            }
        else:
            ref_data = {'coords': coords_map, 'routes': [], 'cost': 0, 'demands': demands_map, 'capacity': capacity}

        out_png = os.path.join(PLOT_DIR,
                               f"{method_choice}_{prob_name}_routes_{TIMESTAMP}{IMAGE_EXT}")
        plot_routes(ref_data, sol_data, out_png, f"{method_choice} vs reference for {prob_name}")

        # convergence + group‐box if metrics present

        if method_choice == "ILS":
            # show the three metaheuristics together
            # Plot all algorithms on one graph for comparison
            all_algorithms = ["aco", "simulated_annealing", "tabu_search"]
            best_costs_per_algo = {}

            # Extract metrics for each algorithm
            for algo in all_algorithms:
                algo_metrics = metrics.get(algo, {})
                if algo_metrics:
                    # Store best costs by algorithm
                    best_costs = algo_metrics.get("best_costs", [])
                    if best_costs:
                        best_costs_per_algo[algo] = best_costs
            if best_costs_per_algo:
                # Find the maximum length of any cost series
                max_len = max([len(costs) for costs in best_costs_per_algo.values()], default=0)
                # Pad shorter lists with their last value for alignment
                for algo in best_costs_per_algo:
                    if len(best_costs_per_algo[algo]) < max_len:
                        last_val = best_costs_per_algo[algo][-1]
                        best_costs_per_algo[algo] += [last_val] * (max_len - len(best_costs_per_algo[algo]))
                # Plotting the convergence
                xs = list(range(1, max_len + 1))
                f_line = os.path.join(PLOT_DIR,
                                      f"{method_choice}_{prob_name}_conv_{TIMESTAMP}{IMAGE_EXT}")
                plot_line(xs, best_costs_per_algo, f_line,
                          f"{method_choice} convergence – {prob_name}", "step", "obj",
                          params_text=f"time: {elapsed:.2f}s")
            metrics = {}

        best_series = (metrics.get('best_cost_per_iter') or
                       metrics.get('best_fitness_per_gen') or
                       metrics.get('best_value_per_sample'))
        if best_series:
            xs = list(range(1, len(best_series) + 1))
            conv_png = os.path.join(PLOT_DIR,
                                    f"{method_choice}_{prob_name}_convergence_{TIMESTAMP}{IMAGE_EXT}")
            plot_line(xs, {'best': best_series}, conv_png,
                      f"{method_choice} convergence for {prob_name}",
                      'step', 'objective', f"time: {elapsed:.2f}s")

            worst_series = metrics.get('worst_fitness_per_gen')
            if worst_series:
                # 5% grouping
                n = len(best_series)
                group = max(1, int(n * 0.05))
                grouped = []
                for i in range(0, n, group):
                    seg_b = best_series[i:i + group]
                    seg_w = worst_series[i:i + group]
                    if seg_b and seg_w:
                        grouped.append([min(seg_b), max(seg_w)])
                if len(grouped) > 1:
                    gbox_png = os.path.join(PLOT_DIR,
                                            f"{method_choice}_{prob_name}_gbox_{TIMESTAMP}{IMAGE_EXT}")
                    plot_box(grouped, gbox_png,
                             f"{method_choice} fitness 5% groups for {prob_name}", '')

    print("\n[INFO] all problems solved. exiting runner.")


if __name__ == '__main__':
    main()