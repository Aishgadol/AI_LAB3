
from __future__ import annotations
import matplotlib.patheffects as PathEffects
import os, sys, glob, math, time, re
from datetime import datetime
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

# ---------------------------------------------------------------------------
# project utils & solvers ----------------------------------------------------
# ---------------------------------------------------------------------------
from utils import cost_function, validate_routes_capacity, validate_routes_cities
from MSH import MSH_alg
from ILS import ILS
import GA_with_islands
from GA_with_islands import ga_island_model_solver
from ALNS import alns_metaheuristic_solver
from BranchBound_LDS import branch_and_bound_lds_solver

os.makedirs('plots', exist_ok=True)
# timestamp for output directory and file names
TIMESTAMP = datetime.now().strftime('%d_%m_%Y_%H_%M_%S')
PLOT_DIR = os.path.join('plots', TIMESTAMP)
IMAGE_EXT=".jpg"
IMAGE_FORMAT = "jpg"  # default image format for plots
DISPLAY_PLOTS = False  # set to False to disable plot display
os.makedirs(PLOT_DIR, exist_ok=True)

# fall-back quick-plots (line / box only) -----------------------------------
try:
    from plots.plots import plot_line, plot_box  # type: ignore
except Exception:
    def plot_line(x, y_dict, filename, title, xlabel, ylabel, params_text=""):
        plt.figure(figsize=(14, 8))
        for lbl, ys in y_dict.items():
            plt.plot(x, ys, label=lbl, linewidth=3.0)
        plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title)
        plt.grid(True); plt.legend()
        if params_text:
            plt.gcf().text(0.02, 0.95, params_text, ha="left", va="top")
        if filename:
            plt.savefig(filename, dpi=150)
        if DISPLAY_PLOTS:
            plt.show()
        plt.close()

    def plot_box(data, filename, title, params_text=""):
        plt.figure(figsize=(14, 8))
        pos = [i * 1.5 for i in range(1, len(data) + 1)]
        bp = plt.boxplot(data,
                         positions=pos,
                         widths=0.8,
                         patch_artist=True,
                         boxprops=dict(linewidth=3.0, facecolor="lightblue"),
                         medianprops=dict(linewidth=3.0, color="orange"),
                         whiskerprops=dict(linewidth=3.0, color="navy"),
                         capprops=dict(linewidth=3.0, color="navy"),
                         flierprops=dict(marker="o", markersize=4,
                                         markerfacecolor="red",
                                         markeredgecolor="k"))

        plt.xticks(pos)
        plt.title(title)
        if params_text:
            plt.gcf().text(0.02, 0.95, params_text, ha="left", va="top")
        legend_elems = [Line2D([0], [0], color="navy", lw=3.0, label="min/max"),
                        Line2D([0], [0], color="orange", lw=3.0, label="median"),
                        Line2D([0], [0], marker='o', color='w', label='outlier',
                               markerfacecolor='red', markeredgecolor='k',
                               markersize=6)]
        plt.legend(handles=legend_elems, fontsize=9)
        if filename:
            plt.savefig(filename, dpi=150)
        if DISPLAY_PLOTS:
            plt.show()
        plt.close()

# ---------------------------------------------------------------------------
# bespoke “side-by-side” plot used during batch loop ------------------------
# ---------------------------------------------------------------------------
def plot_routes(ref_d: dict, sol_d: dict,
                filename: str | None,
                title: str,
                depot: int = 1) -> None:
    """Single figure, two axes: left = optimal, right = algorithm solution."""
    coords = sol_d["coords"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    cmap = plt.get_cmap("tab20")

    def _draw(ax, routes, cost, label):
        for i, rt in enumerate(routes, start=1):
            col = cmap(i % 20)
            xs = [coords[n][0] for n in rt]
            ys = [coords[n][1] for n in rt]
            ax.plot(xs, ys, "-o", c=col, lw=3.0,
                    markeredgecolor="k", markersize=5,
                    label=f"{label} r{i}")
            for n in rt:
                x, y = coords[n]
                txt = ax.text(
                    x, y, str(n),
                    fontsize=9,
                    ha="center", va="center",
                    color="black",  # text fill
                    zorder=6  # draw on top of everything else
                )
                # give each character a white outline
                txt.set_path_effects([
                    PathEffects.Stroke(linewidth=2, foreground="white"),
                    PathEffects.Normal()
                ])

        dx, dy = coords[depot]
        ax.scatter([dx], [dy], marker="*", s=140, c="red",
                   edgecolor="k", label="depot")
        ax.set_title(f"{label}  cost={cost:.1f}")
        ax.set_aspect("equal"); ax.grid(True, ls=":")
        ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.legend(fontsize=7, loc="upper right")

    _draw(axes[0], ref_d["routes"], ref_d["cost"], "opt")
    _draw(axes[1], sol_d["routes"], sol_d["cost"], "algo")

    plt.suptitle(title, fontsize=15)
    if filename:
        plt.savefig(filename, dpi=160)#, bbox_inches="tight")
    if DISPLAY_PLOTS:
        plt.show()
    plt.close()

# ---------------------------------------------------------------------------
# globals: timestamp & plot directory ---------------------------------------
TIMESTAMP = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
PLOT_DIR = os.path.join("plots", TIMESTAMP)
os.makedirs(PLOT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# helpers: file parsing, sol-loader, repair, pretty-print --------------------
# ---------------------------------------------------------------------------
def parse_vrp_file(vrp_path: str) -> Tuple[int | None, int, int,
                                           Dict[int, Tuple[float, float]],
                                           Dict[int, int], int]:
    """Read TSPLIB-style CVRP → (minVehHint, cap, nNodes, coords, demands, depot)."""
    coords, demands = {}, {}
    min_veh_hint = cap = n_nodes = None
    depot_id: int | None = None
    section = None
    with open(vrp_path) as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("COMMENT") and "Min no of trucks:" in line:
                try:
                    min_veh_hint = int(line.split("Min no of trucks:")[1].split(",")[0])
                except ValueError:
                    pass
            elif line.startswith("CAPACITY"):
                cap = int(line.split()[-1])
            elif line.startswith("DIMENSION"):
                n_nodes = int(line.split()[-1])
            elif line == "NODE_COORD_SECTION":
                section = "COORD"; continue
            elif line == "DEMAND_SECTION":
                section = "DEMAND"; continue
            elif line == "DEPOT_SECTION":
                section = "DEPOT"; continue
            elif line == "EOF":
                section = None; continue

            if section == "COORD":
                node, xs, ys = line.split()
                coords[int(node)] = (float(xs), float(ys))
            elif section == "DEMAND":
                node, d = line.split()
                demands[int(node)] = int(d)
            elif section == "DEPOT":
                if line != "-1":
                    depot_id = int(line)
                else:
                    section = None

    if None in (cap, n_nodes, depot_id):
        raise ValueError(f"'{vrp_path}' missing CAPACITY / DIMENSION / DEPOT.")
    print(f"[INFO] parsed '{os.path.basename(vrp_path)}' → cap={cap}, nodes={n_nodes}, "
          f"depot={depot_id}, minVeh={min_veh_hint}")
    return min_veh_hint, cap, n_nodes, coords, demands, depot_id


def load_reference_solution(vrp_path: str,
                            coords: Dict[int, Tuple[float, float]] | None,
                            depot: int = 1):
    """Return (routes, cost) from companion .sol if present."""
    sol_path = vrp_path.replace(".vrp", ".sol")
    if not os.path.exists(sol_path):
        return None, None
    pat = re.compile(r"Route\s*(?:#?\d+)?\s*:\s*(.*)", re.IGNORECASE)
    routes, cost_val = [], None
    with open(sol_path) as fh:
        for ln in fh:
            m = pat.search(ln)
            if m:
                # 1) parse the node IDs, 2) increment each by 1
                nums = [int(tok) + 1 for tok in m.group(1).split()]
                routes.append(nums)
            elif ln.lower().startswith("cost"):
                try:
                    cost_val = float(ln.split()[1])
                except (IndexError, ValueError):
                    pass
    # — now routes is e.g. [[10,8,6,3,2,7], …]

    # add depot boundaries --------------------------------------------------
    fixed: List[List[int]] = []
    for r in routes:
        if not r:
            continue
        if r[0] != depot:
            r = [depot] + r
        if r[-1] != depot:
            r.append(depot)
        fixed.append(r)

    if cost_val is None and coords is not None:
        cost_val = sum(cost_function(r, coords) for r in fixed)

    cstr = f"{cost_val:.1f}" if cost_val else "n/a"
    print(f"[INFO] reference .sol found – {len(fixed)} routes, cost={cstr}")
    return fixed, cost_val


def repair_routes(raw: List[List[int]], demands: Dict[int, int],
                  cap: int, depot: int = 1) -> List[List[int]]:
    """Ensure every customer appears once & capacity respected."""
    customers = {n for n in demands if n != depot}
    visited = set(); final: List[List[int]] = []
    for rr in raw:
        if not rr:
            continue
        r = rr[:]
        if r[0] != depot:
            r.insert(0, depot)
        if r[-1] != depot:
            r.append(depot)
        cleaned = [depot] + [n for n in r[1:-1] if n not in visited] + [depot]
        visited.update(cleaned)
        if len(cleaned) > 2:
            final.append(cleaned)

    missing = customers - visited
    for m in missing:
        final.append([depot, m, depot])
    return final


def print_cvrp_solution(routes: List[List[int]], total_cost: float):
    print("[RESULT] solution:")
    for i, rt in enumerate(routes, 1):
        print(f"  route {i}: {' → '.join(map(str, rt))}")
    print(f"  total cost: {total_cost:.2f}")

# ---------------------------------------------------------------------------
# tiny helper for master-grid cells -----------------------------------------
# ---------------------------------------------------------------------------
def draw_routes_basic(ax, coords, routes, depot: int, title: str,
                      cost: float | None):
    """
    tiny overview cell: coloured solution + dashed optimal.
    now also labels every stop with its node ID on top of the line.
    """
    cmap = plt.get_cmap("tab20")

    # draw each route
    for i, rt in enumerate(routes, 1):
        col = cmap(i % 20)
        xs = [coords[n][0] for n in rt]
        ys = [coords[n][1] for n in rt]
        ax.plot(xs, ys, "-o",
                lw=2.2,
                markersize=4,
                markeredgecolor="k",
                c=col,
                alpha=.8,
                label=f"r{i}")

        # **new**: label each customer node on top of the line
        for n in rt:
            x, y = coords[n]
            txt = ax.text(
                x, y, str(n),
                fontsize=6,
                ha="center", va="center",
                color="black",
                zorder=6
            )
            # give it a white outline so it never disappears
            txt.set_path_effects([
                PathEffects.Stroke(linewidth=1.5, foreground="white"),
                PathEffects.Normal()
            ])

    # draw depot star
    dx, dy = coords[depot]
    ax.scatter([dx], [dy], marker="*", s=70, c="red",
               edgecolors="k", linewidth=.6, label="depot", zorder=7)

    # final cosmetics
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    if cost is not None:
        ax.set_title(f"{title}\ncost={cost:.1f}", fontsize=9)
    else:
        ax.set_title(f"{title}", fontsize=9)
    ax.legend(fontsize=7, loc="upper right", framealpha=.95)


# ---------------------------------------------------------------------------
# MAIN ----------------------------------------------------------------------
# ---------------------------------------------------------------------------
def main():
    # ----------------------------------------------------------------------
    # gather problems



    search_dirs =["./problems/beginner","./problems/intermediate","./problems/advanced"][0:1]      # extend as required
    search_dirs = ["./problems/beginner"]  # for testing, use only one dir





    vrp_files: List[str] = []
    for d in search_dirs:
         if os.path.isdir(d):
             vrp_files.extend(sorted(glob.glob(os.path.join(d, "*.vrp"))))
    if not vrp_files:
        print("[ERROR] no .vrp problems found."); sys.exit(1)







    # algorithms to run
    methods = ["BB"]#["ALNS","GA","BB"]          # add "MSH", "ILS", "ALNS", "BB" freely




    # containers -----------------------------------------------------------
    sols: Dict[str, List[Tuple]] = {m: [None]*len(vrp_files) for m in methods}
    costs: Dict[str, List[float]] = {m: [math.inf]*len(vrp_files) for m in methods}
    metrics: Dict[str, List[dict]] = {m: [None]*len(vrp_files) for m in methods}
    runtimes: Dict[str, List[float]] = {m: [None]*len(vrp_files) for m in methods}
    ref_routes: List[List[List[int]] | None] = [None]*len(vrp_files)
    ref_costs : List[float | None]          = [None]*len(vrp_files)
    depots    : List[int]                   = [1]*len(vrp_files)

    # ----------------------------------------------------------------------
    # batch loop
    for p_idx, vrp_path in enumerate(vrp_files):
        
        # if p_idx != 4:
        #     continue
        pname = os.path.basename(vrp_path)
        print(f"\n[INFO] ===== problem {pname} =====")
        min_hint, cap, n_nodes, coords, demands, depot = parse_vrp_file(vrp_path)
        depots[p_idx] = depot

        opt_routes, opt_cost = load_reference_solution(vrp_path, coords, depot=depot)
        ref_routes[p_idx] = opt_routes
        ref_costs [p_idx] = opt_cost

        for method in methods:
            print(f"  [INFO] --- {method} ---")
            t0 = time.time()

            if method == "MSH":
                raw_routes, _ = MSH_alg(coords, demands, cap, n_nodes)
                met = {}
            elif method == "ILS":
                
                if n_nodes<= 30:
                    max_searches = 3000  # Default number of searches for ILS and MSH
                    time_limit = 800  # Default time limit in seconds
                elif n_nodes <=80:
                    max_searches = 9000
                    time_limit = 1800 #1200  # Default time limit in seconds
                else:
                    max_searches = 20000
                    time_limit = 3600  # Default time limit in seconds
                
                raw_routes, _ , metrics_dict = ILS(coords, demands, cap, n_nodes, "parallel_hybrid_ils",time_limit,max_searches )
                
                met = {}
            elif method == "GA":
                GA_with_islands.__ga_demands  = demands
                GA_with_islands.__ga_capacity = cap
                n_veh = min_hint or math.ceil(sum(demands.values())/cap)
                raw_routes, _, met = ga_island_model_solver(coords, demands, cap, n_veh)
            elif method == "ALNS":
                raw_routes, _, met = alns_metaheuristic_solver(coords, demands, cap, n_nodes)
            elif method == "BB":
                raw_routes, _, met = branch_and_bound_lds_solver(coords, demands, cap, n_nodes)
            else:
                continue

            run_t = time.time() - t0
            print(f"    [INFO] {method} runtime: {run_t:.2f}s")
            fin_routes = repair_routes(raw_routes, demands, cap, depot=depot)
            tot_cost   = sum(cost_function(r, coords) for r in fin_routes)

            print_cvrp_solution(fin_routes, tot_cost)

            # store --------------------------------------------------------
            sols[method][p_idx] = (coords, fin_routes, tot_cost, demands, cap)
            costs[method][p_idx] = tot_cost
            metrics[method][p_idx] = met
            runtimes[method][p_idx] = run_t

            # side-by-side plot -------------------------------------------
            sol_d = dict(coords=coords, routes=fin_routes, cost=tot_cost)
            ref_d = dict(coords=coords,
                         routes=opt_routes or [],
                         cost=opt_cost or 0)
            fn = os.path.join(PLOT_DIR, f"{method}_{pname}_routes_{TIMESTAMP}{IMAGE_EXT}")
            plot_routes(ref_d, sol_d, fn, f"{method} – {pname}", depot=depot)

            # convergence
            
            if method == "ILS":
                # show the three metaheuristics together
                # Plot all algorithms on one graph for comparison
                all_algorithms = ["aco", "simulated_annealing", "tabu_search"]
                best_costs_per_algo = {}
                
                # Extract metrics for each algorithm
                for algo in all_algorithms:
                    algo_metrics = metrics_dict.get(algo, {})
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
                                      f"{method}_{pname}_conv_{TIMESTAMP}{IMAGE_EXT}")
                plot_line(xs, best_costs_per_algo, f_line,
                          f"{method} convergence – {pname}", "step", "obj",
                          params_text=f"time: {run_t:.2f}s")
                
                
                
                
            best_series = met.get("best_cost_per_iter") or \
                          met.get("best_fitness_per_gen")
            if best_series:
                xs = list(range(1, len(best_series)+1))
                f_line = os.path.join(PLOT_DIR,
                                      f"{method}_{pname}_conv_{TIMESTAMP}{IMAGE_EXT}")
                plot_line(xs, {"best": best_series}, f_line,
                          f"{method} convergence – {pname}", "step", "obj",
                          params_text=f"time: {run_t:.2f}s")

                worst = met.get("worst_fitness_per_gen")
                mean = met.get("mean_fitness_per_gen")
                if worst and mean:

                    n = len(best_series)
                    group = max(1, int(n * 0.05))
                    grouped = []
                    for i in range(0, n, group):
                        w_seg = worst[i:i+group]
                        b_seg = best_series[i:i+group]
                        if not w_seg or not b_seg:
                            continue
                        grouped.append([min(b_seg), max(w_seg)])
                    if grouped:

                        f_gbox = os.path.join(PLOT_DIR,
                                             f"{method}_{pname}_gbox_{TIMESTAMP}{IMAGE_EXT}")
                        plot_box(grouped, f_gbox,
                                 f"{method} fitness 5% groups – {pname}")
    # ----------------------------------------------------------------------
    # MASTER GRID  –  1 row per problem
    #                col-0 = optimal, cols 1… = one per algorithm
    # ----------------------------------------------------------------------
    n_probs = len(vrp_files)
    n_cols = 1 + len(methods)  # first column = optimal
    fig, axes = plt.subplots(n_probs, n_cols,
                             figsize=(4 * n_cols, 2.8 * n_probs),
                             squeeze=False)


    for p_idx, vrp_path in enumerate(vrp_files):
        prob_name = os.path.basename(vrp_path).replace(".vrp", "")
        depot = depots[p_idx]
        opt_rts = ref_routes[p_idx]
        opt_cost = ref_costs[p_idx]
        coords_any = sols[methods[0]][p_idx][0]  # any method shares same coords

        # --- first column : optimal ---------------------------------------
        ax_opt = axes[p_idx][0]
        if opt_rts:
            draw_routes_basic(ax_opt, coords_any, opt_rts, depot,
                              title=f"{prob_name} – OPT", cost=opt_cost)
        else:
            ax_opt.text(0.5, 0.5, "no .sol", ha="center", va="center",
                        fontsize=9)
            ax_opt.set_xticks([]);
            ax_opt.set_yticks([])

        # --- subsequent columns : algorithm solutions ---------------------
        for c_idx, method in enumerate(methods, start=1):
            ax = axes[p_idx][c_idx]
            coords, rt, cst, demands, cap = sols[method][p_idx]
            draw_routes_basic(ax, coords, rt, depot,
                              title=f"{prob_name} – {method}", cost=cst)

    # cosmetics -------------------------------------------------------------
    plt.tight_layout();
    plt.subplots_adjust(hspace=0.35)
    grid_path = os.path.join(PLOT_DIR, f"master_grid_{TIMESTAMP}{IMAGE_EXT}")
    plt.savefig(grid_path, dpi=160)
    if DISPLAY_PLOTS:
        plt.show();
    plt.close()

    # ------------------------------------------------------------------
    # SUMMARY CSV + console pivot
    # ------------------------------------------------------------------
    rows = []
    for m in methods:
        for p_idx, vrp_path in enumerate(vrp_files):
            rows.append(dict(algo=m,
                             problem=os.path.basename(vrp_path),
                             cost=costs[m][p_idx],
                             runtime=runtimes[m][p_idx]))
    df = pd.DataFrame(rows)
    print("\n[SUMMARY] cost table:")
    print(df.pivot(index="algo", columns="problem", values="cost"))
    print(f"\n[INFO] batch complete – results in '{PLOT_DIR}'")

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
