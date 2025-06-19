# BranchBound_LDS_updated.py
# b&b + lds for cvrp, now with k-means & clarke-wright seeding
# pure python, plugs straight into cvrp_solver_test2.py
import matplotlib.pyplot as plt
import heapq, random, time, math
from itertools import combinations
from utils import (
    distance_function, cost_function,
    validate_routes_capacity, validate_routes_cities,
    nearest_neighbors_CVRP, Ackley_score, generate_initial_solution_ackley
)
import numpy as np
import random


# ─────────── hyper-params (tweakable) ────────────────────────────────────
time_limit_sec          = 1800
max_discrepancies       = 50
knn_k                   = 50
relax_factor_init       = 1.55   # early broad search
relax_factor_final      = 1.15   # tighten near timeout
num_restarts            = 20
seed_attempts           = 20
kmeans_iter             = 40
weight_distance         = 10.0
weight_balance          = 0.05
weight_capacity         = 5.0
# ─────────────────────────────────────────────────────────────────────────

# ─────────── adaptive fitness ───────────────────────────────────────────
def fitness(routes, coords, demands, cap):
    dists = [cost_function(r, coords) for r in routes]
    total = sum(dists)
    cap_pen = sum(max(0, sum(demands.get(n, 0) for n in r) - cap) ** 2 for r in routes)
    bal = 0
    if len(dists) > 1:
        m = total / len(dists)
        bal = sum((d - m) ** 2 for d in dists) / len(dists)
    return weight_distance * total + weight_capacity * cap_pen + weight_balance * bal

# ─────────── 2-opt + greedy merge polishing ─────────────────────────────
def two_opt(rt, coords):
    best, improved = rt[:], True
    while improved:
        improved = False
        for i in range(1, len(best) - 2):
            for j in range(i + 2, len(best) - 1):
                cand = best[:i] + best[i:j][::-1] + best[j:]
                if cost_function(cand, coords) < cost_function(best, coords):
                    best, improved = cand, True
                    break
            if improved:
                break
    return best

def polish(routes, coords, demands, cap):
    routes = [two_opt(r, coords) for r in routes]
    merged = True
    while merged and len(routes) > 1:
        merged = False
        best_gain, pair = 0, None
        for i, j in combinations(range(len(routes)), 2):
            load = sum(demands.get(n, 0) for n in routes[i] + routes[j])
            if load > cap:
                continue
            merged_rt = two_opt(routes[i][:-1] + routes[j][1:], coords)
            gain = (cost_function(routes[i], coords) +
                    cost_function(routes[j], coords) -
                    cost_function(merged_rt, coords))
            if gain > best_gain:
                best_gain, pair = gain, (i, j, merged_rt)
        if pair:
            i, j, new_rt = pair
            # pop larger index first
            if i < j:
                routes.pop(j)
                routes.pop(i)
            else:
                routes.pop(i)
                routes.pop(j)
            routes.append(new_rt)
            merged = True

    # ensure depot at both ends
    for idx, r in enumerate(routes):
        if r and r[0] != 1:
            r = [1] + r
        if r and r[-1] != 1:
            r = r + [1]
        routes[idx] = r
    return routes

# ─────────── simple k-means partition heuristic ─────────────────────────
def kmeans_seed(coords, demands, cap, rng):
    custs = [n for n in coords if n != 1]
    tot_demand = sum(demands[c] for c in custs)
    k = max(1, math.ceil(tot_demand / cap))
    # if fewer customers than k, reduce k
    k = min(k, len(custs))
    centroids = rng.sample(custs, k)
    for _ in range(kmeans_iter):
        buckets = {c: [] for c in centroids}
        # assign each customer to nearest centroid
        for c in custs:
            cx, cy = coords[c]
            nearest = min(centroids, key=lambda v: distance_function(*(coords[v] + (cx, cy))))
            buckets[nearest].append(c)
        new_centroids = []
        for pts in buckets.values():
            if not pts:
                continue
            mx = sum(coords[p][0] for p in pts) / len(pts)
            my = sum(coords[p][1] for p in pts) / len(pts)
            # pick the actual customer closest to the mean
            best = min(pts, key=lambda v: distance_function(*(coords[v] + (mx, my))))
            new_centroids.append(best)
        if set(new_centroids) == set(centroids):
            break
        centroids = new_centroids

    routes = []
    for cluster in buckets.values():
        cluster = cluster[:]  # copy list
        r = [1]
        load = 0
        curr = 1
        while cluster:
            nxt = min(cluster, key=lambda v: distance_function(*(coords[curr] + coords[v])))
            if load + demands[nxt] > cap:
                # close current route
                r.append(1)
                routes.append(r)
                r = [1]
                load = 0
                curr = 1
            else:
                cluster.remove(nxt)
                r.append(nxt)
                load += demands[nxt]
                curr = nxt
        r.append(1)
        routes.append(r)
    return routes

# ─────────── clarke-wright savings heuristic ────────────────────────────
def clarke_wright_seed(coords, demands, cap):
    custs = [n for n in coords if n != 1]
    routes = {c: [1, c, 1] for c in custs}
    load   = {c: demands[c] for c in custs}
    savings = []
    for i, j in combinations(custs, 2):
        d1i = distance_function(*(coords[1] + coords[i]))
        d1j = distance_function(*(coords[1] + coords[j]))
        dij = distance_function(*(coords[i] + coords[j]))
        savings.append((d1i + d1j - dij, i, j))
    savings.sort(reverse=True)

    for s, i, j in savings:
        ri = routes.get(i)
        rj = routes.get(j)
        if ri is None or rj is None or ri is rj:
            continue
        # i must be at end of ri, j at start of rj
        if ri[-2] != i or rj[1] != j:
            continue
        if load[i] + load[j] > cap:
            continue
        merged = ri[:-1] + rj[1:]
        # update routes and loads
        for node in merged:
            routes[node] = merged
        load[i] = load[j] = load[i] + load[j]

    # collect unique route objects
    seen = set()
    result = []
    for r in routes.values():
        if id(r) not in seen:
            seen.add(id(r))
            result.append(r)
    return result

# ─────────── lb functions ───────────────────────────────────────────────
def capacity_lb(unassigned, demands, cap):
    total = sum(demands[c] for c in unassigned)
    return max(0, math.ceil(total / cap) * 2)

def knn_lb(routes, unassigned, coords, demands, cap, nbrs):
    base_cost = sum(cost_function(r, coords) for r in routes)
    extra = 0
    for c in unassigned:
        # find nearest assigned node or depot
        candidates = [1] + [n for r in routes for n in r if n != 1]
        near = min(candidates, key=lambda v: distance_function(*(coords[c] + coords[v])))
        extra += distance_function(*(coords[c] + coords[near])) * 2
    return base_cost + max(extra, capacity_lb(unassigned, demands, cap))

# ─────────── node class ─────────────────────────────────────────────────
class Node:
    __slots__ = ('routes', 'unassigned', 'lb', 'disc', 'depth')
    def __init__(self, routes, unassigned, lb, disc, depth):
        self.routes = routes
        self.unassigned = unassigned
        self.lb = lb
        self.disc = disc
        self.depth = depth
    def __lt__(self, other):
        return (self.lb, -self.depth) < (other.lb, -other.depth)

# ─────────── adaptive insertion list ────────────────────────────────────
def expand(nd, coords, demands, cap, nbrs, max_disc, rng):
    # choose next customer = farthest from depot for diversity
    c = max(nd.unassigned, key=lambda n: distance_function(*(coords[1] + coords[n])))
    rest = nd.unassigned - {c}
    children = []
    insert_ops = []
    for idx, r in enumerate(nd.routes):
        load = sum(demands.get(n, 0) for n in r)
        if load + demands[c] > cap:
            continue
        best_delta = float('inf')
        best_pos = None
        for i in range(len(r) - 1):
            delta = (distance_function(*(coords[r[i]] + coords[c])) +
                     distance_function(*(coords[c] + coords[r[i+1]])) -
                     distance_function(*(coords[r[i]] + coords[r[i+1]])))
            if delta < best_delta:
                best_delta = delta
                best_pos = i + 1
        if best_pos is not None:
            insert_ops.append((best_delta, idx, best_pos))
    # keep top 50% of insertions for exploration
    insert_ops.sort(key=lambda x: x[0])
    cutoff = max(1, len(insert_ops) // 2)
    chosen = insert_ops[:cutoff]
    rng.shuffle(chosen)
    for _, idx, pos in chosen:
        new_rt = nd.routes[idx][:pos] + [c] + nd.routes[idx][pos:]
        new_routes = nd.routes[:idx] + [new_rt] + nd.routes[idx+1:]
        lb = knn_lb(new_routes, rest, coords, demands, cap, nbrs)
        children.append(Node(new_routes, rest, lb, nd.disc, nd.depth + 1))

    # open new route if allowed
    if nd.disc < max_disc:
        new_routes = nd.routes + [[1, c, 1]]
        lb = knn_lb(new_routes, rest, coords, demands, cap, nbrs)
        children.append(Node(new_routes, rest, lb, nd.disc + 1, nd.depth + 1))

    return children

# ─────────── seeding pool ───────────────────────────────────────────────
def seeds(coords, demands, cap, num_nodes, rng):
    sol_set = []
    # 1) clarke-wright
    sol_set.append(clarke_wright_seed(coords, demands, cap))
    # 2) k-means × seed_attempts
    for _ in range(seed_attempts):
        sol_set.append(kmeans_seed(coords, demands, cap, rng))
    # 3) random nearest-neighbor sweep × seed_attempts
    custs = [n for n in coords if n != 1]
    for _ in range(seed_attempts):
        rng.shuffle(custs)
        routes = []
        r = [1]
        load = 0
        for c in custs:
            if load + demands[c] > cap:
                r.append(1)
                routes.append(r)
                r = [1]
                load = 0
            r.append(c)
            load += demands[c]
        r.append(1)
        routes.append(r)
        sol_set.append(routes)

    return sol_set

def get_knn_k()->int:
    return knn_k


# ─────────── main solver ────────────────────────────────────────────────
def branch_and_bound_lds_solver(coords, demands, capacity, num_nodes,time_limit_sec):
    """branch & bound with limited discrepancy search

    :returns: (routes, cost, metrics) where metrics contain best value per node,
              bound improvement and time per expansion
    """
    t0 = time.time()
    #default to 5 minutes if user didnt pass time limit
    if time_limit_sec is None:
        time_limit_sec = 300
    customers = set(coords) - {1}
    coords_list=[coords[i] for i in sorted(coords) if i != 1]
    max_k=len(coords_list)
    knn_k= min(get_knn_k(), max_k)
    nbrs = nearest_neighbors_CVRP(coords, knn_k)
    best_routes, best_fit = None, float('inf')
    rng_master = random.Random(42)
    best_per_node = []
    bound_improve = []
    time_per_step = []
    relax_history = []
    pq_size_history = []
    disc_history = []

    expansions = 0
    print (f"\nStarting B&B + LDS with {num_nodes} nodes and a time limit of {time_limit_sec} seconds\n")
    for restart in range(num_restarts):
        rng = random.Random(rng_master.randint(0, 1_000_000_000))

        # heuristic seeding
        for s in seeds(coords, demands, capacity, num_nodes, rng):
            polished = polish(s, coords, demands, capacity)
            f = fitness(polished, coords, demands, capacity)
            if f < best_fit:
                best_routes, best_fit = polished, f

        # b&b + lds
        for max_disc in range(max_discrepancies + 1):
            root_lb = knn_lb([], customers, coords, demands, capacity, nbrs)
            pq = [(root_lb, Node([], customers, root_lb, 0, 0))]

            while pq and (time.time() - t0) < time_limit_sec:
                step_t = time.time()
                elapsed = time.time() - t0
                # linearly decay relax factor from init → final
                relax = (relax_factor_init -
                         (relax_factor_init - relax_factor_final) * (elapsed / time_limit_sec))
                #pop ze next node
                _, nd = heapq.heappop(pq)
                #record stuff
                relax_history.append(relax)
                pq_size_history.append(len(pq))
                disc_history.append(nd.disc)
                expansions += 1
                if expansions%2000 == 0:
                    total_elapsed = time.time() - t0
                    print(f'[Branch and Bound with LDS] | nodes expanded: {expansions} | best fit: {best_fit:.2f} | time elapsed: {total_elapsed:.2f}s')
                if nd.lb >= best_fit * relax:
                    best_per_node.append(best_fit)
                    bound_improve.append(0.0)
                    time_per_step.append(time.time() - step_t)
                    continue

                if not nd.unassigned:
                    if (validate_routes_capacity(nd.routes, coords, demands, capacity) and
                        validate_routes_cities(nd.routes, coords, demands, capacity)):
                        cand = polish(nd.routes, coords, demands, capacity)
                        f = fitness(cand, coords, demands, capacity)
                        if f < best_fit:
                            bound_improve.append(best_fit - f)
                            best_routes, best_fit = cand, f
                        else:
                            bound_improve.append(0.0)
                        best_per_node.append(best_fit)
                        time_per_step.append(time.time() - step_t)
                    continue

                for child in expand(nd, coords, demands, capacity, nbrs, max_disc, rng):
                    if child.lb < best_fit * relax:
                        heapq.heappush(pq, (child.lb, child))

                best_per_node.append(best_fit)
                bound_improve.append(0.0)
                time_per_step.append(time.time() - step_t)

            if (time.time() - t0) >= time_limit_sec:
                break  # time’s up

    metrics = {
        'best_value_per_sample': best_per_node,
        'bounds_improvement': bound_improve,
        'computation_time_per_step': time_per_step
    }
    exp_idx = list(range(1, len(disc_history) + 1))
    mark_every = max(1, int(len(exp_idx) * 0.05))  # one marker every 5%

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    fig.subplots_adjust(hspace=0.4)

    # top subplot: LDS discrepancy & pruning relaxation
    ax1.plot(
        exp_idx,
        disc_history,
        marker='o',
        markevery=mark_every,
        linestyle='--',
        linewidth=1,
        color='tab:blue',
        label='LDS Discrepancy'
    )
    ax1.set_ylabel('discrepancies')
    ax1.set_title('Discrepancy & Pruning Relaxation')
    # annotate LDS discrepancy values
    for x, y in zip(exp_idx[::mark_every], disc_history[::mark_every]):
        ax1.text(x, y, str(y),
                 fontsize=12, fontweight='bold',
                 va='bottom', ha='center')

    ax1b = ax1.twinx()
    ax1b.plot(
        exp_idx,
        relax_history,
        marker='^',
        markevery=mark_every,
        linestyle='-.',
        linewidth=1,
        color='tab:red',
        label='Relaxation Factor'
    )
    ax1b.set_ylabel('relax factor')
    # annotate relaxation factor values
    for x, y in zip(exp_idx[::mark_every], relax_history[::mark_every]):
        ax1b.text(x, y, f"{y:.2f}",
                  fontsize=12, fontweight='bold',
                  va='bottom', ha='center')

    # combined legend for top plot
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1b.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='best')

    # bottom subplot: priority‐queue size
    ax2.plot(
        exp_idx,
        pq_size_history,
        marker='s',
        markevery=mark_every,
        linestyle='-',
        linewidth=1,
        color='tab:green',
        label='PQ Size'
    )
    ax2.set_xlabel('expansion #')
    ax2.set_ylabel('queue size')
    ax2.set_title('Priority Queue Size Over Expansions')
    # annotate PQ size values
    for x, y in zip(exp_idx[::mark_every], pq_size_history[::mark_every]):
        ax2.text(x, y, str(y),
                 fontsize=12, fontweight='bold',
                 va='bottom', ha='center')

    ax2.legend(loc='best')

    plt.tight_layout()
    plt.show()

    return best_routes or [], best_fit, metrics

# ───────── Ackley Hyperparameters ─────────────────────────────────────────
ACKLEY_SAMPLES           = 10000   # max number of box‐center evaluations
ACKLEY_INIT_RANDOM       = 100     # pure‐random probes to seed best_val
ACKLEY_MAX_DISCREPANCIES = 5       # allowed LDS detours

# ───────── Imports (if not already) ───────────────────────────────────────
import math, heapq, random, time
import numpy as np
from utils import Ackley_score, generate_initial_solution_ackley

# ───────── BoxNode for Ackley B&B ─────────────────────────────────────────
class BoxNode:
    __slots__ = ("ell","u","lb","disc","depth")
    def __init__(self, ell, u, lb, disc, depth):
        self.ell, self.u, self.lb, self.disc, self.depth = ell[:], u[:], lb, disc, depth
    def __lt__(self, other):
        if self.lb != other.lb:
            return self.lb < other.lb
        if self.disc != other.disc:
            return self.disc < other.disc
        return self.depth > other.depth

# ───────── Analytic Lower Bound on Ackley in a Box ─────────────────────────
def compute_box_lower_bound(ell: list[float], u: list[float]) -> float:
    d = len(ell)
    # 1) quadratic term
    S_min = 0.0
    for i in range(d):
        if ell[i] > 0:
            x = ell[i]
        elif u[i] < 0:
            x = u[i]
        else:
            x = 0.0
        S_min += x*x
    R = math.sqrt(S_min / d)
    term1 = -20.0 * math.exp(-0.2 * R)
    # 2) cosine term
    S_max = 0.0
    for i in range(d):
        width = u[i] - ell[i]
        if width >= 1.0 and math.floor(u[i]) >= math.ceil(ell[i]):
            S_max += 1.0
        else:
            S_max += max(math.cos(2*math.pi*ell[i]),
                         math.cos(2*math.pi*u[i]))
    term2 = -math.exp(S_max / d)
    return term1 + term2 + 20.0 + math.e

# ───────── B&B + LDS Ackley Solver ────────────────────────────────────────
def bb_ackley_solver(
    dim: int = 10,
    lower: float = -32.768,
    upper: float = 32.768,
    samples: int = ACKLEY_SAMPLES,
    early_stop_tol: float | None = None,       # now accepted
    max_discrepancies: int = ACKLEY_MAX_DISCREPANCIES,
    init_random: int = ACKLEY_INIT_RANDOM,
    seed: int | None = None
) -> tuple[np.ndarray, float]:
    """branch-and-bound + LDS for Ackley

    :returns: (vector, value, metrics) with best value per sample,
              bound improvements and step durations
    """
    # reproducibility
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # 1) INITIAL RANDOM PROBES
    best_val = float("inf")
    best_pt = np.zeros(dim)
    expansions = 0

    best_value_per_sample = []
    bounds_improvement = []
    time_per_step = []
    for _ in range(init_random):
        t0 = time.time()
        x = generate_initial_solution_ackley(None, dim=dim,
                                             lower_bound=lower,
                                             upper_bound=upper)
        v = Ackley_score(x)
        expansions += 1
        prev = best_val
        if v < best_val:
            best_val, best_pt = v, x.copy()
        best_value_per_sample.append(best_val)
        bounds_improvement.append(prev - best_val)
        time_per_step.append(time.time() - t0)
        if early_stop_tol is not None and best_val <= early_stop_tol:
            metrics = {
                'best_value_per_sample': best_value_per_sample,
                'bounds_improvement': bounds_improvement,
                'computation_time_per_step': time_per_step
            }
            return best_pt, best_val, metrics

    # 2) BUILD ROOT BOX
    root_ell = [lower]*dim
    root_u   = [upper]*dim
    root_lb  = compute_box_lower_bound(root_ell, root_u)
    root     = BoxNode(root_ell, root_u, root_lb, disc=0, depth=0)

    pq = [(root.lb, root)]
    heapq.heapify(pq)

    # 3) MAIN B&B + LDS LOOP
    while expansions < samples and pq:
        t0 = time.time()
        lb_top, node = heapq.heappop(pq)
        expansions += 1
        prev = best_val

        # prune
        if lb_top >= best_val:
            continue

        # evaluate center (skip depth=0 if you want)
        if node.depth > 0:
            center = np.array([(node.ell[i] + node.u[i]) / 2.0
                               for i in range(dim)])
            v_c = Ackley_score(center)
            if v_c < best_val:
                best_val, best_pt = v_c, center.copy()
            if early_stop_tol is not None and best_val <= early_stop_tol:
                metrics = {
                    'best_value_per_sample': best_value_per_sample,
                    'bounds_improvement': bounds_improvement,
                    'computation_time_per_step': time_per_step
                }
                return best_pt, best_val, metrics

        # LDS cutoff
        if node.disc >= max_discrepancies:
            continue

        # split on the two largest widths
        widths = [node.u[i] - node.ell[i] for i in range(dim)]
        for rank, sd in enumerate(sorted(range(dim),
                                         key=lambda i: widths[i],
                                         reverse=True)[:2]):
            mid = 0.5 * (node.ell[sd] + node.u[sd])

            # lower‐half child
            ell_lo, u_lo = node.ell[:], node.u[:]
            u_lo[sd] = mid
            lb_lo    = compute_box_lower_bound(ell_lo, u_lo)
            disc_lo  = node.disc + (1 if rank == 1 else 0)
            if disc_lo <= max_discrepancies and lb_lo < best_val:
                heapq.heappush(pq, (
                    lb_lo,
                    BoxNode(ell_lo, u_lo, lb_lo, disc_lo, node.depth+1)
                ))

            # upper‐half child
            ell_hi, u_hi = node.ell[:], node.u[:]
            ell_hi[sd] = mid
            lb_hi    = compute_box_lower_bound(ell_hi, u_hi)
            disc_hi  = node.disc + (1 if rank == 1 else 0)
            if disc_hi <= max_discrepancies and lb_hi < best_val:
                heapq.heappush(pq, (
                    lb_hi,
                    BoxNode(ell_hi, u_hi, lb_hi, disc_hi, node.depth+1)
                ))

        best_value_per_sample.append(best_val)
        bounds_improvement.append(prev - best_val)
        time_per_step.append(time.time() - t0)


    metrics = {
        'best_value_per_sample': best_value_per_sample,
        'bounds_improvement': bounds_improvement,
        'computation_time_per_step': time_per_step
    }
    return best_pt, best_val, metrics




