
# ga_with_islands.py
# --------------------------------------------------------------------
# island genetic algorithm for CVRP – enhanced with clustering, aging, and adaptive mutation
# depends on utils.py + std-lib + scikit-learn (for KMeans)
# integrates useful techniques from dtsp_ga_final_engine.py and bin_packing_ga.py
# comments are lowercase, concise, explanatory
# --------------------------------------------------------------------

import random
import math
import matplotlib.pyplot as plt
import copy
import itertools
import time
from typing import Dict, Tuple, List

import numpy as np
from sklearn.cluster import KMeans  # for k-means seeding

from utils import (
    cost_function,
    distance_function,
    validate_routes_capacity,
    Ackley_score,
    generate_initial_solution_ackley,
    validate_routes_cities,
    nearest_neighbors_CVRP,
    most_connected_node,
)

# ─── tunables ──────────────────────────────────────────────────────────────
POP_SIZE = 50  # individuals per island
ISLANDS = 8  # number of separate sub-populations
MIGRANTS = 6  # how many elites migrate per interval
MIGRATE_EVERY = 6  # every this many generations, do migration
GENS = 100  # total generations to run
BASE_MUT = 0.30  # initial mutation probability
BASE_TWOOPT = 0.50  # initial two-opt probability
FLOOR_MUT = 0.05  # minimum mutation probability after decay
FLOOR_TWOOPT = 0.05  # minimum two-opt prob after decay
DECAY_RATE = 0.06  # exp decay rate for both mut and two-opt
HYPER_MUT = 0.80  # hypermutation probability (when triggered)
HYPER_TWOOPT = 0.80  # hyper-twoopt probability (when triggered)
HYPER_LEN = 5  # how many generations hypermode lasts
STAG_THRES = 6  # gens w/out improvement before hypermode
NOVELTY_K = 5  # k for novelty (genome distance)
NOVELTY_W = 0.15  # weight for novelty in combined fitness
W_DIST, W_CAP, W_BAL = 1.0, 5.0, 0.1  # adaptive-fitness weights
CROSS_MODE = "scx"  # {"scx","cyclic","order"} crossover mode
AGE_THRESHOLD = 20  # max age before reinitialization
random.seed()  # seed rng from system time

# ─── module globals (set at run-time) ──────────────────────────────────────
coords_g: Dict[int, Tuple[float, float]] = {}
demands_g: Dict[int, int] = {}
cap_g: int = 0
dist_matrix: Dict[int, Dict[int, float]] = {}  # cache for pairwise distances


# ─── helper: build distance matrix ─────────────────────────────────────────
def build_dist_matrix(coords: Dict[int, Tuple[float, float]]) -> None:
    """
    precompute euclidean distances between all pairs of nodes for o(1) lookup
    """
    global dist_matrix
    dist_matrix = {}
    items = list(coords.items())
    for ni, (xi, yi) in items:
        dist_matrix[ni] = {}
    for ni, (xi, yi) in items:
        for nj, (xj, yj) in items:
            dx = xi - xj
            dy = yi - yj
            dist_matrix[ni][nj] = math.hypot(dx, dy)


# ─── local cost function using dist_matrix ─────────────────────────────────
def local_cost(rt: List[int]) -> float:
    """
    sum of distances along route using dist_matrix
    """
    total = 0.0
    for a, b in zip(rt[:-1], rt[1:]):
        total += dist_matrix[a][b]
    return total


# ─── helper: cool-down curve ───────────────────────────────────────────────
def cool(base: float, low: float, r: float, g: int) -> float:
    """
    exponential decay from base -> low over generations g with rate r
    """
    return low + (base - low) * math.exp(-r * g)


# ─── adaptive fitness + novelty ───────────────────────────────────────────
def route_fitness(routes: List[List[int]],
                  demands: Dict[int, int],
                  cap: int) -> float:
    """
    compute fitness = W_DIST * total_distance
                    + W_CAP  * sum_squared_capacity_violations
                    + W_BAL  * variance_of_route_distances
    lower is better. does not yet include novelty.
    """
    dists = [local_cost(rt) for rt in routes]
    tot = sum(dists)
    cap_pen = 0.0
    for rt in routes:
        load = sum(demands.get(n, 0) for n in rt if n != 1)
        if load > cap:
            cap_pen += (load - cap) ** 2
    bal_pen = 0.0
    if len(dists) > 1:
        μ = tot / len(dists)
        bal_pen = sum((d - μ) ** 2 for d in dists) / len(dists)
    return W_DIST * tot + W_CAP * cap_pen + W_BAL * bal_pen


def edge_set(routes: List[List[int]]) -> set:
    e = set()
    for rt in routes:
        for a, b in zip(rt[:-1], rt[1:]):
            e.add((a, b))
    return e


def genome_dist(ind_a, ind_b) -> int:
    """
    distance between two individuals = size of symmetric difference
    of their edge sets (routes are lists of ints).
    """
    es1 = edge_set(ind_a[2])
    es2 = edge_set(ind_b[2])
    return len(es1.symmetric_difference(es2))


# ─── 2-opt (adjacent-skip) ─────────────────────────────────────────────────
def two_opt(rt: List[int]) -> List[int]:
    """
    classic 2-opt on a single route: try reversing any non-adjacent segment
    until no further improvement.
    """
    best = rt[:]
    improve = True
    while improve:
        improve = False
        n = len(best)
        for i in range(1, n - 2):
            for j in range(i + 2, n - 1):
                a, b = best[i - 1], best[i]
                c, d = best[j], best[j + 1]
                delta = (dist_matrix[a][c] + dist_matrix[b][d]
                         - dist_matrix[a][b] - dist_matrix[c][d])
                if delta < -1e-6:
                    best = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                    improve = True
                    break
            if improve:
                break
    return best


# ─── swap* neighborhood (single-pass) ──────────────────────────────────────
def swap_star_once(routes: List[List[int]],
                   demands: Dict[int, int],
                   cap: int) -> Tuple[List[List[int]], bool]:
    """
    single-pass SWAP*: check all route pairs for a beneficial swap, apply best once.
    returns (new_routes, improved_flag).
    """
    best_gain = 0.0
    best_pair = None
    orig_cost = sum(local_cost(rt) for rt in routes)
    for i, j in itertools.combinations(range(len(routes)), 2):
        r1, r2 = routes[i], routes[j]
        if len(r1) <= 3 or len(r2) <= 3:
            continue
        load1_orig = sum(demands.get(n, 0) for n in r1 if n != 1)
        load2_orig = sum(demands.get(n, 0) for n in r2 if n != 1)
        for p1 in range(1, len(r1) - 1):
            for p2 in range(1, len(r2) - 1):
                c1, c2 = r1[p1], r2[p2]
                load1 = load1_orig - demands[c1] + demands[c2]
                load2 = load2_orig - demands[c2] + demands[c1]
                if load1 > cap or load2 > cap:
                    continue
                tmp1 = r1[:];
                tmp2 = r2[:]
                tmp1[p1], tmp2[p2] = c2, c1
                tmp1_opt = two_opt(tmp1) if len(tmp1) > 3 else tmp1
                tmp2_opt = two_opt(tmp2) if len(tmp2) > 3 else tmp2
                cand_rts = []
                for k, rt in enumerate(routes):
                    if k == i:
                        cand_rts.append(tmp1_opt)
                    elif k == j:
                        cand_rts.append(tmp2_opt)
                    else:
                        cand_rts.append(rt)
                cand_cost = sum(local_cost(rt) for rt in cand_rts)
                gain = orig_cost - cand_cost
                if gain > best_gain:
                    best_gain = gain
                    best_pair = (i, j, tmp1_opt, tmp2_opt)
    if best_pair:
        i, j, new1, new2 = best_pair
        updated = [rt for rt in routes]
        updated[i] = new1
        updated[j] = new2
        return updated, True
    return routes, False


# ─── full polish: swap* once, 2-opt on each route, greedy merge ────────────
def full_polish(rts: List[List[int]],
                demands: Dict[int, int],
                cap: int) -> List[List[int]]:
    """
    full polish used for elites and seeding:
      - apply single-pass swap*
      - apply 2-opt to each route
      - greedy merge of routes if improvement
    """
    curr, _ = swap_star_once(rts, demands, cap)
    curr = [two_opt(rt) if len(rt) > 3 else rt for rt in curr]
    merged = True
    while merged and len(curr) > 1:
        merged = False
        best_gain = 0.0
        pair = None
        orig_costs = {idx: local_cost(rt) for idx, rt in enumerate(curr)}
        for i, j in itertools.combinations(range(len(curr)), 2):
            r1, r2 = curr[i], curr[j]
            m = r1[:-1] + r2[1:]
            load = sum(demands.get(n, 0) for n in m if n != 1)
            if load > cap:
                continue
            m_opt = two_opt(m) if len(m) > 3 else m
            gain = orig_costs[i] + orig_costs[j] - local_cost(m_opt)
            if gain > best_gain:
                best_gain = gain
                pair = (i, j, m_opt)
        if pair:
            i, j, new_rt = pair
            if i < j:
                curr.pop(j);
                curr.pop(i)
            else:
                curr.pop(i);
                curr.pop(j)
            curr.append(new_rt)
            merged = True
    for idx, rt in enumerate(curr):
        if rt and rt[0] != 1:
            rt.insert(0, 1)
        if rt and rt[-1] != 1:
            rt.append(1)
        curr[idx] = rt
    return curr


# ─── light polish: only 2-opt on each route ─────────────────────────────────
def light_polish(rts: List[List[int]]) -> List[List[int]]:
    """
    lighter polish for offspring: only apply 2-opt per route (no swap*, no merge).
    """
    return [two_opt(rt) if len(rt) > 3 else rt for rt in rts]


# ─── flatten helpers ───────────────────────────────────────────────────────
def flatten(rts: List[List[int]]) -> Tuple[List[int], List[int]]:
    """
    convert list-of-routes into (flatperm, lengths_per_route).
    flatperm = concatenated lists of all customers (excluding depot markers).
    lens = number of customers per route.
    """
    flat = []
    lens = []
    for rt in rts:
        seq = rt[1:-1]  # strip leading/trailing 1
        flat.extend(seq)
        lens.append(len(seq))
    return flat, lens


def unflatten(flat: List[int], lens: List[int]) -> List[List[int]]:
    """
    rebuild routes from a flat list + lens. each route = [1] + segment + [1].
    """
    rec = []
    idx = 0
    for ln in lens:
        rec.append([1] + flat[idx:idx + ln] + [1])
        idx += ln
    return rec


# ─── seed pop (k-means + heuristic + full polish) ──────────────────────────
def seed_pop(coords: Dict[int, Tuple[float, float]],
             demands: Dict[int, int],
             cap: int,
             k: int) -> List[Tuple[List[int], List[int], List[List[int]], int]]:
    """
    build an initial population of size k for one island:
      - cluster customers into k_clusters ~ num_vehicles via KMeans
      - for each cluster, build a route via nearest-neighbor / greedy
      - full polish final routes
    each individual = (flatperm, lens, routes, age=0).
    """
    custs = [nid for nid in coords.keys() if nid != 1]
    if not custs:
        return [([], [], [[1, 1]], 0) for _ in range(k)]  # degenerate
    tot_demand = sum(demands[c] for c in custs)
    nveh_est = max(1, math.ceil(tot_demand / cap))
    # prepare data for KMeans: ignore depot
    data = np.array([coords[c] for c in custs])
    # if fewer customers than kmeans clusters, fallback to random
    if len(custs) < nveh_est:
        labels = np.zeros(len(custs), dtype=int)
    else:
        km = KMeans(n_clusters=nveh_est, random_state=random.getrandbits(32))
        labels = km.fit_predict(data)
    pop = []
    for _ in range(k):
        routes = []
        # for each cluster, build a route
        for cluster_id in range(nveh_est):
            cluster_nodes = [custs[i] for i in range(len(custs)) if labels[i] == cluster_id]
            if not cluster_nodes:
                continue
            # nearest-neighbor from depot on this cluster
            unvisited = set(cluster_nodes)
            cur = 1
            route = [1]
            load = 0
            while unvisited:
                # find nearest feasible in cluster
                cand = sorted(unvisited, key=lambda x: dist_matrix[cur][x])
                found = False
                for nxt in cand:
                    if load + demands[nxt] <= cap:
                        route.append(nxt)
                        load += demands[nxt]
                        cur = nxt
                        unvisited.remove(nxt)
                        found = True
                        break
                if not found:
                    break
            route.append(1)
            routes.append(route)
        # any leftover customers from clustering go to own route
        assigned = {n for rt in routes for n in rt if n != 1}
        leftovers = [c for c in custs if c not in assigned]
        for c in leftovers:
            routes.append([1, c, 1])
        # full polish and store
        routes = [rt for rt in routes if len(rt) > 2]
        routes = full_polish(routes, demands, cap)
        flat, lens = flatten(routes)
        pop.append((flat, lens, routes, 0))  # age=0
    return pop


# ─── rank-based stochastic selection ────────────────────────────────────────
def rank_pick(pop: List, fits: List[float]) -> int:
    """
    pick an index from pop by stochastic rank selection:
    sort individuals by ascending fitness, assign exp-prob proportional to rank.
    """
    ranks = sorted(range(len(pop)), key=lambda i: fits[i])  # best at rank 0
    τ = 1.7
    prob = [math.exp(-τ * r / len(pop)) for r in range(len(pop))]
    norm = sum(prob)
    prob = [p / norm for p in prob]
    return random.choices(ranks, weights=prob, k=1)[0]


# ─── scx crossover (capacity-aware) ────────────────────────────────────────
def next_in_parent(parent_routes: List[List[int]], cur: int) -> int:
    """
    given a flattened route list, find the successor of 'cur' in that parent's
    route structure (wrap-around to the depot).
    """
    for rt in parent_routes:
        if cur in rt:
            idx = rt.index(cur)
            return rt[(idx + 1) % (len(rt) - 1)]
    return None


def scx(parent1: Tuple[List[int], List[int], List[List[int]], int],
        parent2: Tuple[List[int], List[int], List[List[int]], int],
        demands: Dict[int, int],
        cap: int) -> List[List[int]]:
    """
    sequential constructive crossover (SCX) for CVRP:
      - start at depot=1, pick next customer by looking at successor in each parent
        (choose whichever is closer and fits capacity), otherwise nearest feasible.
    uses dist_matrix for distance comparisons.
    returns a new list-of-routes (not yet flattened).
    """
    custs = set(parent1[0])
    visited = set()
    child = []
    load = 0
    cur = 1  # start at depot
    while len(visited) < len(custs):
        n1 = next_in_parent(parent1[2], cur)
        n2 = next_in_parent(parent2[2], cur)
        candidates = [n for n in (n1, n2) if n not in visited and n is not None]
        if not candidates:
            left = [c for c in custs if c not in visited]
            candidates = sorted(left, key=lambda x: dist_matrix[cur][x])
        picked = None
        for nxt in candidates:
            if load + demands[nxt] <= cap:
                picked = nxt
                break
        if picked is None:
            child.append(1)
            load = 0
            leftover = [c for c in custs if c not in visited]
            candidates = sorted(leftover, key=lambda x: dist_matrix[cur][x])
            picked = candidates[0]
        child.append(picked)
        visited.add(picked)
        load += demands[picked]
        cur = picked
    child.append(1)
    # rebuild routes
    routes = []
    tmp = [1]
    for c in child[1:]:
        if c == 1:
            if len(tmp) > 1:
                tmp.append(1)
                routes.append(tmp)
            tmp = [1]
        else:
            tmp.append(c)
    return routes


def cyclic_xo(p1: List[int], p2: List[int]) -> List[int]:
    """
    cyclic crossover for permutations. if sets differ, return p1 copy.
    """
    if set(p1) != set(p2):
        return p1[:]
    size = len(p1)
    kid = [None] * size
    idx = 0
    seen = set()
    while idx not in seen:
        kid[idx] = p1[idx]
        seen.add(idx)
        idx = p1.index(p2[idx])
    for i in range(size):
        if kid[i] is None:
            kid[i] = p2[i]
    return kid


def order_xo(p1: List[int], p2: List[int]) -> List[int]:
    """
    order crossover (OX1) for permutations.
    """
    size = len(p1)
    a, b = sorted(random.sample(range(size), 2))
    kid = [None] * size
    kid[a:b + 1] = p1[a:b + 1]
    fill = [g for g in p2 if g not in kid]
    j = 0
    for i in range(size):
        if kid[i] is None:
            kid[i] = fill[j]
            j += 1
    return kid


# ─── mutation ops & adaptive roulette ──────────────────────────────────────
def swap(routes: List[List[int]], dem: Dict[int, int], cap: int) -> List[List[int]]:
    """
    swap two customers (not depot) between two routes (or within same route) if capacity allows.
    """
    s = copy.deepcopy(routes)
    pos = [(i, p) for i, rt in enumerate(s) for p in range(1, len(rt) - 1)]
    if len(pos) < 2:
        return routes
    (i1, p1), (i2, p2) = random.sample(pos, 2)
    if i1 == i2:
        return routes
    c1, c2 = s[i1][p1], s[i2][p2]

    def load(rt: List[int]) -> int:
        return sum(dem.get(n, 0) for n in rt if n != 1)

    new_load1 = load(s[i1]) - dem[c1] + dem[c2]
    new_load2 = load(s[i2]) - dem[c2] + dem[c1]
    if new_load1 <= cap and new_load2 <= cap:
        s[i1][p1], s[i2][p2] = c2, c1
    return s


def relocate(routes: List[List[int]],
             dem: Dict[int, int],
             cap: int) -> List[List[int]]:
    """
    remove one random customer from a random non-trivial route, and insert it at best position.
    """
    s = copy.deepcopy(routes)
    donors = [rt for rt in s if len(rt) > 3]
    if not donors:
        return routes
    src = random.choice(donors)
    pos = random.randrange(1, len(src) - 1)
    cust = src.pop(pos)
    if len(src) <= 2:
        s.remove(src)
    best = None
    bestΔ = float('inf')
    for idx, rt in enumerate(s):
        load_rt = sum(dem.get(n, 0) for n in rt if n != 1)
        if load_rt + dem[cust] > cap:
            continue
        for ins in range(1, len(rt)):
            prev, nxt = rt[ins - 1], rt[ins]
            Δ = (dist_matrix[prev][cust] +
                 dist_matrix[cust][nxt] -
                 dist_matrix[prev][nxt])
            if Δ < bestΔ:
                bestΔ = Δ
                best = (idx, ins)
    if best:
        s[best[0]].insert(best[1], cust)
    else:
        s.append([1, cust, 1])
    return s


def inversion(routes: List[List[int]],
              dem: Dict[int, int],
              cap: int) -> List[List[int]]:
    """
    pick a random route with length >4, reverse a random subsequence inside it.
    """
    s = copy.deepcopy(routes)
    rich = [rt for rt in s if len(rt) > 4]
    if not rich:
        return routes
    rt = random.choice(rich)
    a, b = sorted(random.sample(range(1, len(rt) - 1), 2))
    rt[a:b] = reversed(rt[a:b])
    return s


def cross_exchange(routes: List[List[int]],
                   dem: Dict[int, int],
                   cap: int) -> List[List[int]]:
    """
    pick two routes that each have length>5, exchange random subsequences if capacity allows.
    """
    s = copy.deepcopy(routes)
    big = [rt for rt in s if len(rt) > 5]
    if len(big) < 2:
        return routes
    r1, r2 = random.sample(big, 2)
    a, b = sorted(random.sample(range(1, len(r1) - 1), 2))
    c, d = sorted(random.sample(range(1, len(r2) - 1), 2))
    seg1, seg2 = r1[a:b], r2[c:d]
    new1 = r1[:a] + seg2 + r1[b:]
    new2 = r2[:c] + seg1 + r2[d:]

    def load(rt: List[int]) -> int:
        return sum(dem.get(n, 0) for n in rt if n != 1)

    if load(new1) <= cap and load(new2) <= cap:
        idx1, idx2 = s.index(r1), s.index(r2)
        s[idx1], s[idx2] = new1, new2
    return s


MUT_OPS = [swap, relocate, inversion, cross_exchange]
OP_SCORES = [1, 1, 1, 1]  # track operator success counts


def mutate(ind: Tuple[List[int], List[int], List[List[int]], int],
           demands: Dict[int, int],
           cap: int,
           pmut: float,
           ptwo: float,
           mean_fit: float) -> Tuple[List[int], List[int], List[List[int]], int]:
    """
    adaptive mutation:
      - scale mutation prob by (fitness/mean_fit) capped at 2.0
      - choose one operator from MUT_OPS by weighted roulette (OP_SCORES)
      - apply it, reward if improvement
      - sprinkle 2-opt on each route with prob ptwo
      - apply full_polish with small prob (~0.1) for intensification
      - aging: if age > AGE_THRESHOLD, reinitialize that individual
      - return new individual (flat, lens, routes, age+1) if feasible
      - else return original with age reset to 0
    """
    flat, lens, routes, age = ind
    # aging: if too old, reinitialize
    if age > AGE_THRESHOLD:
        # reinitialize this individual via seeding
        new_indivs = seed_pop(coords_g, demands, cap, 1)[0]
        return new_indivs  # age resets to 0
    fit = route_fitness(routes, demands, cap)
    scale = min(2.0, fit / mean_fit) if mean_fit > 0 else 1.0
    p = max(FLOOR_MUT, min(1.0, pmut * scale))
    child = copy.deepcopy(routes)
    if random.random() < p:
        tot = sum(OP_SCORES)
        w = [s / tot for s in OP_SCORES]
        op_idx = random.choices(range(len(MUT_OPS)), weights=w, k=1)[0]
        before = route_fitness(child, demands, cap)
        op = MUT_OPS[op_idx]
        child = op(child, demands, cap) if op is relocate else op(child, demands, cap)
        after = route_fitness(child, demands, cap)
        if after < before:
            OP_SCORES[op_idx] += 1
    if random.random() < ptwo:
        for i, rt in enumerate(child):
            if len(rt) > 3 and random.random() < ptwo:
                child[i] = two_opt(rt)
    # occasionally run full_polish for deeper local search
    if random.random() < 0.10:
        child = full_polish(child, demands, cap)
    else:
        child = light_polish(child)
    if validate_routes_capacity(child, coords_g, demands, cap) and \
            validate_routes_cities(child, coords_g, demands, cap):
        f2, l2 = flatten(child)
        return (f2, l2, child, age + 1)
    return (flat, lens, routes, 0)


# ─── main solver ──────────────────────────────────────────────────────────
def ga_island_model_solver(coords: Dict[int, Tuple[float, float]],
                           demands: Dict[int, int],
                           cap: int,
                           num_vehicles: int | None = None,
                           time_limit: float | None = None) -> Tuple[
    List[List[int]], float, dict]:
    """solve CVRP via multi-island GA

    :returns: (routes, cost, metrics) where metrics include best/mean/worst
              fitness per generation, diversity and time per generation
    """
    global coords_g, demands_g, cap_g
    coords_g, demands_g, cap_g = coords, demands, cap
    build_dist_matrix(coords)  # precompute distances

    # seed initial population (same for all islands)
    pop0 = seed_pop(coords, demands, cap, POP_SIZE)
    islands = [copy.deepcopy(pop0) for _ in range(ISLANDS)]

    best_routes = None
    best_cost = float('inf')
    stagn = 0  # generations without improvement
    hyper = 0  # hypermode counter

    # metrics init
    best_fitness_per_gen = []
    mean_fitness_per_gen = []
    worst_fitness_per_gen = []
    pmut_history = []
    ptwo_history = []
    stagn_history = []
    op_score_history = []
    hyper_gen_history = []
    diversity_per_gen = []
    time_per_gen = []
    print(
        f"\nStarting GA with {ISLANDS} islands, total {POP_SIZE} per island, {GENS} gens, {BASE_MUT} base mutation rate, {CROSS_MODE} crossover\n")
    t0 = time.time()
    for gen in range(GENS):
        t_start = time.time()
        pmut = HYPER_MUT if hyper > 0 else cool(BASE_MUT, FLOOR_MUT, DECAY_RATE, gen)
        ptwo = HYPER_TWOOPT if hyper > 0 else cool(BASE_TWOOPT, FLOOR_TWOOPT, DECAY_RATE, gen)
        pmut_history.append(pmut)
        ptwo_history.append(ptwo)  # keep track of these two
        if hyper > 0:
            hyper -= 1
        total_elapsed = time.time() - t0
        print(
            f"[ga] | gen {gen + 1:02}/{GENS} |  mut={pmut:.2f} | twoopt={ptwo:.2f} | hyper={'yes' if hyper > 0 else 'no'} | elapsed time = {total_elapsed:.2f}s ")

        improved = False
        # evolve each island separately
        for idx, pop in enumerate(islands):
            # evaluate each individual's fitness and novelty
            fits = [route_fitness(ind[2], demands, cap) for ind in pop]
            nov = []
            for i in range(len(pop)):
                dists = sorted(genome_dist(pop[i], pop[j]) for j in range(len(pop)) if j != i)
                nov.append(sum(dists[:NOVELTY_K]) / NOVELTY_K if dists else 0.0)
            adj = [fits[i] - NOVELTY_W * nov[i] for i in range(len(pop))]
            # select best by adjusted fitness
            best_i = min(range(len(pop)), key=lambda i: adj[i])
            best_ind = pop[best_i]
            best_ind_cost = sum(local_cost(rt) for rt in best_ind[2])
            if best_ind_cost < best_cost:
                best_cost = best_ind_cost
                best_routes = copy.deepcopy(best_ind[2])
                improved = True

            mean_fit = sum(adj) / len(adj) if adj else 1.0
            next_gen = [best_ind]  # preserve elite

            while len(next_gen) < POP_SIZE:
                p1 = pop[rank_pick(pop, adj)]
                p2 = pop[rank_pick(pop, adj)]
                # crossover
                if CROSS_MODE == "scx":
                    child_routes = scx(p1, p2, demands, cap)
                    if not (validate_routes_capacity(child_routes, coords, demands, cap) and
                            validate_routes_cities(child_routes, coords, demands, cap)):
                        child_routes = copy.deepcopy(p1[2])
                    flat_c, lens_c = flatten(child_routes)
                    child = (flat_c, lens_c, child_routes, 0)
                else:
                    xo = cyclic_xo if CROSS_MODE == "cyclic" else order_xo
                    flat_child = xo(p1[0], p2[0])
                    child_routes = unflatten(flat_child, p1[1])
                    child_routes = light_polish(child_routes)
                    flat_c, lens_c = flatten(child_routes)
                    child = (flat_c, lens_c, child_routes, 0)
                # mutate child (includes aging and adaptive strategies)
                child = mutate(child, demands, cap, pmut, ptwo, mean_fit)
                # deterministic crowding: replace the more similar parent
                cand = [p1, p2]
                sims = [genome_dist(c, child) for c in cand]
                replace_parent = cand[sims.index(min(sims))]
                pop_idx = pop.index(replace_parent)
                pop[pop_idx] = child
                next_gen.append(child)

            islands[idx] = next_gen

        # gather metrics for this generation
        fits_all = [route_fitness(ind[2], demands, cap) for isl in islands for ind in isl]
        best_fitness_per_gen.append(min(fits_all))
        mean_fitness_per_gen.append(sum(fits_all) / len(fits_all))
        worst_fitness_per_gen.append(max(fits_all))
        diversity_per_gen.append(float(np.std(fits_all)))
        time_per_gen.append(time.time() - t_start)

        if improved:
            stagn = 0
        else:
            stagn += 1

        # record stagnation count and total op-score each gen
        stagn_history.append(stagn)
        # if too many gens w/out imp, trigger hypermode and shake-up
        if stagn >= STAG_THRES:
            hyper = HYPER_LEN
            hyper_gen_history.append(gen)
            op_score_history.append(sum(OP_SCORES))
            stagn = 0
            for i in range(ISLANDS):
                island = islands[i]
                fits_i = [route_fitness(ind[2], demands, cap) for ind in island]
                sorted_idx = sorted(range(len(fits_i)), key=lambda x: fits_i[x])
                cutoff = POP_SIZE // 2
                survivors = [island[j] for j in sorted_idx[:cutoff]]
                new_indivs = seed_pop(coords, demands, cap, POP_SIZE - cutoff)
                islands[i] = survivors + new_indivs
        else:
            op_score_history.append(sum(OP_SCORES))
            hyper_gen_history.append(0)
        # migration + random immigrants every MIGRATE_EVERY gens
        if (gen + 1) % MIGRATE_EVERY == 0:
            for i in range(ISLANDS):
                fits_i = [route_fitness(ind[2], demands, cap) for ind in islands[i]]
                elite_idx = sorted(range(len(fits_i)), key=lambda x: fits_i[x])[:MIGRANTS]
                migrants = [copy.deepcopy(islands[i][j]) for j in elite_idx]
                target = (i + 1) % ISLANDS
                imms = seed_pop(coords, demands, cap, MIGRANTS)
                islands[target].extend(migrants)
                islands[target].extend(imms)
                fits_t = [route_fitness(ind[2], demands, cap) for ind in islands[target]]
                sorted_idx = sorted(range(len(fits_t)), key=lambda x: fits_t[x])
                islands[target] = [islands[target][j] for j in sorted_idx[:POP_SIZE]]

        if time_limit is not None and (time.time() - t0) > time_limit:
            print(f"[ga] | time limit reached after {gen + 1} generations, stopping early.")
            break

    metrics = {
        'best_fitness_per_gen': best_fitness_per_gen,
        'mean_fitness_per_gen': mean_fitness_per_gen,
        'worst_fitness_per_gen': worst_fitness_per_gen,
        'diversity_per_gen': diversity_per_gen,
        'computation_time_per_step': time_per_gen
    }
    actual_gens = len(pmut_history)
    gens = list(range(1, actual_gens + 1))
    import matplotlib.pyplot as plt

    # one figure with two stacked plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # top plot: mutation rates, 2-opt, stagnation
    ax1.plot(gens, pmut_history, label='mutation rate')
    ax1.plot(gens, ptwo_history, label='2-opt prob')
    ax1.plot(gens, stagn_history, label='stagnation count')
    ax1.set_ylabel('value')
    ax1.set_title('GA Parameters per Generation')
    ax1.legend(loc='upper right')

    # bottom plot: op score + diversity
    ax2.plot(gens, op_score_history, marker='o', linestyle='--', label='total op-score')
    ax2.plot(gens, diversity_per_gen, label='diversity')
    ax2.set_xlabel('generation')
    ax2.set_ylabel('value')
    ax2.set_title('Operator Score and Diversity')
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

    return best_routes or [], best_cost, metrics


import numpy as np
import random
import copy
import itertools

from utils import (
    # … your existing CVRP imports …,
    Ackley_score,
    generate_initial_solution_ackley
)

# ─── (all existing CVRP code here) ────────────────────────────────────────────


# ─── Updated Ackley-mode GA (islands style) ─────────────────────────────────
# def ga_ackley_solver(
#     dim=10,
#     lower=-32.768,
#     upper=32.768,
#     pop_size=POP_SIZE,
#     gens=GENS,
#     mut_rate=BASE_MUT,
#     seed=None
# ):
#     """GA islands variant for Ackley
#
#     :returns: (vector, value, metrics) with best/mean/worst fitness per gen,
#               diversity and step times
#     """
#
#     # Optional seeding for reproducibility
#     if seed is not None:
#         random.seed(seed)
#         np.random.seed(seed)
#
#     # 1) Initialize a population of pop_size individuals in [lower,upper]^dim
#     population = [
#         generate_initial_solution_ackley(None, dim=dim, lower_bound=lower, upper_bound=upper)
#         for _ in range(pop_size)
#     ]
#     # 2) Evaluate all on Ackley
#     fitnesses = [Ackley_score(ind) for ind in population]
#
#     best_fitness_per_gen = []
#     mean_fitness_per_gen = []
#     worst_fitness_per_gen = []
#     pmut_history = []
#     ptwo_history = []
#     stagn_history = []
#     op_score_history = []
#     diversity_per_gen = []
#     time_per_gen = []
#
#     for generation in range(gens):
#         t_start = time.time()
#         new_pop = []
#         new_fit = []
#
#         while len(new_pop) < pop_size:
#             # ── 2-way tournament selection ──────────────────────────────────
#             i1, i2 = random.sample(range(pop_size), 2)
#             parent1 = population[i1] if fitnesses[i1] < fitnesses[i2] else population[i2]
#             j1, j2 = random.sample(range(pop_size), 2)
#             parent2 = population[j1] if fitnesses[j1] < fitnesses[j2] else population[j2]
#
#             # ── Blend crossover (NumPy arrays) ─────────────────────────────
#             α = random.random()
#             child = α * parent1 + (1.0 - α) * parent2  # still a NumPy array
#
#             # ── Gaussian mutation on each coordinate ─────────────────────
#             for k in range(dim):
#                 if random.random() < mut_rate:
#                     σ = (upper - lower) * 0.1
#                     child[k] += random.gauss(0, σ)
#                     # clamp within bounds
#                     child[k] = max(lower, min(upper, child[k]))
#
#             val = Ackley_score(child)
#             new_pop.append(child)
#             new_fit.append(val)
#
#         # ── Merge “parents + offspring”, then keep only best pop_size ──────
#         population += new_pop
#         fitnesses += new_fit
#         paired = sorted(zip(fitnesses, population), key=lambda item: item[0])
#         population = [ind for (_, ind) in paired[:pop_size]]
#         fitnesses = [fval for (fval, _) in paired[:pop_size]]
#
#         best_fitness_per_gen.append(min(fitnesses))
#         mean_fitness_per_gen.append(sum(fitnesses)/len(fitnesses))
#         worst_fitness_per_gen.append(max(fitnesses))
#         diversity_per_gen.append(float(np.std(fitnesses)))
#         time_per_gen.append(time.time() - t_start)
#
#     # Return best individual (NumPy array) + its fitness
#     best_idx = fitnesses.index(min(fitnesses))
#     metrics = {
#         'best_fitness_per_gen': best_fitness_per_gen,
#         'mean_fitness_per_gen': mean_fitness_per_gen,
#         'worst_fitness_per_gen': worst_fitness_per_gen,
#         'diversity_per_gen': diversity_per_gen,
#         'computation_time_per_step': time_per_gen
#     }
#
#     return population[best_idx], fitnesses[best_idx], metrics

import numpy as np
import random
import time
import matplotlib.pyplot as plt

from utils import Ackley_score, generate_initial_solution_ackley


def ga_ackley_solver(
        dim=10,
        lower=-32.768,
        upper=32.768,
        pop_size=150,
        gens=100,
        mut_rate=0.30,
        seed=None
):
    """GA islands variant for Ackley

    :returns: (vector, value, metrics) with metrics including:
              - best, mean, worst fitness per generation
              - diversity per generation
              - computation time per generation
    Also plots all metrics over generations on completion.
    """
    # reproducibility
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # initialize population
    population = [
        generate_initial_solution_ackley(None, dim=dim, lower_bound=lower, upper_bound=upper)
        for _ in range(pop_size)
    ]
    fitnesses = [Ackley_score(ind) for ind in population]

    # metric histories
    best_fitness_per_gen = []
    mean_fitness_per_gen = []
    worst_fitness_per_gen = []
    diversity_per_gen = []
    time_per_gen = []

    # main GA loop
    for generation in range(gens):
        t_start = time.time()
        new_pop = []
        new_fit = []

        while len(new_pop) < pop_size:
            # tournament selection
            i1, i2 = random.sample(range(pop_size), 2)
            parent1 = population[i1] if fitnesses[i1] < fitnesses[i2] else population[i2]
            j1, j2 = random.sample(range(pop_size), 2)
            parent2 = population[j1] if fitnesses[j1] < fitnesses[j2] else population[j2]

            # blend crossover
            alpha = random.random()
            child = alpha * parent1 + (1.0 - alpha) * parent2

            # gaussian mutation
            for k in range(dim):
                if random.random() < mut_rate:
                    sigma = (upper - lower) * 0.1
                    child[k] += random.gauss(0, sigma)
                    child[k] = max(lower, min(upper, child[k]))

            val = Ackley_score(child)
            new_pop.append(child)
            new_fit.append(val)

        # survivor selection
        population += new_pop
        fitnesses += new_fit
        paired = sorted(zip(fitnesses, population), key=lambda x: x[0])
        population = [ind for (_, ind) in paired[:pop_size]]
        fitnesses = [f for (f, _) in paired[:pop_size]]

        # record metrics
        best_fitness_per_gen.append(min(fitnesses))
        mean_fitness_per_gen.append(sum(fitnesses) / len(fitnesses))
        worst_fitness_per_gen.append(max(fitnesses))
        diversity_per_gen.append(float(np.std(fitnesses)))
        time_per_gen.append(time.time() - t_start)

    # prepare metrics dict
    metrics = {
        'best_fitness_per_gen': best_fitness_per_gen,
        'mean_fitness_per_gen': mean_fitness_per_gen,
        'worst_fitness_per_gen': worst_fitness_per_gen,
        'diversity_per_gen': diversity_per_gen,
        'computation_time_per_step': time_per_gen
    }

    # plot all tracked metrics
    gens_list = list(range(1, gens + 1))
    plt.figure(figsize=(10, 6))
    plt.plot(gens_list, best_fitness_per_gen, label='best fitness')
    plt.plot(gens_list, mean_fitness_per_gen, label='mean fitness')
    plt.plot(gens_list, worst_fitness_per_gen, label='worst fitness')
    plt.plot(gens_list, diversity_per_gen, label='diversity')
    plt.plot(gens_list, time_per_gen, label='step time')
    plt.xlabel('generation')
    plt.ylabel('value')
    plt.title('Ackley GA: metrics over generations')
    plt.legend()
    plt.show()

    # return best individual, its value, and all metrics
    best_idx = fitnesses.index(min(fitnesses))
    return population[best_idx], fitnesses[best_idx], metrics