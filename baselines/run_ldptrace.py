from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common

REPO = common.SRC / "ldptrace" / "repo"
sys.path.insert(0, str(REPO / "LDPTrace" / "code"))

import numpy as np

import ldp
import map_func
import trajectory
from grid import Grid, GridMap

COUNTERS = {"non_adjacent_moves": 0}


def convert_raw_to_grid(raw_trajectories, grid_map, interp=True):
    return [trajectory.trajectory_point2grid(t, grid_map, interp) for t in raw_trajectories]


def estimate_max_length(grid_db, epsilon, grid_map, max_len_q):
    ldp_server = ldp.OUEServer(epsilon, grid_map.size, lambda x: x - 1)
    ldp_client = ldp.OUEClient(epsilon, grid_map.size, lambda x: x - 1)

    for t in grid_db:
        if len(t) > grid_map.size:
            binary_vec = ldp_client.privatise(grid_map.size)
        else:
            binary_vec = ldp_client.privatise(len(t))
        ldp_server.aggregate(binary_vec)

    ldp_server.adjust()
    sum_count = np.sum(ldp_server.adjusted_data)
    count = 0
    quantile = len(ldp_server.adjusted_data)
    for i in range(len(ldp_server.adjusted_data)):
        count += ldp_server.adjusted_data[i]
        if count >= max_len_q * sum_count:
            quantile = i + 1
            break

    return ldp_server, quantile


def update_markov_prob(grid_db, epsilon, grid_map, max_len=36):
    ldp_server = ldp.OUEServer(epsilon / (max_len + 1), grid_map.size * 8, lambda x: x)
    ldp_client = ldp.OUEClient(epsilon / (max_len + 1), grid_map.size * 8, lambda x: x)
    start_server = ldp.OUEServer(epsilon / (max_len + 1), grid_map.size,
                                 lambda x: map_func.grid_index_map_func(x, grid_map))
    start_client = ldp.OUEClient(epsilon / (max_len + 1), grid_map.size,
                                 lambda x: map_func.grid_index_map_func(x, grid_map))
    end_server = ldp.OUEServer(epsilon / (max_len + 1), grid_map.size,
                               lambda x: map_func.grid_index_map_func(x, grid_map))
    end_client = ldp.OUEClient(epsilon / (max_len + 1), grid_map.size,
                               lambda x: map_func.grid_index_map_func(x, grid_map))

    for t in grid_db:
        length = min(len(t), max_len)
        start = t[0]
        binary_vec = start_client.privatise(start)
        start_server.aggregate(binary_vec)
        for i in range(length - 1):
            curr_grid = t[i]
            next_grid = t[i + 1]
            if grid_map.is_adjacent_grids(curr_grid, next_grid):
                map_id = map_func.adjacent_pair_grid_map_func((curr_grid, next_grid), grid_map)
                binary_vec = ldp_client.privatise(map_id)
                ldp_server.aggregate(binary_vec)
            else:
                COUNTERS["non_adjacent_moves"] += 1
        end = t[length - 1]
        binary_vec = end_client.privatise(end)
        end_server.aggregate(binary_vec)

    ldp_server.adjust()
    start_server.adjust()
    end_server.adjust()
    return ldp_server, start_server, end_server


def generate_markov_matrix(markov_vec, start_vec, end_vec, grid_map):
    n = grid_map.size + 1
    markov_mat = np.zeros((n, n), dtype=float)
    for k in range(8 * grid_map.size):
        if markov_vec[k] <= 0:
            continue
        g1, g2 = map_func.adjacent_pair_grid_inv_func(k, grid_map)
        if g2 is None:
            continue
        i = map_func.grid_index_map_func(g1, grid_map)
        j = map_func.grid_index_map_func(g2, grid_map)
        markov_mat[i][j] = markov_vec[k]

    for i in range(len(start_vec)):
        if start_vec[i] < 0:
            start_vec[i] = 0
        if end_vec[i] < 0:
            end_vec[i] = 0
    markov_mat[-1, :-1] = start_vec
    markov_mat[:-1, -1] = end_vec

    markov_mat = markov_mat / (markov_mat.sum(axis=1).reshape((-1, 1)) + 1e-8)
    return markov_mat


def sample_start_point(markov_mat, grid_map):
    prob = markov_mat[-1]
    sample_id = np.random.choice(np.arange(grid_map.size), p=prob[:-1])
    return map_func.grid_index_inv_func(sample_id, grid_map)


def sample_length(length_dis):
    prob = length_dis / np.sum(length_dis)
    length = np.random.choice(np.arange(len(length_dis)), p=prob)
    return length + 1


def sample_markov_next(one_level_mat, prev_grid, length, grid_map):
    candidates = grid_map.get_adjacent(prev_grid)
    candidate_probabilities = np.zeros(len(candidates) + 1, dtype=float)
    for k, (i, j) in enumerate(candidates):
        row = map_func.grid_index_map_func(prev_grid, grid_map)
        col = map_func.grid_index_map_func(grid_map.map[i][j], grid_map)
        prob1 = one_level_mat[row][col]
        if np.isnan(prob1):
            candidate_probabilities[k] = 0
        else:
            candidate_probabilities[k] = prob1
    row = map_func.grid_index_map_func(prev_grid, grid_map)
    col = -1
    prob1 = one_level_mat[row][col]
    prob1 *= min(1.0, 0.3 + (length - 1) * 0.2)
    candidate_probabilities[-1] = prob1
    if candidate_probabilities.sum() < 0.00001:
        return prev_grid
    candidate_probabilities = candidate_probabilities / candidate_probabilities.sum()
    sample_id = np.random.choice(np.arange(len(candidate_probabilities)), p=candidate_probabilities)
    if sample_id == len(candidate_probabilities) - 1:
        return prev_grid
    i, j = candidates[sample_id]
    return grid_map.map[i][j]


def generate_synthetic_database(length_dis, markov_mat, size, grid_map):
    for i in range(len(length_dis)):
        if length_dis[i] < 0:
            length_dis[i] = 0
    synthetic_db = list()
    for i in range(size):
        start_grid = sample_start_point(markov_mat, grid_map)
        length = sample_length(length_dis)
        syn_trajectory = [start_grid]
        for j in range(1, length):
            prev_grid = syn_trajectory[j - 1]
            next_grid = sample_markov_next(markov_mat, prev_grid, len(syn_trajectory), grid_map)
            if next_grid.equal(prev_grid):
                break
            syn_trajectory.append(next_grid)
        synthetic_db.append(syn_trajectory)
    return synthetic_db


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_common_args(ap, default_length_mode="native", default_venue_prior="uniform")
    ap.add_argument("--grid_num", type=int, default=6, help="N for the N x N grid (official default/recommended: 6)")
    ap.add_argument("--max_len_q", type=float, default=0.9, help="quantile of the estimated length (official: 0.9)")
    args = ap.parse_args()
    t_start = time.time()
    s_aux, s_np, s_py, s_map, s_prior = common.seeds(args.seed, 5)
    rng_aux, rng_map = np.random.default_rng(s_aux), np.random.default_rng(s_map)
    np.random.seed(s_np % (2 ** 32))
    random.seed(s_py)

    ds = common.load_dataset(args.dataset, args.user_level, args.user_seed)
    n_out = args.n_out
    bud = common.split_budget(args.eps, args.aux_frac, use_len_hist=args.length_mode == "hist",
                               use_venue_prior=args.venue_prior == "dp")
    common.log(f"LDPTrace {args.dataset}: eps={args.eps} (local DP), user_level={args.user_level}, budget={bud}")
    aux = common.AuxModel.fit_ldp(ds, bud["eps_start"], bud["eps_gap"], bud["eps_len"], rng_aux)
    prior = common.make_venue_prior(args, ds, bud, np.random.default_rng(s_prior), local=True)

    b = ds.bbox
    grid_map = GridMap(args.grid_num, b["lon_min"], b["lat_min"], b["lon_max"], b["lat_max"])
    db = [[(float(x), float(y)) for x, y in t] for t in ds.trails]
    t0 = time.time()
    grid_trajectories = convert_raw_to_grid(db, grid_map)
    glen = np.array([len(t) for t in grid_trajectories])
    common.log(f"grid trajectories: mean len {glen.mean():.2f}, single-cell {np.mean(glen == 1):.3f}, "
               f"{time.time() - t0:.1f}s")
    eps_gen = bud["eps_gen"]
    length_server, quantile = estimate_max_length(grid_trajectories, eps_gen / 10, grid_map, args.max_len_q)
    common.log(f"LDP length quantile ({args.max_len_q}) = {quantile}")
    markov_servers = update_markov_prob(grid_trajectories, 9 * eps_gen / 10, grid_map, max_len=quantile)
    one_level_mat = generate_markov_matrix(markov_servers[0].adjusted_data, markov_servers[1].adjusted_data,
                                           markov_servers[2].adjusted_data, grid_map)
    t0 = time.time()
    syn = generate_synthetic_database(length_server.adjusted_data, one_level_mat, n_out, grid_map)
    t_gen = time.time() - t0
    common.log(f"synthesized {len(syn)} grid trajectories in {t_gen:.1f}s")

    k_raw = np.array([len(t) for t in syn])
    L = common.realize_lengths(k_raw, aux, args.length_mode, rng_map)
    rects = []
    for t, l in zip(syn, L):
        for p in common.fit_path(np.arange(len(t)), l, rng_map):
            g = t[p]
            rects.append((g.min_x, g.max_x, g.min_y, g.max_y))
    vidx, n_empty = common.VenueIndex(ds.pois, ds.bbox).sample_in_rects(np.array(rects), rng_map, prior.weights)
    tmin = aux.sample_times(L, rng_map)
    df = common.build_frame(vidx, L, tmin, ds.pois, prefix="ldptrace")
    runtime = time.time() - t_start
    common.write_outputs(df, Path(args.out), {
        "method": "LDPTrace", "privacy_model": "trail-level local DP (each trail = one user)",
        "source": "https://github.com/zealscott/LDPTrace (-> yuntaod/LDPTrace)", "commit": common.git_commit(REPO),
        "dataset": args.dataset, "eps": args.eps, "delta": 0.0, "seed": args.seed, "budget": bud,
        "user_level": bool(args.user_level), "user_seed": args.user_seed,
        "grid_num": args.grid_num, "max_len_q": args.max_len_q, "ldp_length_quantile": int(quantile),
        "length_mode": args.length_mode, "aux": aux.to_json(), "venue_prior": prior.info,
        "non_adjacent_moves": COUNTERS["non_adjacent_moves"], "generated_cells_mean": float(k_raw.mean()),
        "paths_longer_than_20": int((k_raw > common.LEN_MAX).sum()), "empty_cell_fallbacks": int(n_empty),
        "runtime_sec": runtime, "generation_sec": t_gen})
    common.log(f"done in {runtime:.1f}s")


if __name__ == "__main__":
    main()
