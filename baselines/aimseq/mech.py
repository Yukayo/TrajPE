from __future__ import annotations
import os as _tp_os
def _tp_root():
    p = _tp_os.path.dirname(_tp_os.path.abspath(__file__))
    while not _tp_os.path.isfile(_tp_os.path.join(p, "trajpe", "__init__.py")) and _tp_os.path.dirname(p) != p:
        p = _tp_os.path.dirname(p)
    return p
_TP_ROOT = _tp_os.environ.get("TRAJPE_ROOT", _tp_root())
_TP_WORK = _tp_os.environ.get("TRAJPE_WORK", _tp_os.path.join(_TP_ROOT, "work"))
_TP_PY = _tp_os.environ.get("TRAJPE_PYTHON", "python")
_TP_PY_DPLORA = _tp_os.environ.get("TRAJPE_PYTHON_DPLORA", _TP_PY)
_TP_PY_AIMSEQ = _tp_os.environ.get("TRAJPE_PYTHON_AIMSEQ", _TP_PY)
_TP_PY_GRID = _tp_os.environ.get("TRAJPE_PYTHON_GRID", _TP_PY)

import os
import sys
from pathlib import Path

import numpy as np

PGM_REPO = Path(f"{_TP_WORK}/baselines_src/private-pgm/repo")
for p in (PGM_REPO / "mechanisms", PGM_REPO / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
os.environ.setdefault("MPLBACKEND", "Agg")

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_enable_compilation_cache", False)

import mbi
from mbi import Dataset, Domain, estimation

import aim as _aim
import mst as _mst

from accounting import Ledger


def git_commit() -> str:
    head = PGM_REPO / ".git" / "HEAD"
    try:
        ref = head.read_text().strip()
        if ref.startswith("ref:"):
            return (PGM_REPO / ".git" / ref.split()[1]).read_text().strip()
        return ref
    except OSError:
        return "unknown"


def make_dataset(rec: dict, attrs: dict) -> Dataset:
    dom = Domain(list(attrs), [int(attrs[a]) for a in attrs])
    return Dataset({a: np.asarray(rec[a], np.int64) for a in attrs}, dom)


OFFICIAL_MAX_CELLS = 10000


def official_workload(attrs: dict, degree: int = 2, max_cells: int = OFFICIAL_MAX_CELLS) -> list:
    import itertools
    out = []
    for cl in itertools.combinations(list(attrs), degree):
        if int(np.prod([int(attrs[a]) for a in cl])) <= max_cells:
            out.append((tuple(cl), 1.0))
    return out


def build_workload(attrs: dict, name: str = "seq") -> list:
    if name == "official2":
        return official_workload(attrs, 2)
    return _seq_workload(attrs, name)


def _seq_workload(attrs: dict, name: str = "seq") -> list:
    W = []

    def add(*cl):
        if all(a in attrs for a in cl) and (tuple(cl), 1.0) not in W:
            W.append((tuple(cl), 1.0))

    tod = "hod" if "hod" in attrs else "wh"
    add("dow", "hod")
    for a in (tod, "dow", "sc", "ec", "r1", "d2", "g2"):
        add("L", a)
    add(tod, "r1"), add(tod, "g2"), add(tod, "r2"), add(tod, "g3"), add("dow", "r1")
    add("sc", "ec"), add("sc", "r1"), add("sc", "d2"), add("sc", "b2")
    add("r1", "s1")
    last = None if "L" in attrs else max([int(a[1:]) for a in attrs if a[0] == "r" and a[1:].isdigit()])
    if last is not None and last >= 2:
        add("ec", f"r{last}"), add("ec", f"d{last}")
    for k in range(2, 21):
        add("L", f"r{k}")
        add(f"r{k - 1}", f"r{k}")
        add(f"r{k}", f"s{k}")
        add(f"r{k}", f"d{k}"), add(f"r{k}", f"g{k}")
        add(f"d{k}", f"g{k}"), add(f"d{k}", f"b{k}")
        if k >= 3:
            add(f"d{k - 1}", f"d{k}"), add(f"b{k - 1}", f"b{k}"), add(f"g{k - 1}", f"g{k}")
    if name != "seq":
        raise ValueError(f"unknown workload {name}")
    missing = set(attrs) - {a for cl, _ in W for a in cl}
    for a in sorted(missing):
        W.append(((a,), 1.0))
    return W


def presence_init(attrs: dict) -> list:
    out = []
    for a in attrs:
        k = a[1:]
        if a[0] in "dbgrs" and k.isdigit() and int(k) >= 2:
            pick = {"r": ("L", a), "d": (f"r{k}", a), "g": (f"r{k}", a), "s": (f"r{k}", a), "b": (f"d{k}", a)}[a[0]]
        else:
            pick = {"dow": ("dow", "hod"), "hod": ("hod", "r1"), "wh": ("wh", "r1"), "r1": ("r1", "s1")}.get(a, (a,))
        pick = tuple(x for x in pick if x in attrs) or (a,)
        out.append(pick)
    assert len(out) == len(attrs)
    return out


def _einsum(terms, out):
    ids: dict = {}
    args = []
    for arr, labs in terms:
        args += [arr, [ids.setdefault(lab, len(ids)) for lab in labs]]
    args.append([ids.setdefault(lab, len(ids)) for lab in out])
    return np.einsum(*args)


class FastMarginals:

    def __init__(self, model):
        import networkx as nx
        from mbi import junction_tree, marginal_oracles
        jt = junction_tree.make_junction_tree(model.domain, [tuple(c) for c in model.cliques], connected=True)[0]
        nodes = [tuple(c) for c in jt.nodes]
        bel = marginal_oracles.message_passing_hugin(model.potentials.expand(nodes), float(model.total))
        self.bel = {C: np.asarray(bel[C].values, np.float64) for C in nodes}
        self.of: dict = {}
        for C in nodes:
            for a in C:
                self.of.setdefault(a, []).append(C)
        self.paths = dict(nx.all_pairs_shortest_path(jt))
        self._cache: dict = {}

    def _proj(self, C, labs):
        key = (C, labs)
        if key not in self._cache:
            arr = self.bel[C]
            drop = tuple(i for i, a in enumerate(C) if a not in labs)
            kept = [a for a in C if a in labs]
            arr = arr.sum(axis=drop) if drop else arr
            self._cache[key] = np.transpose(arr, [kept.index(a) for a in labs]) if labs else np.asarray(arr.sum())
        return self._cache[key]

    def marginal(self, cl):
        cl = tuple(cl)
        for C in self.of[cl[0]]:
            if set(cl) <= set(C):
                return self._proj(C, cl).reshape(-1)
        if len(cl) != 2:
            return None
        a, b = cl
        path = min((self.paths[Ca][Cb] for Ca in self.of[a] for Cb in self.of[b]), key=len)
        s1 = tuple(sorted(set(path[0]) & set(path[1])))
        lab = (a,) + s1
        J = self._proj(path[0], lab)
        for k in range(1, len(path)):
            C = path[k]
            sin = tuple(sorted(set(path[k - 1]) & set(C)))
            new = tuple(sorted(set(C) & set(path[k + 1]))) if k + 1 < len(path) else (b,)
            extra = tuple(x for x in new if x not in sin)
            num = self._proj(C, sin + extra)
            den = self._proj(C, sin).reshape(self._proj(C, sin).shape + (1,) * len(extra))
            cond = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
            out = (a,) + new
            J = _einsum([(J, lab), (cond, sin + extra)], out)
            lab = out
        return J.reshape(-1)


def filter_candidates_fast(candidates, model, size_limit):
    from mbi import junction_tree
    ans = {}
    free_cliques = set(_aim.downward_closure(model.cliques))
    for cl in candidates:
        if cl in free_cliques or junction_tree.hypothetical_model_size(
                model.domain, list(model.cliques) + [cl]) <= size_limit:
            ans[cl] = candidates[cl]
    return ans


class LedgerAIM(_aim.AIM):

    def __init__(self, rho: float, prng, rounds=None, max_model_size=80, max_iters=1000, check_rounds=3,
                 check_per_round=4):
        super().__init__(epsilon=1.0, delta=1e-6, prng=prng, rounds=rounds, max_model_size=max_model_size,
                         max_iters=max_iters)
        self.rho = float(rho)
        self.ledger = Ledger()
        self.round = 0
        self.check_rounds, self.check_per_round = check_rounds, check_per_round
        self.check_maxdiff = 0.0
        self.check_n = 0
        self._check_rng = np.random.default_rng(12345)

    def gaussian_noise(self, sigma, size):
        self.ledger.add_gauss(sigma, size)
        return super().gaussian_noise(sigma, size)

    def exponential_mechanism(self, qualities, epsilon, sensitivity=1.0, base_measure=None):
        self.ledger.add_em(epsilon, sensitivity, len(qualities))
        return super().exponential_mechanism(qualities, epsilon, sensitivity, base_measure)

    def worst_approximated(self, candidates, answers, model, eps, sigma):
        self.round += 1
        jax.clear_caches()
        fm = FastMarginals(model)
        errors = {}
        sensitivity = {}
        for cl in candidates:
            wgt = candidates[cl]
            x = answers[cl]
            bias = np.sqrt(2 / np.pi) * sigma * model.domain.size(cl)
            xest = fm.marginal(cl)
            if xest is None:
                xest = np.asarray(model.project(cl).datavector())
            errors[cl] = wgt * (np.linalg.norm(x - xest, 1) - bias)
            sensitivity[cl] = abs(wgt)
        if self.round <= self.check_rounds:
            keys = list(candidates)
            for j in self._check_rng.choice(len(keys), min(self.check_per_round, len(keys)), replace=False):
                cl = keys[j]
                ref = np.asarray(model.project(cl).datavector())
                self.check_maxdiff = max(self.check_maxdiff, float(np.abs(ref - fm.marginal(cl)).max()))
                self.check_n += 1
        max_sensitivity = max(sensitivity.values())
        return self.exponential_mechanism(errors, eps, max_sensitivity)


def run_aim(data: Dataset, workload: list, rho: float, seed_mech: int, rounds=None, max_model_size=80.0,
            max_iters=1000, initial_cliques=None):
    n_oneway = len(data.domain)
    T = rounds or 16 * n_oneway
    if 0.9 * n_oneway / T + 2.0 / T > 1.0:
        raise ValueError(f"AIM rounds T={T} too small for {n_oneway} one-way marginals (need T >= "
                         f"{int(np.ceil(0.9 * n_oneway + 2))})")
    if initial_cliques is not None and len(initial_cliques) != n_oneway:
        raise ValueError("initial_cliques must hold exactly one clique per attribute (official accounting)")
    mech = LedgerAIM(rho, np.random.RandomState(seed_mech % 2 ** 32), rounds=rounds,
                     max_model_size=max_model_size, max_iters=max_iters)
    np.random.seed(seed_mech % 2 ** 32)
    orig_filter = _aim.filter_candidates
    _aim.filter_candidates = filter_candidates_fast
    try:
        model, _ = mech.run(data, workload, num_synth_rows=1,
                            initial_cliques=initial_cliques)
    finally:
        _aim.filter_candidates = orig_filter
    if mech.check_maxdiff > 1e-6 * max(float(model.total), 1.0):
        raise RuntimeError(f"FastMarginals disagrees with model.project: {mech.check_maxdiff}")
    info = {"rounds_param": T, "rounds_run": mech.round, "workload_size": len(workload),
            "initial_cliques": [list(c) for c in initial_cliques] if initial_cliques else "one-way (official)",
            "model_cliques": [list(cl) for cl in model.cliques], "ledger": mech.ledger.summary(),
            "fastmarginals_check": {"n": mech.check_n, "max_abs_diff": mech.check_maxdiff}}

    def sample(rows: int, seed: int):
        np.random.seed(seed % 2 ** 32)
        return model.synthetic_data(rows=rows)

    return sample, float(model.total), mech.ledger, info


def select_fast(data, rho, measurement_log, cliques=()):
    import itertools
    import networkx as nx
    from scipy.cluster.hierarchy import DisjointSet
    est = estimation.MirrorDescent().estimate(data.domain, measurement_log, iters=2500)
    fm = FastMarginals(est)
    weights = {}
    candidates = list(itertools.combinations(data.domain.attrs, 2))
    for a, b in candidates:
        xhat = fm.marginal((a, b))
        x = data.project([a, b]).datavector()
        weights[a, b] = np.linalg.norm(x - xhat, 1)
    T = nx.Graph()
    T.add_nodes_from(data.domain.attrs)
    ds = DisjointSet(data.domain.attrs)
    for e in cliques:
        T.add_edge(*e)
        ds.merge(*e)
    r = len(list(nx.connected_components(T)))
    epsilon = np.sqrt(8 * rho / (r - 1))
    for _ in range(r - 1):
        candidates = [e for e in candidates if not ds.connected(*e)]
        wgts = np.array([weights[e] for e in candidates])
        idx = _mst.exponential_mechanism(wgts, epsilon, sensitivity=1.0)
        e = candidates[idx]
        T.add_edge(*e)
        ds.merge(*e)
    return list(T.edges)


def run_mst(data: Dataset, rho: float, seed: int):
    if not hasattr(Domain, "attrs"):
        Domain.attrs = property(lambda self: self.attributes)
    ledger = Ledger()
    np.random.seed(seed % 2 ** 32)
    sigma = np.sqrt(3 / (2 * rho))

    def measure_logged(d, cliques):
        ms = _mst.measure(d, cliques, sigma)
        for m in ms:
            ledger.add_gauss(m.stddev, d.domain.size(m.clique))
        return ms

    cliques = [(col,) for col in data.domain]
    log1 = measure_logged(data, cliques)
    cdata, log1, undo = _mst.compress_domain(data, log1)
    em_orig = _mst.exponential_mechanism

    def em_logged(q, eps, sensitivity, prng=np.random, monotonic=False):
        ledger.add_em(eps, sensitivity, len(q))
        return em_orig(q, eps, sensitivity, prng, monotonic)

    _mst.exponential_mechanism = em_logged
    try:
        tree = select_fast(cdata, rho / 3.0, log1)
    finally:
        _mst.exponential_mechanism = em_orig
    log2 = measure_logged(cdata, tree)
    est = estimation.MirrorDescent().estimate(cdata.domain, log1 + log2, iters=10000)
    info = {"tree": [list(e) for e in tree], "compressed_domain": dict(zip(cdata.domain.attrs, cdata.domain.shape)),
            "ledger": ledger.summary()}

    def sample(rows: int, seed: int):
        np.random.seed(seed % 2 ** 32)
        return undo(est.synthetic_data(rows=rows))

    return sample, float(est.total), ledger, info
