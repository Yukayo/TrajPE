from __future__ import annotations

import numpy as np

from .api_llm import Grounder
from .trails import TrailSet


class PoolAPI:
    def __init__(self, catalog, records, sim, rule, pool_frac: float = 1.0, refresh: float = 0.0,
                 ground_k: int = 10, ground_scale_km: float = 0.05, pop_radius_km: float = 1.0, max_len: int = 20,
                 refresh_share: float = 0.0):
        self.c = catalog
        self.records = [(int(d), list(cks)[:max_len]) for d, cks in records]
        self.sim, self.rule = sim, rule
        self.pool_frac, self.refresh = float(pool_frac), float(refresh)
        self.refresh_share = float(refresh_share)
        self.ground = Grounder(catalog, k=ground_k, scale_km=ground_scale_km, jitter_min=0)
        self.pop_radius = pop_radius_km
        self.pool: TrailSet | None = None
        self._order, self._next, self._prng = None, 0, None
        self.n_underflow = 0
        self.n_used = 0
        self.n_wraps = 0
        self.sem = None

    def set_marginals(self, venue_w, start_p, cat_share=None, use=("prompt", "ground", "var"), popular=None,
                      transitions=None):
        self.sim.set_marginals(venue_w if "prompt" in use else None, start_p if "prompt" in use else None,
                               venue_w if "ground" in use else None)
        self.sim.set_popular(popular if "ground" in use else None)
        self.sim.set_transitions(transitions if "ground" in use else None)
        self.rule.set_venue_weights(venue_w if "var" in use else None)
        self.rule.set_popular(popular if "var" in use else None)
        self.rule.set_transitions(transitions if "var" in use else None)
        self.ground.venue_w = venue_w if "ground" in use else None
        self.ground.set_popular(popular if "ground" in use else None, radius_km=self.pop_radius)
        if self.pool is not None:
            raise RuntimeError("set_marginals after the pool was grounded")

    def set_semantic(self, sem, frac: float, var: float = 0.0):
        self.sim.set_semantic(sem, frac)
        self.rule.sem, self.rule.sem_var = (sem, float(var)) if var > 0 else (None, 0.0)
        self.sem = sem

    def set_hop_law(self, edges, probs, cap_km: float):
        self.sim.set_hop_law(edges, probs, cap_km)

    def semantic_refine(self, ts, p, radius_km, rng, mode="snap", rho=None):
        return self.rule.semantic_refine(ts, self.sem, p, radius_km, rng, mode=mode, rho=rho)

    def diversify(self, ts, p, radius_km, rng):
        return self.rule.diversify(ts, p, radius_km, rng)

    def random(self, n: int, rng) -> TrailSet:
        sub = int(rng.integers(0, 2**31 - 1))
        if self.pool is None:
            self._ground_all(np.random.default_rng([sub, 1]))
            self._prng = np.random.default_rng([sub, 2])
            self._order = self._prng.permutation(len(self.pool))
        n_pool = int(round(n * self.pool_frac))
        parts = [self._take(n_pool)] if n_pool else []
        if n - n_pool > 0:
            parts.append(self.sim(n - n_pool, np.random.default_rng([sub, 3])))
        return parts[0] if len(parts) == 1 else TrailSet.from_lists(sum((p.to_lists() for p in parts), []))

    def raw_pool(self) -> TrailSet:
        if self.pool is None:
            raise RuntimeError("raw_pool before the first RANDOM_API call")
        return self.pool

    def variation(self, ts: TrailSet, degree: float, rng) -> TrailSet:
        out = self.rule(ts, degree, rng)
        if self.refresh_share > 0 and self.pool is not None and len(self.pool):
            m = rng.random(len(out)) < self.refresh_share
        elif self.refresh <= 0 or degree <= 0 or self.pool is None:
            return out
        else:
            m = rng.random(len(out)) < min(1.0, self.refresh * degree)
        if not m.any():
            return out
        fresh = self._take(int(m.sum())).to_lists()
        lists = out.to_lists()
        for j, tr in zip(np.flatnonzero(m), fresh):
            lists[j] = tr
        return TrailSet.from_lists(lists)

    def usage(self):
        u = {"pool_records": len(self.records), "pool_used": self.n_used, "pool_wraps": self.n_wraps,
             "ground_underflow_fallbacks": self.n_underflow, "pool_frac": self.pool_frac, "refresh": self.refresh}
        if self.refresh_share > 0:
            u["refresh_share"] = self.refresh_share
        return u

    def _take(self, n: int) -> TrailSet:
        idx = []
        while len(idx) < n:
            if self._next >= len(self._order):
                self._order, self._next = self._prng.permutation(len(self.pool)), 0
                self.n_wraps += 1
            k = min(n - len(idx), len(self._order) - self._next)
            idx.extend(self._order[self._next:self._next + k].tolist())
            self._next += k
        self.n_used += n
        return self.pool.subset(np.asarray(idx, dtype=np.int64))

    def _ground_all(self, rng):
        g = self.ground
        base = g._choose
        g._choose = self._choose_with(base)
        try:
            trails = [g(cks, day, rng) for day, cks in self.records]
        finally:
            g._choose = base
        self.pool = TrailSet.from_lists(trails)

    def _choose_with(self, base):
        def f(xy, cat, rng):
            try:
                return base(xy, cat, rng)
            except ValueError:
                self.n_underflow += 1
                v = self.c.nearest(xy, cat=cat, k=1) if cat is not None else None
                if v is None:
                    v = self.c.nearest(xy, k=1)
                return int(np.atleast_1d(v)[0])
        return f
