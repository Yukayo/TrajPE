from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.spatial import cKDTree

ROOTS = ["Arts and Entertainment", "Business and Professional Services", "Community and Government",
         "Dining and Drinking", "Event", "Health and Medicine", "Landmarks and Outdoors", "Retail",
         "Sports and Recreation", "Travel and Transportation"]
N_ROOTS = len(ROOTS) + 1
DIST_EDGES = np.array([0, .25, .5, 1, 2, 3, 5, 7.5, 10, 15, 20, 30, 50, np.inf])
GAP_EDGES = np.array([0, 5, 10, 20, 30, 45, 60, 90, 120, 180, 240, 360, 481])
N_DIST, N_BEAR, N_GAP = len(DIST_EDGES) - 1, 8, len(GAP_EDGES) - 1
N_WH = 7 * 24
LEN_MIN, LEN_MAX = 2, 20
POSITIONS = list(range(2, LEN_MAX + 1))
KM_PER_DEG_LAT = 110.574
FULL = (LEN_MIN, LEN_MAX)


@dataclass(frozen=True)
class Spec:
    grid: int = 16
    od_grid: int = 8
    k_sub: int = 16
    min_gap: int = 1
    max_pos: int = LEN_MAX
    time_split: bool = False
    strata: tuple = (FULL,)

    def __post_init__(self):
        cover = sorted(L for lo, hi in self.strata for L in range(lo, hi + 1))
        if cover != list(range(LEN_MIN, LEN_MAX + 1)):
            raise ValueError(f"strata {self.strata} must partition {LEN_MIN}..{LEN_MAX}")

    def positions(self, stratum=FULL) -> list:
        return list(range(2, min(stratum[1], self.max_pos) + 1))

    def stratum_of(self, lengths: np.ndarray) -> np.ndarray:
        out = np.full(len(lengths), -1, np.int64)
        for j, (lo, hi) in enumerate(self.strata):
            out[(lengths >= lo) & (lengths <= hi)] = j
        return out

    def to_json(self) -> dict:
        d = asdict(self)
        d["strata"] = [list(s) for s in self.strata]
        return d


def bearing_bin(dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    th = np.mod(np.arctan2(dy, dx), 2 * np.pi)
    return np.minimum((th / (2 * np.pi / N_BEAR)).astype(np.int64), N_BEAR - 1)


def dist_bin(d: np.ndarray) -> np.ndarray:
    return np.clip(np.searchsorted(DIST_EDGES, d, side="right") - 1, 0, N_DIST - 1)


def gap_bin(g: np.ndarray) -> np.ndarray:
    return np.clip(np.searchsorted(GAP_EDGES, g, side="right") - 1, 0, N_GAP - 1)


class PublicDomain:

    def __init__(self, pois, bbox: dict, spec: Spec = Spec()):
        self.spec, self.bbox = spec, dict(bbox)
        lon = pois["lon"].to_numpy(np.float64)
        lat = pois["lat"].to_numpy(np.float64)
        lat0 = 0.5 * (bbox["lat_min"] + bbox["lat_max"])
        self.kx, self.ky = 111.320 * np.cos(np.radians(lat0)), KM_PER_DEG_LAT
        self.width = (bbox["lon_max"] - bbox["lon_min"]) * self.kx
        self.height = (bbox["lat_max"] - bbox["lat_min"]) * self.ky
        self.xy = np.c_[(lon - bbox["lon_min"]) * self.kx, (lat - bbox["lat_min"]) * self.ky]
        self.n_venues = len(lon)
        cats = pois["category"].astype(str).to_numpy()
        roots = pois["root"].astype(str).to_numpy()
        root_of = dict(zip(cats, roots))
        self.categories = sorted(set(cats))
        cat_code = {c: i for i, c in enumerate(self.categories)}
        self.venue_cat = np.array([cat_code[c] for c in cats], np.int64)
        self.cat_root = np.array([ROOTS.index(root_of[c]) if root_of[c] in ROOTS else len(ROOTS)
                                  for c in self.categories], np.int64)
        n_per_cat = np.bincount(self.venue_cat, minlength=len(self.categories))
        self.cat_sub = np.full(len(self.categories), spec.k_sub, np.int64)
        for r in range(N_ROOTS):
            members = np.flatnonzero(self.cat_root == r)
            order = sorted(members, key=lambda c: (-n_per_cat[c], self.categories[c]))
            for j, c in enumerate(order[:spec.k_sub]):
                self.cat_sub[c] = j
        self.n_sub = spec.k_sub + 1 if spec.k_sub > 0 else 1
        self.venue_root = self.cat_root[self.venue_cat]
        self.venue_sub = self.cat_sub[self.venue_cat]
        self.venue_cls = self.venue_root * self.n_sub + self.venue_sub
        self.n_cls = N_ROOTS * self.n_sub
        self.venue_cell = self.cell(self.xy, spec.grid)
        self.venue_ocell = self.cell(self.xy, spec.od_grid)

    def cell(self, xy: np.ndarray, g: int) -> np.ndarray:
        xy = np.atleast_2d(xy)
        gx = np.clip((xy[:, 0] / self.width * g).astype(np.int64), 0, g - 1)
        gy = np.clip((xy[:, 1] / self.height * g).astype(np.int64), 0, g - 1)
        return gy * g + gx

    def cell_rect(self, c: int, g: int) -> tuple:
        gy, gx = divmod(int(c), g)
        return (gx * self.width / g, (gx + 1) * self.width / g, gy * self.height / g, (gy + 1) * self.height / g)

    @property
    def use_sub(self) -> bool:
        return self.spec.k_sub > 0

    def attributes(self, stratum=FULL) -> dict:
        lo, hi = stratum
        a = {}
        if hi > lo:
            a["L"] = hi - lo + 1
        if self.spec.time_split:
            a["dow"], a["hod"] = 7, 24
        else:
            a["wh"] = N_WH
        a["sc"], a["ec"], a["r1"] = self.spec.grid ** 2, self.spec.od_grid ** 2, N_ROOTS
        if self.use_sub:
            a["s1"] = self.n_sub
        for k in self.spec.positions(stratum):
            x = int(k > lo)
            a[f"d{k}"], a[f"b{k}"], a[f"g{k}"], a[f"r{k}"] = N_DIST + x, N_BEAR + x, N_GAP + x, N_ROOTS + x
            if self.use_sub:
                a[f"s{k}"] = self.n_sub + x
        return a

    def summary(self) -> dict:
        return {"spec": self.spec.to_json(), "n_venues": int(self.n_venues), "n_categories": len(self.categories),
                "n_classes": int(self.n_cls), "width_km": float(self.width), "height_km": float(self.height),
                "strata": {f"{lo}-{hi}": self.attributes((lo, hi)) for lo, hi in self.spec.strata}}

    def encode(self, venues: np.ndarray, lengths: np.ndarray, start_wh: np.ndarray, gaps_min: np.ndarray,
               stratum=FULL) -> dict:
        lo, hi = stratum
        lengths = np.asarray(lengths, np.int64)
        if len(lengths) and (lengths.min() < lo or lengths.max() > hi):
            raise ValueError(f"trail lengths must lie in the stratum [{lo}, {hi}]")
        n = len(lengths)
        attrs = self.attributes(stratum)
        off = np.concatenate([[0], np.cumsum(lengths)])
        first, last = off[:-1], off[1:] - 1
        v = np.asarray(venues, np.int64)
        wh = np.asarray(start_wh, np.int64)
        rec = {"sc": self.venue_cell[v[first]], "ec": self.venue_ocell[v[last]], "r1": self.venue_root[v[first]]}
        if "L" in attrs:
            rec["L"] = lengths - lo
        if self.spec.time_split:
            rec["dow"], rec["hod"] = wh // 24, wh % 24
        else:
            rec["wh"] = wh
        if self.use_sub:
            rec["s1"] = self.venue_sub[v[first]]
        goff = np.concatenate([[0], np.cumsum(lengths - 1)])
        gaps_min = np.asarray(gaps_min, np.int64)
        for k in self.spec.positions(stratum):
            i = np.flatnonzero(lengths >= k)
            cur, prev = v[first[i] + k - 1], v[first[i] + k - 2]
            dxy = self.xy[cur] - self.xy[prev]
            cols = {"d": dist_bin(np.hypot(dxy[:, 0], dxy[:, 1])), "b": bearing_bin(dxy[:, 0], dxy[:, 1]),
                    "g": gap_bin(gaps_min[goff[i] + k - 2]), "r": self.venue_root[cur]}
            if self.use_sub:
                cols["s"] = self.venue_sub[cur]
            for p, val in cols.items():
                name = f"{p}{k}"
                col = np.full(n, attrs[name] - 1, np.int64)
                col[i] = val
                rec[name] = col
        for name, size in attrs.items():
            assert n == 0 or (rec[name].min() >= 0 and rec[name].max() < size), name
        return {name: rec[name] for name in attrs}


def spatial_tilt(dom: PublicDomain, weights: np.ndarray | None, sc: np.ndarray) -> np.ndarray:
    g2 = dom.spec.grid ** 2
    w = np.ones(dom.n_venues) if weights is None else np.asarray(weights, np.float64)
    P = np.bincount(np.asarray(sc, np.int64), minlength=g2) + 0.5
    Q = np.bincount(dom.venue_cell, weights=w, minlength=g2)
    P, Q = P / P.sum(), Q / max(Q.sum(), 1e-300)
    return np.divide(P, Q, out=np.ones(g2), where=Q > 0)


class Decoder:

    LEVELS = ["class+sector+end", "class+annulus+end", "class+sector", "class+annulus", "root+sector",
              "root+annulus", "any+sector", "any+annulus", "nearest_class", "nearest_any"]

    def __init__(self, dom: PublicDomain, weights: np.ndarray | None, tilt: np.ndarray | None = None,
                 end_any: bool = False, use_end: bool = True):
        self.d = dom
        self.end_any, self.use_end = end_any, use_end
        self.w = np.ones(dom.n_venues) if weights is None else np.asarray(weights, np.float64)
        self.w_hop = self.w if tilt is None else self.w * np.asarray(tilt, np.float64)[dom.venue_cell]
        self._trees: dict = {}
        self._start = {}
        for name, key, mult in (("cls", dom.venue_cls, dom.spec.grid ** 2), ("root", dom.venue_root, dom.spec.grid ** 2),
                                ("any", np.zeros(dom.n_venues, np.int64), dom.spec.grid ** 2)):
            k = key * mult + dom.venue_cell
            order = np.argsort(k, kind="stable")
            self._start[name] = (k[order], order)

    def _tree(self, kind: str, code: int):
        key = (kind, int(code))
        if key not in self._trees:
            if kind == "cls":
                idx = np.flatnonzero(self.d.venue_cls == code)
            elif kind == "root":
                idx = np.flatnonzero(self.d.venue_root == code)
            else:
                idx = np.arange(self.d.n_venues)
            self._trees[key] = (cKDTree(self.d.xy[idx]) if len(idx) else None, idx)
        return self._trees[key]

    def _pick(self, cand: np.ndarray, rng: np.random.Generator) -> int:
        w = self.w_hop[cand]
        return int(cand[np.searchsorted(np.cumsum(w), rng.random() * w.sum(), side="right").clip(0, len(cand) - 1)])

    def _start_venue(self, cls: np.ndarray, root: np.ndarray, sc: np.ndarray, rng, stats) -> np.ndarray:
        g2 = self.d.spec.grid ** 2
        out = np.empty(len(cls), np.int64)
        keys = {"cls": cls * g2 + sc, "root": root * g2 + sc, "any": sc}
        todo = np.arange(len(cls))
        for level in ("cls", "root", "any"):
            if not len(todo):
                break
            sk, order = self._start[level]
            kk = keys[level][todo]
            uk, inv = np.unique(kk, return_inverse=True)
            lo, hi = np.searchsorted(sk, uk, "left"), np.searchsorted(sk, uk, "right")
            grp = np.argsort(inv.reshape(-1), kind="stable")
            gb = np.concatenate([[0], np.cumsum(np.bincount(inv.reshape(-1), minlength=len(uk)))])
            left = []
            for j in range(len(uk)):
                members = todo[grp[gb[j]:gb[j + 1]]]
                if hi[j] > lo[j]:
                    cand = order[lo[j]:hi[j]]
                    w = self.w[cand]
                    out[members] = cand[rng.choice(len(cand), size=len(members), p=w / w.sum())]
                else:
                    left.append(members)
            stats[f"start_{level}"] = stats.get(f"start_{level}", 0) + int(len(todo) - sum(len(m) for m in left))
            todo = np.concatenate(left) if left else np.zeros(0, np.int64)
        for i in todo:
            x0, x1, y0, y1 = self.d.cell_rect(sc[i], self.d.spec.grid)
            tree, idx = self._tree("cls", cls[i])
            if tree is None:
                tree, idx = self._tree("any", 0)
            out[i] = idx[tree.query([0.5 * (x0 + x1), 0.5 * (y0 + y1)])[1]]
            stats["start_nearest"] = stats.get("start_nearest", 0) + 1
        return out

    def _hop(self, prev: int, cls: int, root: int, db: int, bb: int, ec: int | None, rng, stats) -> int:
        p = self.d.xy[prev]
        a, b = DIST_EDGES[db], DIST_EDGES[db + 1]
        r_out = b if np.isfinite(b) else np.hypot(self.d.width, self.d.height) + 1.0
        for kind, code in (("cls", cls), ("root", root), ("any", 0)):
            tree, idx = self._tree(kind, code)
            if tree is None:
                continue
            cand = idx[np.asarray(tree.query_ball_point(p, r_out), np.int64)]
            if not len(cand):
                continue
            dxy = self.d.xy[cand] - p
            dist = np.hypot(dxy[:, 0], dxy[:, 1])
            ann = (dist >= a) & (dist < b)
            sec = ann & ((bearing_bin(dxy[:, 0], dxy[:, 1]) == bb) | (dist < 1e-9))
            masks = [("sector", sec), ("annulus", ann)]
            if kind == "cls" and ec is not None:
                inend = self.d.venue_ocell[cand] == ec
                masks = [("sector+end", sec & inend), ("annulus+end", ann & inend)] + masks
            for name, m in masks:
                if m.any():
                    lvl = f"{'class' if kind == 'cls' else kind}+{name}"
                    stats[lvl] = stats.get(lvl, 0) + 1
                    return self._pick(cand[m], rng)
                if name == "annulus+end" and self.end_any:
                    allc = idx[self.d.venue_ocell[idx] == ec]
                    if len(allc):
                        stats["class+end"] = stats.get("class+end", 0) + 1
                        return self._pick(allc, rng)
        th = (bb + 0.5) * 2 * np.pi / N_BEAR
        rm = 0.5 * (a + (b if np.isfinite(b) else a + 10.0))
        q = np.clip(p + rm * np.array([np.cos(th), np.sin(th)]), 0, [self.d.width, self.d.height])
        tree, idx = self._tree("cls", cls)
        lvl = "nearest_class"
        if tree is None:
            tree, idx = self._tree("any", 0)
            lvl = "nearest_any"
        stats[lvl] = stats.get(lvl, 0) + 1
        return int(idx[tree.query(q)[1]])

    def decode(self, rec: dict, rng: np.random.Generator, min_gap: int = 1, stratum=FULL, stats=None):
        d = self.d
        lo, hi = stratum
        attrs = d.attributes(stratum)
        n = len(next(iter(rec.values())))
        L = np.asarray(rec["L"], np.int64) + lo if "L" in rec else np.full(n, lo, np.int64)
        stats = {} if stats is None else stats
        stats["records"] = stats.get("records", 0) + int(n)
        rec = {k: np.asarray(v, np.int64).copy() for k, v in rec.items()}
        positions = d.spec.positions(stratum)
        n_fix = 0

        def redraw(name, bad, key):
            absent = attrs[name] - 1
            ok = rec[name] != absent
            out = rec[name].copy()
            for kv in np.unique(key[bad]):
                m = bad & (key == kv)
                pool = rec[name][ok & (key == kv)]
                if not len(pool):
                    pool = rec[name][ok]
                if not len(pool):
                    pool = np.arange(absent)
                out[m] = rng.choice(pool, size=int(m.sum()))
            rec[name] = out

        for k in positions:
            if k <= lo:
                continue
            act = L >= k
            bad = act & (rec[f"r{k}"] == attrs[f"r{k}"] - 1)
            if bad.any():
                redraw(f"r{k}", bad, rec[f"r{k - 1}"])
                n_fix += int(bad.sum())
            for p in ("d", "b", "g", "s"):
                name = f"{p}{k}"
                if name not in rec:
                    continue
                bad = act & (rec[name] == attrs[name] - 1)
                if bad.any():
                    redraw(name, bad, rec[f"r{k}"])
                    n_fix += int(bad.sum())
        stats["absent_redrawn"] = stats.get("absent_redrawn", 0) + n_fix
        Lmax = int(L.max()) if n else lo
        P = positions[-1] if positions else 1
        if Lmax > P:
            src = np.flatnonzero(L >= P)
            names = [p for p in ("d", "b", "g", "r", "s") if f"{p}{P}" in rec]
            pool = {p: rec[f"{p}{P}"][src] for p in names}
            pool_prev = rec[f"r{P - 1}"][src]
            by_prev = {int(r): np.flatnonzero(pool_prev == r) for r in np.unique(pool_prev)}
            n_ext = 0
            for k in range(P + 1, Lmax + 1):
                for p in names:
                    rec[f"{p}{k}"] = np.zeros(n, np.int64)
                act = np.flatnonzero(L >= k)
                prev = rec[f"r{k - 1}"][act]
                pick = np.empty(len(act), np.int64)
                for r in np.unique(prev):
                    m = prev == r
                    cand = by_prev.get(int(r))
                    if cand is None:
                        cand = np.arange(len(src))
                    pick[m] = cand[rng.integers(0, len(cand), int(m.sum()))]
                for p in names:
                    rec[f"{p}{k}"][act] = pool[p][pick]
                n_ext += len(act)
            stats["positions_beyond_max_pos"] = stats.get("positions_beyond_max_pos", 0) + n_ext
        sub = (lambda k: rec[f"s{k}"]) if d.use_sub else (lambda k: np.zeros(n, np.int64))
        venues = np.empty((n, LEN_MAX), np.int64)
        venues[:, 0] = self._start_venue(rec["r1"] * d.n_sub + sub(1), rec["r1"], rec["sc"], rng, stats)
        for k in range(2, Lmax + 1):
            for i in np.flatnonzero(L >= k):
                r = int(rec[f"r{k}"][i])
                venues[i, k - 1] = self._hop(int(venues[i, k - 2]), r * d.n_sub + int(sub(k)[i]), r,
                                             int(rec[f"d{k}"][i]), int(rec[f"b{k}"][i]),
                                             int(rec["ec"][i]) if (k == L[i] and self.use_end) else None,
                                             rng, stats)
        mask = np.arange(LEN_MAX)[None, :] < L[:, None]
        vidx = venues[mask]
        wh = rec["dow"] * 24 + rec["hod"] if "dow" in rec else rec["wh"]
        t0 = wh * 60 + rng.integers(0, 60, size=n)
        gmin = np.zeros((n, LEN_MAX), np.int64)
        for k in range(2, Lmax + 1):
            gb = rec[f"g{k}"].clip(0, N_GAP - 1)
            glo = np.maximum(GAP_EDGES[gb], min_gap)
            ghi = GAP_EDGES[gb + 1]
            gmin[:, k - 1] = glo + np.floor(rng.random(n) * (ghi - glo)).astype(np.int64)
        gmin[:, 0] = t0
        tmin = np.cumsum(gmin, axis=1)[mask]
        stats["end_cell_kept"] = stats.get("class+sector+end", 0) + stats.get("class+annulus+end", 0)
        return vidx, L, tmin, stats
