from __future__ import annotations

import hashlib
import json

import numpy as np

from .api_sim import PopularIndex, RuleVariation, SimulatorRandom, VariationConfig, normalize_times
from .trails import TrailSet

PROMPT_VERSION = "v5"
CITY_NAMES = {"nyc": "New York City, USA", "istanbul": "Istanbul, Turkey", "tokyo": "Tokyo, Japan",
              "jakarta": "Jakarta, Indonesia", "sao_paulo": "São Paulo, Brazil", "kuwait_city": "Kuwait City, Kuwait"}
CITY_NAMES.update({"tsmc_nyc": "New York City, USA", "petaling_jaya": "Petaling Jaya, Malaysia", "moscow": "Moscow, Russia", "bandung": "Bandung, Indonesia", "sydney": "Sydney, Australia", "melbourne": "Melbourne, Australia", "tangerang": "Tangerang, Indonesia", "palembang": "Palembang, Indonesia", "shanghai": "Shanghai, China", "beijing": "Beijing, China", "ms_tokyo": "Tokyo, Japan", "gw_stockholm": "Stockholm, Sweden", "gw_san_francisco": "San Francisco, USA"})
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
AGES = ["18-24", "25-34", "35-44", "45-64", "65+"]
ROLES = ["university student", "office worker", "software engineer", "nurse", "teacher", "retail worker",
         "restaurant worker", "freelancer working from cafes", "construction worker", "retiree",
         "stay-at-home parent", "tourist visiting for a few days", "business traveler", "artist",
         "delivery courier", "government employee", "doctor", "sales representative", "high school student",
         "musician", "fitness trainer", "banker", "shop owner", "graduate student", "taxi driver"]
TRAITS = ["foodie", "night owl", "early riser", "fitness enthusiast", "long commute", "loves shopping",
          "coffee lover", "outdoorsy", "social butterfly", "homebody", "art and culture fan",
          "sports fan", "budget-conscious", "frequent traveler", "religious", "parent of young kids"]
SYSTEM = ("You simulate realistic human mobility in {city}. You produce Foursquare check-in trails: short "
          "sequences of venue visits by one person, where consecutive check-ins are at most 8 hours apart.")
PROFILE = ("\nTypical check-in mix in this city (share of all check-ins): {mix}.\n"
           "Consecutive check-ins are sometimes only a few minutes apart (for example a station and a place "
           "next to it), and check-in times rarely fall on round minutes. People often check in at several "
           "venues of the same kind in a row (changing trains, bar hopping, a shopping street), and not every "
           "trail involves food.")


def checkin_schema(categories, max_len):
    item = {"type": "object",
            "properties": {"time": {"type": "string", "pattern": "^([01][0-9]|2[0-3]):[0-5][0-9]$"},
                           "category": {"type": "string", "enum": list(categories)},
                           "lat": {"type": "number"}, "lon": {"type": "number"}},
            "required": ["time", "category", "lat", "lon"], "additionalProperties": False}
    return {"type": "object",
            "properties": {"checkins": {"type": "array", "items": item, "minItems": 2, "maxItems": max_len}},
            "required": ["checkins"], "additionalProperties": False}


class Grounder:

    def __init__(self, catalog, k: int = 10, scale_km: float = 0.5, max_gap_min: int = 480, jitter_min: int = 7):
        self.c, self.k, self.scale, self.max_gap, self.jitter = catalog, k, scale_km, max_gap_min, jitter_min
        self.venue_w = None
        self.popular_by_cat = {}
        self.consistent = False

    def set_popular(self, popular, radius_km: float = 1.0):
        self.popular_by_cat = {}
        if popular is None:
            return
        idx, w = np.asarray(popular[0]), np.asarray(popular[1])
        for cat in np.unique(self.c.cat[idx]):
            m = self.c.cat[idx] == cat
            self.popular_by_cat[int(cat)] = PopularIndex(self.c, (idx[m], w[m]), radius_km)

    def _choose(self, xy, cat, rng):
        c = self.c
        k = self.k if self.venue_w is not None else 1
        cand = c.nearest(xy, cat=cat, k=k) if cat is not None else None
        if cand is None:
            cand = c.nearest(xy, k=k)
        cand = np.atleast_1d(cand)
        pop = self.popular_by_cat.get(cat) if cat is not None else None
        if pop is not None and pop.tree is not None:
            d, j = pop.tree.query(xy, k=pop.k, distance_upper_bound=pop.r)
            d, j = np.atleast_1d(d), np.atleast_1d(j)
            cand = np.unique(np.concatenate([cand, pop.idx[j[np.isfinite(d)]]]))
        if self.venue_w is None or len(cand) == 1:
            return int(cand[0])
        w = self.venue_w[cand] * np.exp(-np.linalg.norm(c.xy[cand] - xy, axis=1) / self.scale)
        return int(cand[rng.choice(len(cand), p=w / w.sum())])

    def __call__(self, checkins: list[dict], day: int, rng, first_poi: int | None = None):
        c = self.c
        pois, times = [], []
        prev = None
        seen = {}
        for i, ck in enumerate(checkins):
            hh, mm = map(int, ck["time"].split(":"))
            m = day * 1440 + hh * 60 + mm
            if prev is not None:
                while m < prev:
                    m += 1440
                m = min(m, prev + self.max_gap)
            prev = m
            cat = c.cat_index.get(ck["category"])
            place = (cat, round(float(ck["lat"]), 3), round(float(ck["lon"]), 3))
            if i == 0 and first_poi is not None and cat == c.cat[first_poi]:
                v = int(first_poi)
            elif self.consistent and place in seen:
                v = seen[place]
            else:
                v = self._choose(c.clip_xy(c.to_xy(float(ck["lat"]), float(ck["lon"]))), cat, rng)
            seen.setdefault(place, v)
            pois.append(v)
            times.append(m)
        t = np.array(times, np.float64) + rng.integers(-self.jitter, self.jitter + 1, len(times))
        return np.array(pois, np.int32), normalize_times(t)

    def render(self, poi, t) -> list[dict]:
        c = self.c
        return [{"time": f"{(m % 1440) // 60:02d}:{m % 60:02d}", "category": c.categories[c.cat[p]],
                 "lat": round(float(c.lat[p]), 4), "lon": round(float(c.lon[p]), 4)} for p, m in zip(poi, t)]


class LLMAPI:
    def __init__(self, catalog, variation: str = "rule", llm_frac: float = 0.2, max_len: int = 10,
                 engine=None, var_cfg: VariationConfig = VariationConfig(), sim_frac: float = 0.0):
        self.c = catalog
        self.sim_frac = sim_frac
        self._engine = engine
        self.city = CITY_NAMES[catalog.name]
        self.schema = checkin_schema(catalog.categories, max_len)
        self.max_len = max_len
        self.ground = Grounder(catalog)
        self.rule = RuleVariation(catalog, var_cfg)
        self.fallback = SimulatorRandom(catalog)
        self.var_mode, self.llm_frac = variation, llm_frac
        self.venue_w = None
        self.start_p = None
        self.profile = ""
        self.n_fail = 0
        self.n_requests = 0
        self._sem = None
        self._hop_law = None

    def set_semantic(self, sem, frac: float, var: float = 0.0):
        self._sem = (sem, float(frac)) if frac > 0 else None
        self.rule.sem, self.rule.sem_var = (sem, float(var)) if var > 0 else (None, 0.0)
        self.sem = sem

    def set_hop_law(self, edges, probs, cap_km: float):
        self._hop_law = (edges, probs, float(cap_km))

    def semantic_refine(self, ts, p, radius_km, rng, mode="snap", rho=None):
        return self.rule.semantic_refine(ts, self.sem, p, radius_km, rng, mode=mode, rho=rho)

    @property
    def engine(self):
        if self._engine is None:
            from .llm import LLMEngine
            self._engine = LLMEngine()
        return self._engine

    def set_marginals(self, venue_w, start_p, cat_share=None, use=("prompt", "ground", "var"), popular=None,
                      transitions=None):
        prompt = "prompt" in use
        self.venue_w, self.start_p = (venue_w, start_p) if prompt else (None, None)
        self.ground.venue_w = venue_w if "ground" in use else None
        self.rule.set_venue_weights(venue_w if "var" in use else None)
        self.rule.set_popular(popular if "var" in use else None)
        self.ground.set_popular(popular if "ground" in use else None)
        self.fallback.set_marginals(self.venue_w, self.start_p, venue_w if "ground" in use else None)
        self.fallback.set_popular(popular if "ground" in use else None)
        self.rule.set_transitions(transitions if "var" in use else None)
        self.fallback.set_transitions(transitions if "ground" in use else None)
        if cat_share is not None and prompt:
            top = np.argsort(-cat_share)[:25]
            mix = ", ".join(f"{self.c.categories[i]} {100 * cat_share[i]:.1f}%" for i in top if cat_share[i] > 0)
            self.profile = PROFILE.format(mix=mix)

    def usage(self):
        e = self._engine
        return {"model": e.model if e else None, "prompt_tokens": e.n_prompt_tokens if e else 0,
                "output_tokens": e.n_output_tokens if e else 0, "requests": self.n_requests,
                "parse_failures": self.n_fail}

    def _generate(self, convs, rng, days, first_pois=None):
        self.n_requests += len(convs)
        outs = self.engine.chat(convs, temperature=1.0, top_p=0.95, max_tokens=60 * self.max_len + 40,
                                json_schema=self.schema, seeds=rng.integers(0, 2**31 - 1, len(convs)).tolist())
        first_pois = first_pois if first_pois is not None else [None] * len(convs)
        res = []
        for text, day, fp in zip(outs, days, first_pois):
            try:
                cks = json.loads(text)["checkins"]
                res.append(self.ground(cks, int(day), rng, fp) if len(cks) >= 2 else None)
            except (json.JSONDecodeError, KeyError, ValueError, TypeError):
                res.append(None)
        return res

    def _cache_key(self, n, sub_seed):
        h = hashlib.md5((PROMPT_VERSION + self.profile + ("|consistent" if self.ground.consistent else "")).encode())
        for a in (self.venue_w, self.start_p, self.ground.venue_w,
                  None if not self.ground.popular_by_cat else np.array(sorted(self.ground.popular_by_cat))):
            h.update(b"none" if a is None else np.ascontiguousarray(a, np.float64).tobytes())
        return f"llm_random_n{n}_s{sub_seed}_{h.hexdigest()[:10]}.npz"

    def random(self, n: int, rng) -> TrailSet:
        from .paths import RUNS
        sub_seed = int(rng.integers(0, 2**31 - 1))
        n_sim = int(round(n * self.sim_frac))
        n_llm = n - n_sim
        cache = RUNS / self.c.name / "_cache" / self._cache_key(n_llm, sub_seed)
        if cache.exists():
            ts = TrailSet.load(cache)
        else:
            ts = self._random(n_llm, np.random.default_rng(sub_seed))
            cache.parent.mkdir(parents=True, exist_ok=True)
            ts.save(cache)
        if n_sim == 0:
            return ts
        if self._sem is not None:
            self.fallback.set_semantic(*self._sem)
        if self._hop_law is not None:
            self.fallback.set_hop_law(*self._hop_law)
        try:
            sim = self.fallback(n_sim, np.random.default_rng(sub_seed + 1))
        finally:
            self.fallback.set_semantic(None, 0.0)
            self.fallback.hop_law = None
        return TrailSet.from_lists(ts.to_lists() + sim.to_lists())

    def _random(self, n: int, rng) -> TrailSet:
        c = self.c
        start, t0 = self.fallback.sample_starts(n, rng)
        days, hours = t0 // 1440, (t0 % 1440) // 60
        lens = np.minimum(2 + rng.geometric(0.4, n) - 1, self.max_len)
        system = SYSTEM.format(city=self.city) + self.profile
        convs = []
        for i in range(n):
            persona = (f"{AGES[rng.integers(len(AGES))]} year old {ROLES[rng.integers(len(ROLES))]}, "
                       f"{TRAITS[rng.integers(len(TRAITS))]}")
            s = start[i]
            user = (f"Generate one realistic check-in trail for this person in {self.city}.\n"
                    f"Persona: {persona}\nDay: {WEEKDAYS[days[i]]}\n"
                    f"The first check-in is at a {c.categories[c.cat[s]]} around {hours[i]:02d}:00, located at "
                    f"latitude {c.lat[s]:.4f}, longitude {c.lon[s]:.4f}.\nNumber of check-ins: {lens[i]}.\n"
                    "For each check-in give the local time (HH:MM), the Foursquare venue category, and the "
                    "venue's approximate latitude and longitude. Use realistic travel times between places.")
            convs.append([{"role": "system", "content": system}, {"role": "user", "content": user}])
        res = self._generate(convs, rng, days, first_pois=start)
        self.n_fail += sum(r is None for r in res)
        fb = self.fallback(sum(r is None for r in res), rng).to_lists()
        return TrailSet.from_lists([r if r is not None else fb.pop() for r in res])

    def _llm_vary(self, trails, degree, rng):
        system = SYSTEM.format(city=self.city) + self.profile
        convs, days = [], []
        for poi, t in trails:
            k = max(1, int(round(degree * len(poi))))
            minutes = max(10, int(120 * degree))
            user = (f"Here is a check-in trail on a {WEEKDAYS[int(t[0] // 1440) % 7]}:\n"
                    + json.dumps({"checkins": self.ground.render(poi, t)})
                    + f"\nWrite a realistic variation of this trail: change about {k} check-in(s) (a related "
                      "category, a nearby venue, or add/remove one visit) and shift times by up to "
                      f"{minutes} minutes. Keep the person's overall routine plausible.")
            convs.append([{"role": "system", "content": system}, {"role": "user", "content": user}])
            days.append(int(t[0] // 1440) % 7)
        res = self._generate(convs, rng, days)
        self.n_fail += sum(r is None for r in res)
        return [r if r is not None else self.rule.vary_one(p, t, degree, rng) for r, (p, t) in zip(res, trails)]

    def diversify(self, ts, p, radius_km, rng):
        return self.rule.diversify(ts, p, radius_km, rng)

    def variation(self, ts: TrailSet, degree: float, rng) -> TrailSet:
        if self.var_mode == "rule":
            return self.rule(ts, degree, rng)
        trails = ts.to_lists()
        use_llm = np.ones(len(trails), bool) if self.var_mode == "llm" else rng.random(len(trails)) < self.llm_frac
        out = [None] * len(trails)
        idx = np.flatnonzero(use_llm)
        for j, r in zip(idx, self._llm_vary([trails[j] for j in idx], degree, rng)):
            out[j] = r
        for j in np.flatnonzero(~use_llm):
            out[j] = self.rule.vary_one(*trails[j], degree, rng)
        return TrailSet.from_lists(out)
