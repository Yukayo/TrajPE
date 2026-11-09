import argparse
import csv
import hashlib
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from trajpe.data import DATASETS, GOWALLA_SPOTS, V2_DATASETS, _load_massive_steps, _load_tsmc
from trajpe.paths import PROCESSED, RAW, ROOT
from trajpe.public_catalog import CATEGORY_MAP_PATH, CATEGORY_MAP_V3_PATH, MULTI_TARGET

V2_OVERRIDES = {"Subway": "Metro Station", "Pizza Place": "Pizzeria", "Light Rail": "Light Rail Station",
                "Ice Cream Shop": "Ice Cream Parlor", "Dentist's Office": "Dentist",
                "Massage Studio": "Massage Clinic", "Ferry": "Marine Terminal", "Gun Shop": "Hunting Supply Store",
                "Building": "Office Building", "Fabric Shop": "Fabric Store", "Salad Place": "Salad Restaurant",
                "Soup Place": "Soup Spot"}
_R = {"KIND": "LLM leaf is a different kind of place", "EXACT": "a leaf with the same meaning exists",
      "V2": "same rename as the v2 overrides", "PARENT": "public 2014 Foursquare parent",
      "CHAIN": "brand mapped to the leaf of its plain kind, as its siblings"}
_O = [
    ("gowalla:454", "Sandwich Spot", "KIND", "'Subway' outside the tree is the sandwich chain; the metro is gowalla:164"),
    ("Ash and Haleem Place", "Persian Restaurant", "EXACT", "Iranian dishes"),
    ("Botanero", "Bar", "KIND", "Mexican snack bar, not a garden center"),
    ("Catalan Restaurant", "Spanish Restaurant", "KIND", "not a cafe"),
    ("Dizi Place", "Persian Restaurant", "KIND", "dizi is a Persian stew"),
    ("Frame Store", "Framing Store", "KIND", "picture framing, not a boutique"),
    ("German Pop-Up Restaurant", "German Restaurant", "KIND", "a restaurant, not a pop-up store"),
    ("Magirio", "Greek Restaurant", "PARENT", "Greek cookhouse; LLM chose a pickle shop"),
    ("Malga", "Italian Restaurant", "PARENT", ""),
    ("Marche Restaurant", "Italian Restaurant", "PARENT", "Italian region"),
    ("Norman Restaurant", "French Restaurant", "KIND", "Normandy; LLM chose Chinese"),
    ("Northeast Indian Restaurant", "Indian Restaurant", "PARENT", "LLM chose South Indian"),
    ("Palatine Restaurant", "German Restaurant", "KIND", "Palatinate; LLM chose Chinese"),
    ("Pelmeni House", "Russian Restaurant", "PARENT", ""),
    ("Shaanxi Restaurant", "Chinese Restaurant", "PARENT", "Shaanxi is not Shanxi"),
    ("Tabbakhi", "Middle Eastern Restaurant", "KIND", "Arabic home cooking; LLM chose a pickle shop"),
    ("gowalla:89", "Residential Building", "KIND", "Community > Home > Craftsman is a house style; LLM chose Boutique"),
    ("gowalla:29", "Apartment or Condo", "KIND", "Condo (MULTI_TARGET applies anyway)"),
    ("gowalla:121", "Office", "EXACT", "Corporate Office under Community > Office"),
    ("gowalla:23", "Office Building", "V2", "Skyscraper under Community > Office; LLM chose a surveyor"),
    ("gowalla:139", "College Residence Hall", "KIND", "Dormitory"),
    ("gowalla:135", "College Administrative Building", "KIND", "Campus Spot > Administration"),
    ("gowalla:142", "College Lab", "KIND", "Campus Spot > Lab"),
    ("gowalla:208", "Spiritual Center", "KIND", "Other - Place of Worship; LLM chose Religious School"),
    ("gowalla:60", "Landmarks and Outdoors", "KIND", "Outdoors > Architecture; LLM chose Architecture Firm"),
    ("gowalla:13", "Landmarks and Outdoors", "KIND", "Outdoors > Architecture; LLM chose Architecture Firm"),
    ("gowalla:164", "Metro Station", "V2", "Travel > Subway"),
    ("gowalla:395", "Metro Station", "EXACT", "London Underground"),
    ("gowalla:94", "Light Rail Station", "V2", "Travel > Light Rail"),
    ("gowalla:95", "Rail Station", "EXACT", "Travel > Train Station"),
    ("gowalla:172", "Marine Terminal", "V2", "Travel > Ferry; LLM chose Boat Rental"),
    ("gowalla:226", "Airport Tram Station", "EXACT", "Travel > Airport > Tram"),
    ("gowalla:914", "Bus Station", "KIND", "Capital Metro (Austin transit); LLM chose Government Building"),
    ("gowalla:375", "Rental Car Location", "EXACT", "Car2Go car sharing"),
    ("gowalla:915", "Airport", "KIND", "bmibaby is an airline; LLM chose Boutique"),
    ("gowalla:155", "Pizzeria", "V2", "Food > Pizza"),
    ("gowalla:909", "Ice Cream Parlor", "V2", "Food > Dessert > Ice Cream"),
    ("gowalla:913", "Frozen Yogurt Shop", "EXACT", ""),
    ("gowalla:361", "Donut Shop", "EXACT", "Doughnuts"),
    ("gowalla:187", "Hot Dog Joint", "KIND", "LLM chose Diner"),
    ("gowalla:237", "Southern Food Restaurant", "EXACT", "Soul Food"),
    ("gowalla:123", "Breakfast Spot", "EXACT", ""),
    ("gowalla:64", "Dining and Drinking", "KIND", "Other - Food; LLM chose Food Court"),
    ("gowalla:178", "Street Food Gathering", "KIND", "Street Fare; LLM chose Food Court (as gowalla:204)"),
    ("gowalla:97", "Bar", "KIND", "Saloon; LLM chose Night Club"),
    ("gowalla:455", "Donut Shop", "CHAIN", "Krispy Kreme; LLM chose Diner"),
    ("gowalla:449", "Donut Shop", "CHAIN", "Dunkin' Donuts"),
    ("gowalla:902", "Ice Cream Parlor", "CHAIN", "Ben & Jerry's; LLM chose Boutique"),
    ("gowalla:905", "Fast Food Restaurant", "CHAIN", "Dairy Queen; LLM chose Dairy Store"),
    ("gowalla:461", "Sandwich Spot", "CHAIN", "Potbelly; LLM chose Bakery"),
    ("gowalla:363", "Mexican Restaurant", "CHAIN", "Chipotle"),
    ("gowalla:906", "Burger Joint", "CHAIN", "Whataburger"),
    ("gowalla:441", "Coffee Shop", "CHAIN", "Seattle's Best"),
    ("gowalla:437", "Grocery Store", "CHAIN", "Trader Joe's"),
    ("gowalla:434", "Furniture and Home Store", "CHAIN", "IKEA"),
    ("gowalla:920", "Furniture and Home Store", "CHAIN", "Design Within Reach"),
    ("gowalla:443", "Hardware Store", "KIND", "The Home Depot; LLM chose a service"),
    ("gowalla:442", "Hardware Store", "KIND", "Lowe's; LLM chose a service"),
    ("gowalla:458", "Electronics Store", "CHAIN", "Best Buy"),
    ("gowalla:230", "Electronics Store", "CHAIN", "Apple Store"),
    ("gowalla:406", "Mobile Phone Store", "CHAIN", "AT&T"),
    ("gowalla:468", "Mobile Phone Store", "CHAIN", "Verizon"),
    ("gowalla:426", "Mobile Phone Store", "CHAIN", "Sprint"),
    ("gowalla:469", "Mobile Phone Store", "CHAIN", "T-Mobile"),
    ("gowalla:438", "Big Box Store", "CHAIN", "Target (as Walmart)"),
    ("gowalla:904", "Big Box Store", "CHAIN", "Costco"),
    ("gowalla:459", "Department Store", "CHAIN", "Nordstrom"),
    ("gowalla:473", "Clothing Store", "CHAIN", "Gap (as Old Navy, H&M)"),
    ("gowalla:453", "Clothing Store", "CHAIN", "Banana Republic"),
    ("gowalla:474", "Clothing Store", "CHAIN", "Urban Outfitters"),
    ("gowalla:466", "Clothing Store", "CHAIN", "lululemon"),
    ("gowalla:451", "Lingerie Store", "CHAIN", "Victoria's Secret"),
    ("gowalla:394", "Sporting Goods Retail", "CHAIN", "Nike"),
    ("gowalla:435", "Outdoor Supply Store", "CHAIN", "REI"),
    ("gowalla:445", "Toy Store", "CHAIN", "LEGO Store"),
    ("gowalla:448", "Gym", "CHAIN", "24 Hour Fitness"),
    ("gowalla:447", "Gym", "CHAIN", "Life Time Fitness"),
    ("gowalla:383", "Hotel", "CHAIN", "Staybridge Suites"),
    ("gowalla:424", "Amusement Park", "CHAIN", "Walt Disney World Resort"),
    ("gowalla:425", "Amusement Park", "CHAIN", "Disneyland Resort; LLM chose Theme Restaurant"),
    ("gowalla:36", "Electronics Store", "KIND", "Shopping > Technology; LLM chose Tech Startup"),
    ("gowalla:911", "Music Store", "KIND", "Shopping > Music; LLM chose Music Venue"),
    ("gowalla:61", "Sporting Goods Retail", "KIND", "Shopping > Sports & Outdoors; LLM chose a venue root"),
    ("gowalla:413", "Sporting Goods Retail", "KIND", "Shopping > Sports & Outdoors > Other Sports"),
    ("gowalla:415", "Sporting Goods Retail", "KIND", "Shopping > Sports & Outdoors > Ski Shop"),
    ("gowalla:416", "Outdoor Supply Store", "KIND", "Shopping > Sports & Outdoors > Camping & Outdoors"),
    ("gowalla:137", "Swimming Pool", "KIND", "Aquatics; LLM chose a pool supply store"),
    ("gowalla:163", "Stationery Store", "EXACT", "Paper Goods"),
    ("gowalla:242", "Video Store", "EXACT", "Movie Rental"),
    ("gowalla:217", "Business and Professional Services", "KIND", "Shopping > Other - Services; LLM chose a store"),
    ("gowalla:386", "Swimming Pool", "EXACT", "Outdoors > Pool"),
    ("gowalla:39", "State or Provincial Park", "EXACT", "Regional/State Park"),
    ("gowalla:55", "Park", "EXACT", "Park - Other"),
    ("gowalla:165", "Plaza", "EXACT", "Plaza/Square"),
    ("gowalla:114", "Lake", "EXACT", "Lake & Pond"),
    ("gowalla:166", "Historic and Protected Site", "EXACT", "Historic Landmark"),
    ("gowalla:131", "Ski Resort and Area", "EXACT", "Ski & Snowboard Area"),
    ("gowalla:211", "Skate Park", "EXACT", "Skatepark"),
    ("gowalla:663", "Skating Rink", "EXACT", "Ice Skating"),
    ("gowalla:78", "Race Track", "EXACT", "Racetrack"),
    ("gowalla:662", "Stadium", "KIND", "Arena or Stadium > Arena; LLM chose Concert Hall"),
    ("gowalla:58", "Arts and Entertainment", "KIND", "Other - Entertainment; LLM chose an event"),
    ("gowalla:404", "Campaign Office", "KIND", "election campaign; LLM chose Electronics Store (as gowalla:403)"),
    ("gowalla:405", "Campaign Office", "KIND", "election campaign; LLM chose Electronics Store (as gowalla:403)"),
    ("gowalla:400", "Other Event", "KIND", "Political Rally (as gowalla:399)"),
    ("gowalla:397", "Other Event", "KIND", "Democratic Event; LLM chose a service"),
    ("gowalla:398", "Other Event", "KIND", "Republican Event; LLM chose a service"),
    ("gowalla:430", "Movie Theater", "KIND", "Sundance Screening; LLM chose Film Studio"),
]
V3_OVERRIDES = {k: (t, f"{c}: {_R[c]}" + (f" ({note})" if note else "")) for k, t, c, note in _O}
assert len(V3_OVERRIDES) == len(_O)
SXSW_RULE = [("Film > Screening", "Movie Theater"), ("Music > Showcase", "Music Venue"),
             ("Film > ", "Event Space"), ("Interactive > ", "Event Space"), ("Music > ", "Event Space")]
LEAVES_FILE = "frozen/fsq_leaves_v2.json"
STD_CATEGORIES = "semantic_trails/categories.csv"
TSMC_FILES = ["tsmc2014/extracted/dataset_tsmc2014/dataset_TSMC2014_NYC.txt",
              "tsmc2014/extracted/dataset_tsmc2014/dataset_TSMC2014_TKY.txt"]
GOWALLA_TREE = "gowalla_liu/gowalla_category_structure.json"
QUESTION = ("Which category in the list above describes the same kind of venue most closely? Prefer a specific "
            "category over a top-level one. Answer with the exact category name only.")


def norm(s: str) -> str:
    s = s.lower().replace("&", " and ").replace("/", " ")
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def system_prompt(pub: list, roots: dict) -> str:
    grouped = {}
    for p in pub:
        grouped.setdefault(roots.get(p, "Other"), []).append(p)
    listing = "\n".join(f"{r}: " + "; ".join(sorted(v)) for r, v in sorted(grouped.items()))
    return ("You are an expert on the Foursquare venue category taxonomy. The current Foursquare Places "
            "categories, grouped by top-level category, are:\n" + listing)


def fsq_user_prompt(n: str) -> str:
    return f'A venue category from the 2014 Foursquare taxonomy: "{n}".\n' + QUESTION


def gowalla_user_prompt(name: str, path) -> str:
    where = f" (Gowalla category path: {path})" if path else ""
    return f'A venue category from the Gowalla location-based social network (2009-2011): "{name}"{where}.\n' + QUESTION


def v2_leaves() -> tuple[list, dict]:
    cities = list(V2_DATASETS)
    pub = sorted(set().union(*[set(pd.read_parquet(PROCESSED / c / "pois.parquet", columns=["category"]).category)
                               for c in cities]))
    roots = {}
    for c in cities:
        roots.update(dict(pd.read_parquet(PROCESSED / c / "pois.parquet", columns=["category", "root"]).drop_duplicates().values))
    return pub, roots


def run_v2(check: bool) -> None:
    pub, roots = v2_leaves()
    names = set()
    for c in V2_DATASETS:
        loader, rel, tz = DATASETS[c]
        raw = _load_massive_steps(rel, tz) if loader == "massive_steps" else _load_tsmc(rel)
        names |= set(raw["category"].astype(str))
    names = sorted(names)
    cache = json.loads(CATEGORY_MAP_PATH.read_text()) if CATEGORY_MAP_PATH.exists() else {}
    by_norm = {norm(p): p for p in pub}
    mapping = {n: by_norm[norm(n)] for n in names if norm(n) in by_norm}
    mapping.update({n: cache[n] for n in names if n not in mapping and n in cache and cache[n] in pub})
    rest = [n for n in names if n not in mapping]
    print(f"public leaves {len(pub)}, source names {len(names)}, matched by name {len(mapping)}, LLM {len(rest)}")
    if rest:
        from trajpe.llm import LLMEngine
        engine = LLMEngine(gpu_memory_utilization=0.6, max_model_len=16384)
        system = system_prompt(pub, roots)
        convs = [[{"role": "system", "content": system}, {"role": "user", "content": fsq_user_prompt(n)}] for n in rest]
        answers = engine.chat(convs, temperature=0.0, max_tokens=24, choice=pub)
        mapping.update(dict(zip(rest, answers)))
        for n, a in zip(rest, answers):
            print(f"  {n!r:42} -> {a!r} ({roots.get(a)})")
    pub_set = set(pub)
    for k, v in V2_OVERRIDES.items():
        if k in mapping and v in pub_set:
            mapping[k] = v
    text = json.dumps(mapping, indent=1, ensure_ascii=False)
    if check:
        old = CATEGORY_MAP_PATH.read_text()
        same_content = json.loads(old) == mapping
        print(f"v2 map reproduced: same entries {same_content}, same bytes {old == text}")
        if not same_content:
            sys.exit(1)
    else:
        CATEGORY_MAP_PATH.write_text(text)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen_leaves() -> tuple[list, dict, str]:
    f = ROOT / LEAVES_FILE
    digest = sha256(f)
    prereg = (ROOT / "frozen" / "datasets_v3_preregistration.sha256").read_text()
    assert f"{digest}  {LEAVES_FILE}" in prereg, "frozen leaf list differs from the pre-registered one"
    leaves = json.loads(f.read_text())["leaves"]
    pub, roots = v2_leaves()
    assert pub == sorted(leaves) and all(roots[p] == leaves[p] for p in pub), "leaf list != union of v2 catalogs"
    return sorted(leaves), leaves, digest


def gowalla_taxonomy() -> dict:
    import ast
    meta = {}

    def walk(nodes, prefix):
        for n in nodes:
            p = prefix + [n["name"]]
            k = "gowalla:" + n["url"].rsplit("/", 1)[-1]
            if k not in meta:
                meta[k] = {"name": n["name"], "path": " > ".join(p), "public_spots": 0}
            walk(n.get("spot_categories", []), p)
    walk(json.loads((RAW / GOWALLA_TREE).read_text())["spot_categories"], [])
    sp = pd.read_csv(RAW / GOWALLA_SPOTS, usecols=["spot_categories"])["spot_categories"].astype(str).value_counts()
    for s, cnt in sp.items():
        (c,) = ast.literal_eval(s)
        k = "gowalla:" + c["url"].rsplit("/", 1)[-1]
        m = meta.setdefault(k, {"name": c["name"], "path": None, "public_spots": 0})
        m["public_spots"] += int(cnt)
    return dict(sorted(meta.items(), key=lambda kv: int(kv[0].split(":")[1])))


def run_v3(reuse_llm: bool) -> None:
    pub, roots, leaves_sha = frozen_leaves()
    pub_set = set(pub)
    v2 = json.loads(CATEGORY_MAP_PATH.read_text())
    std = [r[1] for r in csv.reader((RAW / STD_CATEGORIES).open(encoding="utf-8"))]
    tsmc = set()
    for f in TSMC_FILES:
        tsmc |= set(pd.read_csv(RAW / f, sep="\t", header=None, encoding="latin-1", usecols=[3])[3].astype(str))
    fsq_domain = sorted(set(std) | tsmc)
    new_fsq = [n for n in fsq_domain if n not in v2]
    gw = gowalla_taxonomy()
    by_norm = {norm(p): p for p in pub}
    method, mapping_fsq, mapping_gw, todo = {}, dict(v2), {}, []
    for n in new_fsq:
        if norm(n) in by_norm:
            mapping_fsq[n] = by_norm[norm(n)]
            method[n] = "name"
        else:
            todo.append(("fsq", n, fsq_user_prompt(n)))
    for k, m in gw.items():
        if norm(m["name"]) in by_norm:
            mapping_gw[k] = by_norm[norm(m["name"])]
            method[k] = "name"
        else:
            todo.append(("gowalla", k, gowalla_user_prompt(m["name"], m["path"])))
    print(f"frozen leaves {len(pub)}; fsq domain {len(fsq_domain)} (v2 {len(v2)}, new {len(new_fsq)}); "
          f"gowalla ids {len(gw)}; matched by name {len(method)}; LLM {len(todo)}", flush=True)
    if reuse_llm:
        old = json.loads(CATEGORY_MAP_V3_PATH.read_text())
        answers = [old["llm_answers"][key] for _, key, _ in todo]
        llm_info = old["llm"]
    else:
        from trajpe.llm import LLMEngine
        import vllm
        t0 = time.time()
        engine = LLMEngine(gpu_memory_utilization=None, max_model_len=16384)
        system = system_prompt(pub, roots)
        convs = [[{"role": "system", "content": system}, {"role": "user", "content": prompt}] for _, _, prompt in todo]
        answers = engine.chat(convs, temperature=0.0, max_tokens=24, choice=pub)
        llm_info = {"model": engine.model, "vllm": vllm.__version__, "temperature": 0.0, "max_tokens": 24,
                    "structured_choice": LEAVES_FILE, "max_model_len": 16384, "seed": 0, "n_prompts": len(todo),
                    "seconds": round(time.time() - t0, 1), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    llm_answers = {}
    for (kind, key, _), a in zip(todo, answers):
        assert a in pub_set, (key, a)
        llm_answers[key] = a
        (mapping_fsq if kind == "fsq" else mapping_gw)[key] = a
        method[key] = "llm"
    overrides = {}
    for k, (target, reason) in V3_OVERRIDES.items():
        assert target in pub_set and k not in v2 and (k in mapping_fsq or k in mapping_gw), k
        (mapping_fsq if k in mapping_fsq else mapping_gw)[k] = target
        method[k] = "override"
        overrides[k] = {"target": target, "reason": reason}
    for k, m in gw.items():
        rule = next(((p, t) for p, t in SXSW_RULE if m["name"].startswith(p)), None)
        if rule and k not in V3_OVERRIDES:
            assert rule[1] in pub_set
            mapping_gw[k] = rule[1]
            method[k] = "override"
            overrides[k] = {"target": rule[1], "reason": f"SXSW family rule: name starts with '{rule[0]}'"}
    assert all(mapping_fsq[k] == v for k, v in v2.items()), "a v2 entry changed"
    assert set(mapping_fsq) == set(v2) | set(fsq_domain) and set(mapping_gw) == set(gw)
    assert all(v in pub_set for v in list(mapping_fsq.values()) + list(mapping_gw.values()))
    out = {"_doc": "data v3 category map (configs/datasets_v3.json, data/DATASETS_V3.md): fixed function evaluated on "
                   "full public source taxonomies against the frozen leaf list. 'fsq' = every v2 entry unchanged + "
                   "all other Semantic Trails / TSMC2014 names; 'gowalla' = every Gowalla category id. Built by "
                   "scripts/map_categories_v2_to_fsq.py --v3. MULTI_TARGET (trajpe/public_catalog.py) takes "
                   "precedence for the listed keys.",
           "leaves_file": LEAVES_FILE, "leaves_sha256": leaves_sha, "v2_map_sha256": sha256(CATEGORY_MAP_PATH),
           "sources_sha256": {STD_CATEGORIES: sha256(RAW / STD_CATEGORIES), GOWALLA_TREE: sha256(RAW / GOWALLA_TREE),
                              GOWALLA_SPOTS: sha256(RAW / GOWALLA_SPOTS),
                              **{f: sha256(RAW / f) for f in TSMC_FILES}},
           "llm": llm_info,
           "fsq": dict(sorted(mapping_fsq.items())),
           "gowalla": dict(sorted(mapping_gw.items(), key=lambda kv: int(kv[0].split(":")[1]))),
           "multi_target": {k: v for k, v in MULTI_TARGET.items() if k.startswith("gowalla:")},
           "gowalla_meta": gw,
           "method": dict(sorted(method.items())),
           "overrides": overrides,
           "llm_answers": dict(sorted(llm_answers.items()))}
    CATEGORY_MAP_V3_PATH.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    counts = pd.Series(method).value_counts().to_dict()
    print(f"wrote {CATEGORY_MAP_V3_PATH}: fsq {len(mapping_fsq)}, gowalla {len(mapping_gw)}, methods {counts}")
    for kind, key, _ in todo:
        label = key if kind == "fsq" else f"{key} {gw[key]['name']!r} [{gw[key]['path']}] n={gw[key]['public_spots']}"
        tgt = mapping_fsq.get(key) if kind == "fsq" else mapping_gw[key]
        print(f"  {label:70} -> {tgt!r} ({roots.get(tgt)}){' OVERRIDE' if key in overrides else ''}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--v3", action="store_true")
    ap.add_argument("--reuse_llm", action="store_true", help="--v3 only: reuse the stored LLM answers (no GPU)")
    ap.add_argument("--check", action="store_true", help="v2 only: compare with the existing file, do not write")
    a = ap.parse_args()
    run_v3(a.reuse_llm) if a.v3 else run_v2(a.check)
