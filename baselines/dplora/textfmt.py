from __future__ import annotations

import re

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
CITY_NAMES = {"nyc": "New York City, USA", "istanbul": "Istanbul, Turkey", "tokyo": "Tokyo, Japan",
              "jakarta": "Jakarta, Indonesia", "sao_paulo": "São Paulo, Brazil", "kuwait_city": "Kuwait City, Kuwait"}
CITY_NAMES.update({"tsmc_nyc": "New York City, USA", "petaling_jaya": "Petaling Jaya, Malaysia", "moscow": "Moscow, Russia", "bandung": "Bandung, Indonesia", "sydney": "Sydney, Australia", "melbourne": "Melbourne, Australia", "tangerang": "Tangerang, Indonesia", "palembang": "Palembang, Indonesia", "shanghai": "Shanghai, China", "beijing": "Beijing, China", "ms_tokyo": "Tokyo, Japan", "gw_stockholm": "Stockholm, Sweden", "gw_san_francisco": "San Francisco, USA"})
MAX_LEN = 20
MIN_LEN = 2
SEP = " | "
_TIME = re.compile(r"^([01][0-9]|2[0-3]):([0-5][0-9])$")
_COORD = re.compile(r"^(-?\d{1,3}\.\d{1,6}),(-?\d{1,3}\.\d{1,6})$")


def messages(city: str) -> list[dict]:
    name = CITY_NAMES[city]
    return [{"role": "system", "content": f"You generate realistic Foursquare check-in trails in {name}."},
            {"role": "user", "content": "Generate one check-in trail: the day of the week, then one line per "
                                        "check-in as HH:MM | category | lat,lon."}]


def render(weekday: int, hhmm: list[tuple[int, int]], cats: list[str], lats, lons, decimals: int = 3) -> str:
    lines = [f"Day: {WEEKDAYS[weekday]}"]
    for (h, m), c, la, lo in zip(hhmm, cats, lats, lons):
        if "|" in c or "\n" in c:
            raise ValueError(f"category {c!r} clashes with the separator")
        lines.append(f"{h:02d}:{m:02d}{SEP}{c}{SEP}{float(la):.{decimals}f},{float(lo):.{decimals}f}")
    return "\n".join(lines)


def serialize_frame(df, decimals: int = 3) -> list[str]:
    df = df.sort_values(["trail_id", "pos"], kind="stable")
    t = df["t_local"]
    hh, mm = t.dt.hour.to_numpy(), t.dt.minute.to_numpy()
    dow = t.dt.dayofweek.to_numpy()
    cats = df["category"].astype(str).to_numpy()
    lat, lon = df["lat"].to_numpy(), df["lon"].to_numpy()
    codes = df["trail_id"].to_numpy()
    out, start = [], 0
    n = len(df)
    for i in range(1, n + 1):
        if i == n or codes[i] != codes[start]:
            sl = slice(start, i)
            out.append(render(int(dow[start]), list(zip(hh[sl].tolist(), mm[sl].tolist())), cats[sl].tolist(),
                              lat[sl], lon[sl], decimals))
            start = i
    return out


def parse(text: str, categories: set, finished: bool = True):
    lines = text.split("\n")
    if not finished:
        lines = lines[:-1]
    lines = [ln.strip() for ln in lines]
    while lines and not lines[-1]:
        lines.pop()
    if not lines or not lines[0].startswith("Day: "):
        return None, "bad_day"
    day = lines[0][5:].strip()
    if day not in WEEKDAYS:
        return None, "bad_day"
    cks = []
    for ln in lines[1:]:
        parts = ln.split(SEP)
        if len(parts) != 3:
            return None, "bad_fields"
        tm, cat, xy = (p.strip() for p in parts)
        if not _TIME.match(tm):
            return None, "bad_time"
        if cat not in categories:
            return None, "bad_category"
        m = _COORD.match(xy)
        if not m:
            return None, "bad_coord"
        cks.append({"time": tm, "category": cat, "lat": float(m.group(1)), "lon": float(m.group(2))})
    if not finished and len(cks) < MAX_LEN:
        return None, "unfinished"
    if len(cks) < MIN_LEN:
        return None, "too_short"
    return (WEEKDAYS.index(day), cks[:MAX_LEN]), "ok"


def clean(text: str, categories: set, finished: bool = True):
    lines = text.split("\n")
    if not finished:
        lines = lines[:-1]
    lines = [ln.strip() for ln in lines]
    while lines and not lines[-1]:
        lines.pop()
    if not lines or not lines[0].startswith("Day: ") or lines[0][5:].strip() not in WEEKDAYS:
        return None, "bad_day", 0
    keep, dropped = [f"Day: {lines[0][5:].strip()}"], 0
    for ln in lines[1:]:
        parts = ln.split(SEP)
        ok = len(parts) == 3
        if ok:
            tm, cat, xy = (p.strip() for p in parts)
            ok = bool(_TIME.match(tm)) and cat in categories and bool(_COORD.match(xy))
        if ok:
            keep.append(f"{tm}{SEP}{cat}{SEP}{xy}")
        else:
            dropped += 1
    n = len(keep) - 1
    if not finished and n < MAX_LEN:
        return None, "unfinished", dropped
    if n < MIN_LEN:
        return None, "too_short", dropped
    return "\n".join(keep), "ok", dropped


def ebnf(categories, decimals: int = 3, max_lines: int = MAX_LEN) -> str:
    def lit(s: str) -> str:
        if '"' in s or "\\" in s or "\n" in s:
            raise ValueError(f"category {s!r} cannot be a grammar literal")
        return '"' + s + '"'
    cats = sorted(set(categories))
    return "\n".join([
        f'root ::= "Day: " day line{{1,{max_lines}}}',
        "day ::= " + " | ".join(lit(d) for d in WEEKDAYS),
        f'line ::= "\\n" time "{SEP}" cat "{SEP}" coord "," coord',
        'time ::= [01] [0-9] ":" [0-5] [0-9] | "2" [0-3] ":" [0-5] [0-9]',
        "cat ::= " + " | ".join(lit(c) for c in cats),
        f'coord ::= "-"? [0-9] [0-9]? [0-9]? "." [0-9]{{{decimals}}}',
    ]) + "\n"
