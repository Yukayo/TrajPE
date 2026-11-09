import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from trajpe.catalog import ROOTS
from trajpe.data import V2_DATASETS as DATASETS, load_pois
from trajpe.llm import LLMEngine
from trajpe.paths import PROCESSED

ROOT_GUIDE = """Top-level Foursquare categories:
- Arts & Entertainment: museums, galleries, theaters, cinemas, concert halls, stadiums, arenas, casinos, zoos, aquariums, theme parks, bowling alleys, arcades.
- College & University: any college or university building or facility, including campus residence halls, college cafeterias, college libraries and labs.
- Event: temporary happenings such as conferences, festivals, parades, street fairs.
- Food: places mainly for eating or non-alcoholic drinks: restaurants of every cuisine, cafés, coffee shops, coffeehouses, tea rooms, bakeries, delis / bodegas, dessert and ice cream shops, food trucks, food courts.
- Nightlife Spot: bars, pubs, beer gardens, breweries, wine bars, lounges, nightclubs, karaoke bars.
- Outdoors & Recreation: parks, plazas, gardens, beaches, trails, sports fields, gyms and fitness centers, pools, playgrounds, scenic lookouts, rivers, lakes, mountains, ski areas, and municipalities such as neighborhoods, cities, towns and villages.
- Professional & Other Places: offices, workplaces, coworking spaces, government buildings, hospitals, doctors and other medical places, primary and high schools, places of worship, community centers, factories, convention centers, public libraries.
- Residence: homes, apartments, residential buildings, housing developments, assisted living.
- Shop & Service: stores and shops of every kind, malls, markets, supermarkets, salons, spas, banks, gas stations, laundry, repair and other services, post offices.
- Travel & Transport: airports, train / metro / subway / bus / tram stations and stops, ferries, taxis, roads, bridges, tunnels, hotels and other lodging, travel agencies, rest areas, parking."""

if __name__ == "__main__":
    names = sorted(set(pd.concat([load_pois(d)["category"] for d in DATASETS]).astype(str)))
    engine = LLMEngine()
    convs = [[
        {"role": "system", "content": "You are an expert on the Foursquare venue category taxonomy."},
        {"role": "user", "content": ROOT_GUIDE + f'\n\nFoursquare venue category: "{n}"\n'
                                    "Which top-level Foursquare category does it belong to? "
                                    "Answer with the top-level category name only."},
    ] for n in names]
    answers = engine.chat(convs, temperature=0.0, max_tokens=16, choice=ROOTS)
    mapping = dict(zip(names, answers))
    for n in ["Neighborhood", "City", "Town", "Village", "County", "State", "Country"]:
        if n in mapping:
            mapping[n] = "Outdoors & Recreation"
    (PROCESSED / "category_roots.json").write_text(json.dumps(mapping, indent=1, ensure_ascii=False))
    counts = pd.Series(answers).value_counts()
    print(counts.to_string())
    for probe in ["Coffee Shop", "Subway", "Home (private)", "Office", "Gym / Fitness Center", "University",
                  "Bar", "Park", "Train Station", "Mall", "Movie Theater", "Neighborhood"]:
        print(probe, "->", mapping.get(probe))
