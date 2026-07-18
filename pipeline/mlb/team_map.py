"""Canonical team-abbreviation mapping. Every MLB data source we use speaks a
different dialect for the same 30 teams:
  - Statcast (batter/pitcher game logs): AZ, CWS, KC, SD, SF, TB, WSH, ATH (all years)
  - Baseball-Reference (team_schedule_raw): ARI, CHW, KCR, SDP, SFG, TBR, WSN, OAK (pre-2025) / ATH (2025+)
  - MLB Stats API / The Odds API: full team names, e.g. "Arizona Diamondbacks"

We standardize everything to the STATCAST abbreviation, since that's what
the player-prop tables are keyed on. Discovered by comparing the actual
abbreviation sets in the pulled data -- don't assume these match without
checking; they silently don't for 7 teams."""

BR_TO_STATCAST = {
    "ARI": "AZ", "CHW": "CWS", "KCR": "KC", "OAK": "ATH",
    "SDP": "SD", "SFG": "SF", "TBR": "TB", "WSN": "WSH",
}


def br_to_statcast(abbr: str) -> str:
    return BR_TO_STATCAST.get(abbr, abbr)


FULL_NAME_TO_STATCAST = {
    "Arizona Diamondbacks": "AZ", "Atlanta Braves": "ATL", "Baltimore Orioles": "BAL",
    "Boston Red Sox": "BOS", "Chicago Cubs": "CHC", "Chicago White Sox": "CWS",
    "Cincinnati Reds": "CIN", "Cleveland Guardians": "CLE", "Colorado Rockies": "COL",
    "Detroit Tigers": "DET", "Houston Astros": "HOU", "Kansas City Royals": "KC",
    "Los Angeles Angels": "LAA", "Los Angeles Dodgers": "LAD", "Miami Marlins": "MIA",
    "Milwaukee Brewers": "MIL", "Minnesota Twins": "MIN", "New York Mets": "NYM",
    "New York Yankees": "NYY", "Athletics": "ATH", "Oakland Athletics": "ATH",
    "Philadelphia Phillies": "PHI", "Pittsburgh Pirates": "PIT", "San Diego Padres": "SD",
    "San Francisco Giants": "SF", "Seattle Mariners": "SEA", "St. Louis Cardinals": "STL",
    "Tampa Bay Rays": "TB", "Texas Rangers": "TEX", "Toronto Blue Jays": "TOR",
    "Washington Nationals": "WSH",
}


def full_name_to_statcast(name: str) -> str:
    if name not in FULL_NAME_TO_STATCAST:
        raise KeyError(f"Unknown MLB team full name: {name!r} -- add it to FULL_NAME_TO_STATCAST")
    return FULL_NAME_TO_STATCAST[name]
