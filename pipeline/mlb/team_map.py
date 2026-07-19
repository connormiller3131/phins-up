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


# MLB Stats API numeric team IDs, keyed by Statcast abbreviation. Confirmed
# against /api/v1/teams -- its own `abbreviation` field matches our Statcast
# convention exactly for all 30 teams, no translation needed there.
STATCAST_TO_MLB_TEAM_ID = {
    "ATH": 133, "PIT": 134, "SD": 135, "SEA": 136, "SF": 137, "STL": 138,
    "TB": 139, "TEX": 140, "TOR": 141, "MIN": 142, "PHI": 143, "ATL": 144,
    "CWS": 145, "MIA": 146, "NYY": 147, "MIL": 158, "LAA": 108, "AZ": 109,
    "BAL": 110, "BOS": 111, "CHC": 112, "CIN": 113, "CLE": 114, "COL": 115,
    "DET": 116, "HOU": 117, "KC": 118, "LAD": 119, "WSH": 120, "NYM": 121,
}


# Real, fixed 2025-current division alignment (Statcast abbreviations).
# Used as a win-probability model feature (division_game) -- backtested to
# improve held-out Brier and accuracy before being deployed.
DIVISIONS = {
    "NYY": "AL East", "BOS": "AL East", "TOR": "AL East", "BAL": "AL East", "TB": "AL East",
    "CLE": "AL Central", "MIN": "AL Central", "KC": "AL Central", "CWS": "AL Central", "DET": "AL Central",
    "HOU": "AL West", "SEA": "AL West", "TEX": "AL West", "LAA": "AL West", "ATH": "AL West",
    "ATL": "NL East", "PHI": "NL East", "NYM": "NL East", "MIA": "NL East", "WSH": "NL East",
    "MIL": "NL Central", "CHC": "NL Central", "STL": "NL Central", "CIN": "NL Central", "PIT": "NL Central",
    "LAD": "NL West", "SD": "NL West", "AZ": "NL West", "SF": "NL West", "COL": "NL West",
}


def same_division(team_a: str, team_b: str) -> bool:
    return DIVISIONS.get(team_a) == DIVISIONS.get(team_b)
