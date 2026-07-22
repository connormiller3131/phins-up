"""NHL team identity, a single-source problem unlike MLB's Statcast-vs-
Baseball-Reference split -- api-web.nhle.com is the only data source here,
so the only real mapping needed is for a franchise that changed its
abbreviation mid-history: the Arizona Coyotes relocated to Utah for the
2024-25 season (ARI -> UTA). Treated as one continuous franchise for rating
carryover, same convention as MLB's OAK -> ATH handling."""

RENAME = {
    "ARI": "UTA",
}

# The 32 current NHL team abbreviations (post-Utah, post-Seattle-expansion).
# Not currently used for validation anywhere -- kept as a reference/sanity-
# check list for whoever builds on this next.
CURRENT_TEAMS = {
    "ANA", "BOS", "BUF", "CGY", "CAR", "CHI", "COL", "CBJ", "DAL", "DET",
    "EDM", "FLA", "LAK", "MIN", "MTL", "NSH", "NJD", "NYI", "NYR", "OTT",
    "PHI", "PIT", "SJS", "SEA", "STL", "TBL", "TOR", "UTA", "VAN", "VGK",
    "WSH", "WPG",
}


def normalize_team(abbrev):
    return RENAME.get(abbrev, abbrev)
