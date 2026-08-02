"""Shared season-simulation engine: given each team's current Elo rating and
the real remaining schedule, Monte Carlo simulate the rest of the season many
times to get a distribution of final win totals per team.

Ratings are held fixed at their current value through the whole simulation --
no in-simulation rating updates. Updating ratings game-by-game during the
simulation would need a simulated margin of victory too (the real rating-
update formulas in each sport's elo_model.py are margin-sensitive), which
would need its own margin model this project doesn't have. Whether the
fixed-rating approximation is good enough for the win-total distribution it
needs to produce is checked empirically per sport (see each sport's
backtest_season_sim.py), not assumed."""
import numpy as np


def simulate_remaining_wins(remaining_games, current_ratings, home_adv, scale, n_sims=5000, seed=0):
    """remaining_games: DataFrame with home_team/away_team columns, every
    team already a key in current_ratings.
    current_ratings: {team: rating}.
    Returns (teams, sim_wins): teams is the sorted list of team codes,
    sim_wins is an (n_sims, len(teams)) int array of each team's simulated
    win count from ONLY these remaining games -- add each team's real
    current win count yourself to get final projected season win totals."""
    rng = np.random.default_rng(seed)
    teams = sorted(current_ratings.keys())
    idx = {t: i for i, t in enumerate(teams)}

    home_idx = remaining_games["home_team"].map(idx).values.astype(int)
    away_idx = remaining_games["away_team"].map(idx).values.astype(int)
    ratings = np.array([current_ratings[t] for t in teams])

    diff = (ratings[home_idx] + home_adv) - ratings[away_idx]
    p_home = 1.0 / (1.0 + 10 ** (-diff / scale))

    n_games = len(remaining_games)
    home_wins = rng.random((n_sims, n_games)) < p_home

    sim_wins = np.zeros((n_sims, len(teams)), dtype=int)
    for g in range(n_games):
        sim_wins[:, home_idx[g]] += home_wins[:, g]
        sim_wins[:, away_idx[g]] += ~home_wins[:, g]

    return teams, sim_wins


def shrink_toward_field(rate, n_teams_in_group, shrinkage):
    """A fixed-rating, independent-per-game simulation measurably
    overstates confidence in its own high-end picks -- it can't capture
    real-world sources of variance like rating-estimation uncertainty,
    injuries/trades, or a team's current form genuinely reversing
    (confirmed on real data: pipeline/mlb/backtest_season_sim.py, pooled
    across 5 real MLB seasons x 2 snapshots each -- the raw simulator's
    80-100%-confidence picks actually won the division only 75% of the
    time). shrinkage in [0,1] pulls rate toward the field's naive
    1/n_teams_in_group baseline; shrinkage=1 returns rate unchanged,
    shrinkage=0 returns the flat baseline. The right shrinkage value is
    sport-specific (division size differs) and must be derived from that
    sport's own backtest, not assumed -- see each sport's
    backtest_season_sim.py for how its value was chosen and validated
    out-of-sample."""
    baseline = 1.0 / n_teams_in_group
    return shrinkage * rate + (1 - shrinkage) * baseline


def division_title_rates(teams, final_wins, divisions):
    """final_wins: (n_sims, len(teams)) total projected win counts.
    divisions: {team: division_name}.
    Returns {division_name: {team: title_rate}} -- fraction of simulations
    where that team has the most wins in its division (random tiebreak on
    exact ties, same simplification as the real leagues' more complex
    tiebreaker rules -- flagged, not silently assumed equivalent)."""
    n_sims = final_wins.shape[0]
    by_division = {}
    for team, div in divisions.items():
        by_division.setdefault(div, []).append(team)

    rng = np.random.default_rng(0)
    rates = {}
    for div, div_teams in by_division.items():
        cols = [teams.index(t) for t in div_teams]
        sub = final_wins[:, cols]
        # random tiny tiebreak noise so np.argmax doesn't always resolve
        # exact ties toward the same (first) team
        noisy = sub + rng.random(sub.shape) * 1e-6
        winner_col = np.argmax(noisy, axis=1)
        counts = np.bincount(winner_col, minlength=len(div_teams))
        rates[div] = {t: round(counts[i] / n_sims, 4) for i, t in enumerate(div_teams)}
    return rates


def playoff_berth_rates(teams, final_wins, conferences, berths_per_conference, division_winners_guaranteed=None,
                         top_n_per_division=1):
    """conferences: {team: conference_name}.
    berths_per_conference: total playoff spots per conference (e.g. NFL 7,
    NHL 8, MLB 6 -- callers pass their own real current format).
    division_winners_guaranteed: optional {team: division_name} -- if given,
    the top top_n_per_division team(s) in each division (per trial) are
    guaranteed a berth before filling remaining spots by win total (matches
    how every 3 sports here actually seed their playoffs -- NFL/MLB
    guarantee just the division winner, top_n_per_division=1; NHL guarantees
    the top 3 in each of its 2 divisions per conference, so callers pass
    top_n_per_division=3 there); if division_winners_guaranteed is omitted,
    berths are awarded purely by win total within the conference.
    Returns {team: berth_rate}."""
    n_sims = final_wins.shape[0]
    by_conf = {}
    for team, conf in conferences.items():
        by_conf.setdefault(conf, []).append(team)

    berth_counts = {t: 0 for t in conferences}
    for conf, conf_teams in by_conf.items():
        cols = [teams.index(t) for t in conf_teams]
        sub = final_wins[:, cols]
        n_berths = berths_per_conference[conf] if isinstance(berths_per_conference, dict) else berths_per_conference

        if division_winners_guaranteed:
            conf_divisions = {}
            for t in conf_teams:
                conf_divisions.setdefault(division_winners_guaranteed[t], []).append(t)
            for trial in range(n_sims):
                trial_wins = sub[trial]
                guaranteed = set()
                for div_teams in conf_divisions.values():
                    div_cols = [conf_teams.index(t) for t in div_teams]
                    div_order = np.argsort(trial_wins[div_cols])[::-1]
                    for i in div_order[:top_n_per_division]:
                        guaranteed.add(div_teams[i])
                remaining_slots = n_berths - len(guaranteed)
                remaining_teams = [t for t in conf_teams if t not in guaranteed]
                remaining_wins = [trial_wins[conf_teams.index(t)] for t in remaining_teams]
                order = np.argsort(remaining_wins)[::-1]
                wildcards = {remaining_teams[i] for i in order[:remaining_slots]}
                for t in guaranteed | wildcards:
                    berth_counts[t] += 1
        else:
            for trial in range(n_sims):
                order = np.argsort(sub[trial])[::-1]
                for i in order[:n_berths]:
                    berth_counts[conf_teams[i]] += 1

    return {t: round(c / n_sims, 4) for t, c in berth_counts.items()}
