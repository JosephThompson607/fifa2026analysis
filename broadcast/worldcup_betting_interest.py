"""
worldcup_betting_interest.py
----------------------------
Derive per-country / per-match signals for World Cup games from *betting
markets*, using The Odds API (https://the-odds-api.com).

What betting data actually measures
-----------------------------------
Betting odds are NOT a direct measure of popular *interest*; they encode the
market's view of the likely *outcome*. From them we derive:
  * implied, vig-free win probabilities per team (a "perceived strength" signal),
  * a match "closeness" score (entropy of the 1X2 distribution) -- close games
    tend to draw more attention, so this is a proxy for how compelling a fixture
    is to neutrals and to each team's fans,
  * bookmaker coverage (how many books price the match) -- a crude attention proxy.

The most *direct* betting analog to interest is traded VOLUME / liquidity
(e.g. Betfair Exchange `totalMatched`). The Odds API does not expose volume, so
see the note at the bottom of this file for extending to a betting exchange.

Install
-------
    pip install requests pandas
Get a free key at https://the-odds-api.com and set it:
    export ODDS_API_KEY=xxxxxxxx
"""

from __future__ import annotations

import os
import math
from collections import defaultdict

import requests
import pandas as pd

API_BASE = "https://api.the-odds-api.com/v4"
API_KEY = os.environ.get("ODDS_API_KEY", "PUT-YOUR-KEY-HERE")
REGIONS = "uk,eu,us"      # bookmaker regions to aggregate over
MARKETS = "h2h"           # head-to-head = 1X2 (home / draw / away)
ODDS_FORMAT = "decimal"   # decimal odds make the probability maths trivial


# --------------------------------------------------------------------------
# Endpoint helpers.
# --------------------------------------------------------------------------

def find_world_cup_key() -> str:
    """Look up the active World Cup sport key instead of hard-coding it."""
    r = requests.get(f"{API_BASE}/sports", params={"apiKey": API_KEY}, timeout=30)
    r.raise_for_status()
    for sport in r.json():
        title = f"{sport.get('group', '')} {sport.get('title', '')}".lower()
        if "world cup" in title and sport.get("active"):
            print(f"Using sport key: {sport['key']}  ({sport['title']})")
            return sport["key"]
    print("World Cup not found in active sports; falling back to "
          "'soccer_fifa_world_cup'.")
    return "soccer_fifa_world_cup"


def fetch_odds(sport_key: str) -> list[dict]:
    """Current odds for all upcoming / live matches of the tournament."""
    r = requests.get(
        f"{API_BASE}/sports/{sport_key}/odds",
        params={"apiKey": API_KEY, "regions": REGIONS,
                "markets": MARKETS, "oddsFormat": ODDS_FORMAT},
        timeout=30,
    )
    r.raise_for_status()
    print(f"Quota: used {r.headers.get('x-requests-used')}, "
          f"remaining {r.headers.get('x-requests-remaining')}")
    return r.json()


# --------------------------------------------------------------------------
# Odds -> probabilities.
# --------------------------------------------------------------------------

def consensus_probabilities(event: dict) -> dict | None:
    """
    Average decimal odds per outcome across bookmakers, convert to implied
    probabilities, then remove the bookmaker margin (overround / vig) so the
    three probabilities sum to 1.
    """
    home, away = event["home_team"], event["away_team"]
    sums, counts = defaultdict(float), defaultdict(int)

    for book in event.get("bookmakers", []):
        for market in book.get("markets", []):
            if market["key"] != "h2h":
                continue
            for outcome in market["outcomes"]:
                sums[outcome["name"]] += outcome["price"]
                counts[outcome["name"]] += 1

    if not counts:
        return None

    avg_odds = {name: sums[name] / counts[name] for name in counts}
    raw = {name: 1.0 / odds for name, odds in avg_odds.items()}   # implied prob
    overround = sum(raw.values())                                 # > 1 = the vig
    fair = {name: p / overround for name, p in raw.items()}       # de-vigged

    # entropy-based closeness, scaled so a perfect 3-way toss-up -> 1.0
    entropy = -sum(p * math.log(p) for p in fair.values() if p > 0)
    closeness = entropy / math.log(len(fair)) if len(fair) > 1 else 0.0

    return {
        "home_team": home,
        "away_team": away,
        "commence_time": event.get("commence_time"),
        "n_bookmakers": len(event.get("bookmakers", [])),
        "p_home": fair.get(home),
        "p_draw": fair.get("Draw"),
        "p_away": fair.get(away),
        "overround": overround - 1.0,    # bookmaker margin, e.g. 0.05 = 5%
        "closeness": closeness,
    }


def to_team_level(match_rows: list[dict]) -> pd.DataFrame:
    """One row per (country, match): the team's own win probability."""
    out = []
    for m in match_rows:
        out.append({"country": m["home_team"], "opponent": m["away_team"],
                    "commence_time": m["commence_time"], "venue": "home",
                    "p_win": m["p_home"], "closeness": m["closeness"]})
        out.append({"country": m["away_team"], "opponent": m["home_team"],
                    "commence_time": m["commence_time"], "venue": "away",
                    "p_win": m["p_away"], "closeness": m["closeness"]})
    return pd.DataFrame(out)


# --------------------------------------------------------------------------
# Driver.
# --------------------------------------------------------------------------

def main() -> None:
    if API_KEY in ("", "PUT-YOUR-KEY-HERE"):
        raise SystemExit("Set ODDS_API_KEY (free key at the-odds-api.com).")

    key = find_world_cup_key()
    events = fetch_odds(key)
    print(f"Fetched {len(events)} matches.")

    match_rows = [r for ev in events if (r := consensus_probabilities(ev))]
    if not match_rows:
        raise SystemExit("No h2h odds returned (tournament may not be priced yet).")

    matches = pd.DataFrame(match_rows).sort_values("commence_time")
    matches.to_csv("betting_match_probabilities.csv", index=False)
    print("\nPer-match de-vigged probabilities:")
    print(matches.to_string(index=False))

    teams = to_team_level(match_rows).sort_values(["commence_time", "country"])
    teams.to_csv("betting_team_probabilities.csv", index=False)
    print("\nPer-team win probabilities saved -> betting_team_probabilities.csv")


if __name__ == "__main__":
    main()


# --------------------------------------------------------------------------
# Extending to traded VOLUME (the most direct interest proxy)
# --------------------------------------------------------------------------
# The Odds API returns prices, not amounts wagered. To measure money/attention
# per match, use a betting *exchange*. Betfair's Exchange API exposes
# `totalMatched` (total matched stake) per market and per runner via
# listMarketCatalogue + listMarketBook. It needs a Betfair account, an
# application key, and a certificate (SSL) login, so it is heavier to set up:
#   1. listMarketCatalogue(eventTypeId="1" /* soccer */, competitionId=<WC id>)
#         -> market ids + selection ids
#   2. listMarketBook(marketIds=[...])  -> per-market `totalMatched`
# `totalMatched` rises with interest, making it a cleaner "interest" signal than
# odds alone. Odds *movement* (open vs close, available from the historical
# endpoint /v4/historical/sports/{key}/odds?date=...) captures shifting sentiment
# over time and is another useful attention signal.
