"""
worldcup_combined.py
--------------------
Join the two FREE interest signals into one table:

  * ATTENTION   -> Google Trends search interest (via pytrends)
  * MONEY-ON-IT -> Polymarket traded volume      (via the public Gamma API)

It produces:
  1. a per-country comparison (attention vs money, normalised + ranked, with a
     `divergence` column flagging where the two disagree);
  2. a best-effort per-fixture table (each country's search-interest spike around
     a match next to that fixture's traded volume).

Both signals are free and need no betting account. Trends is geo-normalised
0-100; Polymarket is USD volume. They are NOT the same unit, so each column is
normalised to 0-1 (share of the leader) before being compared.

Install
-------
    pip install pytrends pandas requests
No API key needed.
"""

from __future__ import annotations

import json
import time
import random
from dataclasses import dataclass, field

import pandas as pd
import requests
from pytrends.request import TrendReq


# ==========================================================================
# Configuration
# ==========================================================================

@dataclass
class Team:
    country: str                 # canonical label used in the output
    geo: str                     # Google Trends geo code (ISO-3166), e.g. "FR"
    term: str                    # search term measured *inside* that country
    aliases: list[str] = field(default_factory=list)  # how Polymarket may name it

    def all_aliases(self) -> list[str]:
        return [self.country.lower()] + [a.lower() for a in self.aliases]


# Configure the teams you care about. `aliases` map Polymarket's English market
# labels onto your canonical country (e.g. Polymarket "USA" -> "United States").
# Note: England/Scotland/Wales have no ISO country code; use "GB" (captures the
# whole UK) so the cross-country snapshot can match.
TEAMS = [
    Team("France",        "FR", "equipe de France",   ["france"]),
    Team("Brazil",        "BR", "selecao",            ["brazil", "brasil"]),
    Team("Argentina",     "AR", "seleccion",          ["argentina"]),
    Team("Spain",         "ES", "seleccion espanola", ["spain", "espana"]),
    Team("England",       "GB", "England",            ["england"]),
    Team("United States", "US", "USMNT",              ["usa", "united states"]),
    Team("Mexico",        "MX", "seleccion mexicana", ["mexico"]),
    # add the rest ...
]

TIMEFRAME = "2026-06-01 2026-07-31"    # Trends window (daily resolution)
MATCH_WINDOW_DAYS = 1                   # peak interest within +/- N days of a match
SNAPSHOT_TERM = "FIFA World Cup"        # shared term for the cross-country snapshot
HL, TZ = "en-US", 0

# Polymarket
GAMMA = "https://gamma-api.polymarket.com"
MATCH_KEYWORDS = ["world cup"]
INCLUDE_CLOSED = False
PAGE_SIZE, MAX_PAGES = 100, 30
SEP_PATTERNS = [" vs. ", " vs ", " v. ", " v "]


def _norm(s: pd.Series) -> pd.Series:
    """Normalise to 0-1 as a share of the leader (top value -> 1.0)."""
    m = s.max()
    return s / m if m and m > 0 else s * 0.0


# ==========================================================================
# Google Trends
# ==========================================================================

def make_client() -> TrendReq:
    return TrendReq(hl=HL, tz=TZ, timeout=(10, 25), retries=2, backoff_factor=0.5)


def _build_with_retry(pt: TrendReq, kw, geo, timeframe, max_tries: int = 5) -> None:
    for attempt in range(1, max_tries + 1):
        try:
            pt.build_payload(kw_list=kw, cat=0, timeframe=timeframe, geo=geo, gprop="")
            return
        except Exception as exc:
            wait = min(60, 2 ** attempt) + random.uniform(0, 1.5)
            print(f"  [trends retry {attempt}/{max_tries}] {type(exc).__name__}; "
                  f"waiting {wait:.1f}s")
            time.sleep(wait)
    raise RuntimeError(f"Google Trends kept throttling for {kw} ({geo}).")


def trends_timeseries(pt: TrendReq) -> dict[str, pd.Series]:
    """Daily 0-100 interest for each team's term, measured inside its geo."""
    out: dict[str, pd.Series] = {}
    for team in TEAMS:
        try:
            _build_with_retry(pt, [team.term], team.geo, TIMEFRAME)
            df = pt.interest_over_time()
            out[team.country] = df[team.term] if not df.empty else pd.Series(dtype="float64")
        except Exception as exc:
            print(f"  ! trends timeseries failed for {team.country}: {exc}")
            out[team.country] = pd.Series(dtype="float64")
        time.sleep(random.uniform(2.0, 4.0))
    return out


def trends_snapshot(pt: TrendReq) -> dict[str, float]:
    """
    Cross-country comparable interest in SNAPSHOT_TERM, keyed by geo code.
    One query at COUNTRY resolution -> values ARE comparable across countries.
    """
    try:
        _build_with_retry(pt, [SNAPSHOT_TERM], geo="", timeframe=TIMEFRAME)
        df = pt.interest_by_region(resolution="COUNTRY", inc_low_vol=True,
                                   inc_geo_code=True)
        return dict(zip(df["geoCode"], df[SNAPSHOT_TERM]))
    except Exception as exc:
        print(f"  ! trends snapshot failed: {exc}")
        return {}


# ==========================================================================
# Polymarket (public Gamma API, no auth)
# ==========================================================================

def _loads(x):
    """Gamma returns `outcomes` / `outcomePrices` as stringified JSON lists."""
    if isinstance(x, str):
        try:
            return json.loads(x)
        except json.JSONDecodeError:
            return []
    return x or []


def fetch_polymarket_events() -> list[dict]:
    keep, offset = [], 0
    for _ in range(MAX_PAGES):
        r = requests.get(f"{GAMMA}/events",
                         params={"closed": str(INCLUDE_CLOSED).lower(),
                                 "limit": PAGE_SIZE, "offset": offset,
                                 "order": "volume", "ascending": "false"},
                         timeout=30)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        for ev in batch:
            hay = f"{ev.get('title', '')} {ev.get('slug', '')}".lower()
            if any(k in hay for k in MATCH_KEYWORDS):
                keep.append(ev)
        offset += PAGE_SIZE
        time.sleep(0.3)
        if len(batch) < PAGE_SIZE:
            break
    return keep


def split_teams(title: str):
    for sep in SEP_PATTERNS:            # most-specific separator first
        if sep in title:
            a, b = title.split(sep, 1)
            return a.strip(" ?"), b.strip(" ?")
    return None


def prob_for(team: str, pairs):
    t = team.lower()
    for label, p in pairs:
        if t and t in label.lower():
            return round(p, 4)
    return None


def event_prices(ev: dict):
    """(label, probability) pairs, robust to single multi-outcome or Yes/No layouts."""
    pairs = []
    for m in ev.get("markets", []):
        outs = _loads(m.get("outcomes"))
        prices = [float(p) for p in (_loads(m.get("outcomePrices")) or [])]
        if len(outs) == len(prices) and len(outs) > 2:
            pairs.extend(zip(outs, prices))
        elif len(outs) == 2 and len(prices) == 2:
            yes = prices[outs.index("Yes")] if "Yes" in outs else prices[0]
            pairs.append(((m.get("groupItemTitle") or m.get("question") or "").strip(), yes))
    return pairs


def polymarket_fixtures(events: list[dict]) -> pd.DataFrame:
    """Team-level fixture rows: each head-to-head event -> two rows."""
    rows = []
    for ev in events:
        teams = split_teams(ev.get("title", ""))
        if not teams:
            continue
        pairs = event_prices(ev)
        vol = float(ev.get("volume") or 0.0)
        vol24 = float(ev.get("volume24hr") or 0.0)
        for country, opp in ((teams[0], teams[1]), (teams[1], teams[0])):
            rows.append({"pm_label": country, "opponent": opp,
                         "match": ev.get("title"), "start_date": ev.get("startDate"),
                         "pm_volume_usd": vol, "pm_volume_24h_usd": vol24,
                         "pm_p_win": prob_for(country, pairs)})
    return pd.DataFrame(rows)


# ==========================================================================
# Resolve Polymarket labels -> configured teams
# ==========================================================================

def resolve(raw_label: str) -> Team | None:
    raw = (raw_label or "").lower().strip()
    for team in TEAMS:
        for a in team.all_aliases():
            if a == raw or (len(a) >= 4 and (a in raw or raw in a)):
                return team
    return None


def interest_near(series: pd.Series, when, window_days: int) -> float | None:
    """Peak Trends value within +/- window_days of `when`."""
    if series is None or series.empty:
        return None
    d = pd.to_datetime(when, errors="coerce")
    if pd.isna(d):
        return None
    d = (d.normalize().tz_localize(None) if d.tzinfo else d.normalize())
    lo, hi = d - pd.Timedelta(days=window_days), d + pd.Timedelta(days=window_days)
    idx = series.index.tz_localize(None) if series.index.tz is not None else series.index
    window = series[(idx >= lo) & (idx <= hi)]
    return float(window.max()) if not window.empty else None


# ==========================================================================
# Build the combined tables
# ==========================================================================

def main() -> None:
    # ---- Polymarket ----
    print("Fetching Polymarket World Cup events (public Gamma API)...")
    try:
        events = fetch_polymarket_events()
        fixtures = polymarket_fixtures(events)
    except Exception as exc:
        print(f"  ! Polymarket fetch failed: {exc}")
        fixtures = pd.DataFrame()
    print(f"  -> {len(fixtures)} fixture rows.")

    # ---- Google Trends (slow / rate-limited) ----
    print("Fetching Google Trends (this is the slow, rate-limited part)...")
    pt = make_client()
    ts = trends_timeseries(pt)
    snap = trends_snapshot(pt)   # geoCode -> comparable interest

    # ---- per-fixture table ----
    if not fixtures.empty:
        fixtures["country"] = fixtures["pm_label"].map(
            lambda x: (resolve(x).country if resolve(x) else x))

        def _local_interest(row):
            t = resolve(row["pm_label"])
            return interest_near(ts.get(t.country) if t else None,
                                 row["start_date"], MATCH_WINDOW_DAYS)

        fixtures["trends_interest_local"] = fixtures.apply(_local_interest, axis=1)
        # normalise each signal across the fixture table for comparison
        fixtures["pm_volume_norm"] = _norm(fixtures["pm_volume_usd"])
        fixtures["trends_norm"] = _norm(fixtures["trends_interest_local"].fillna(0))
        fixtures = fixtures.sort_values("pm_volume_usd", ascending=False)
        fixtures.to_csv("combined_per_fixture.csv", index=False)
        print("\nPer-fixture (top 15 by volume):")
        cols = ["country", "opponent", "start_date", "pm_volume_usd",
                "trends_interest_local", "pm_p_win"]
        print(fixtures[cols].head(15).to_string(index=False))

    # ---- per-country comparison ----
    pm_by_country = (fixtures.groupby("country")["pm_volume_usd"].sum()
                     if not fixtures.empty else pd.Series(dtype="float64"))
    comp = pd.DataFrame([{
        "country": team.country,
        "trends_interest": snap.get(team.geo[:2]),       # comparable 0-100
        "pm_volume_usd": float(pm_by_country.get(team.country, 0.0)),
    } for team in TEAMS])

    comp["trends_norm"] = _norm(comp["trends_interest"].fillna(0))
    comp["pm_volume_norm"] = _norm(comp["pm_volume_usd"])
    comp["combined"] = (comp["trends_norm"] + comp["pm_volume_norm"]) / 2
    # divergence > 0 => more money than attention; < 0 => more attention than money
    comp["divergence"] = comp["pm_volume_norm"] - comp["trends_norm"]
    comp["trends_rank"] = comp["trends_norm"].rank(ascending=False, method="min")
    comp["pm_rank"] = comp["pm_volume_norm"].rank(ascending=False, method="min")
    comp = comp.sort_values("combined", ascending=False)
    comp.to_csv("combined_per_country.csv", index=False)

    print("\nPer-country: attention (Trends) vs money (Polymarket):")
    show = ["country", "trends_norm", "pm_volume_norm", "combined",
            "divergence", "trends_rank", "pm_rank"]
    print(comp[show].round(3).to_string(index=False))
    print("\nSaved: combined_per_country.csv"
          + (", combined_per_fixture.csv" if not fixtures.empty else ""))


if __name__ == "__main__":
    main()


# --------------------------------------------------------------------------
# Reading the output
# --------------------------------------------------------------------------
# trends_norm / pm_volume_norm are each scaled to the leader (1.0 = the country
# with the most search interest / the most traded volume). They are different
# *units* (attention vs money), so compare ranks and the `divergence` column,
# not raw magnitudes:
#   divergence > 0  -> the market backs this team more than the public searches
#                      for it (money running ahead of attention);
#   divergence < 0  -> lots of search interest but comparatively little money.
# `trends_interest` (per-country) is comparable across countries because it comes
# from one COUNTRY-resolution query. `trends_interest_local` (per-fixture) is each
# country's own 0-100 spike around that match and is NOT comparable in absolute
# terms between countries -- normalise within each country first if you need that.
#
# To add bookmaker-implied probability as a third signal, merge in
# betting_team_probabilities.csv from the earlier The Odds API script on
# (country, match) or (country, date).
