"""
worldcup_open_interest.py
-------------------------
Measure the RELATIVE INTEREST of each country in the FIFA World Cup using two
openly available, free, per-country attention signals:

  * SEARCH    -> Google Trends, cross-country interest in "FIFA World Cup"
                 (one COUNTRY-resolution query -> values comparable across countries)
  * WIKIPEDIA -> Wikimedia Pageviews API (fully open, no key), views of each
                 country's national-team article, with an uplift-vs-baseline ratio

It merges them into one per-country table (each signal normalised 0-1, ranked,
with a `divergence` column showing where the two disagree).

Why these two: both are free and openly available, both are per-country attention
measures, and triangulating two independent signals is more trustworthy than one.
Betting odds/volume are deliberately excluded -- they measure expected outcome and
a single global money pool, not how much each *country* cares.

Install
-------
    pip install pytrends pandas requests
No API keys needed.
"""

from __future__ import annotations

import time
import random
import datetime as dt
from dataclasses import dataclass
from urllib.parse import quote

import pandas as pd
import requests
from pytrends.request import TrendReq


# ==========================================================================
# Configuration
# ==========================================================================

@dataclass
class Team:
    country: str          # canonical label used in the output
    geo: str              # Google Trends geo code (ISO-3166 country), e.g. "FR"
    wiki_project: str     # Wikipedia language edition, e.g. "fr.wikipedia.org"
    wiki_article: str     # exact (canonical) national-team article title
    population: float     # population in millions (2026 estimate)


# All 48 teams qualified for the 2026 World Cup, with their Trends geo code and
# English-Wikipedia national-team article.
#
# Why English Wikipedia for everyone: the titles follow a consistent pattern so
# all 48 actually resolve (no silent 404s from guessing local titles), and one
# edition keeps the countries mutually comparable. This makes the Wikipedia signal
# a *global-audience* attention measure; the Trends `geo` column is still genuinely
# per-country (domestic search interest). They are two different lenses on purpose.
#
# To read DOMESTIC-language interest for a team instead, point it at its own
# edition + localized title, e.g.:
#     Team("France", "FR", "fr.wikipedia.org", "Équipe de France de football")
#     Team("Brazil", "BR", "pt.wikipedia.org", "Seleção Brasileira de Futebol")
# (verify the exact title in a browser -- redirects are counted separately).
#
# Note: England and Scotland both map to geo "GB" (the UK) for the Trends snapshot,
# since Trends has no sub-UK country resolution; their Wikipedia articles still
# separate them.
TEAMS = [
    # Hosts
    Team("United States",          "US", "en.wikipedia.org", "United States men's national soccer team", 340.0),
    Team("Mexico",                 "MX", "en.wikipedia.org", "Mexico national football team", 128.0),
    Team("Canada",                 "CA", "en.wikipedia.org", "Canada men's national soccer team", 40.0),
    # UEFA
    Team("England",                "GB", "en.wikipedia.org", "England national football team", 57.0),
    Team("France",                 "FR", "en.wikipedia.org", "France national football team", 67.0),
    Team("Croatia",                "HR", "en.wikipedia.org", "Croatia national football team", 4.0),
    Team("Norway",                 "NO", "en.wikipedia.org", "Norway national football team", 5.5),
    Team("Portugal",               "PT", "en.wikipedia.org", "Portugal national football team", 10.5),
    Team("Germany",                "DE", "en.wikipedia.org", "Germany national football team", 84.0),
    Team("Netherlands",            "NL", "en.wikipedia.org", "Netherlands national football team", 17.5),
    Team("Switzerland",            "CH", "en.wikipedia.org", "Switzerland national football team", 8.7),
    Team("Scotland",               "GB", "en.wikipedia.org", "Scotland national football team", 5.5),
    Team("Spain",                  "ES", "en.wikipedia.org", "Spain national football team", 48.0),
    Team("Austria",                "AT", "en.wikipedia.org", "Austria national football team", 9.0),
    Team("Belgium",                "BE", "en.wikipedia.org", "Belgium national football team", 11.5),
    Team("Bosnia and Herzegovina", "BA", "en.wikipedia.org", "Bosnia and Herzegovina national football team", 3.3),
    Team("Sweden",                 "SE", "en.wikipedia.org", "Sweden national football team", 10.5),
    Team("Turkey",                 "TR", "en.wikipedia.org", "Turkey national football team", 86.0),
    Team("Czechia",                "CZ", "en.wikipedia.org", "Czech Republic national football team", 10.5),
    # CONCACAF (besides hosts)
    Team("Panama",                 "PA", "en.wikipedia.org", "Panama national football team", 4.4),
    Team("Curaçao",                "CW", "en.wikipedia.org", "Curaçao national football team", 0.155),
    Team("Haiti",                  "HT", "en.wikipedia.org", "Haiti national football team", 11.4),
    # CAF
    Team("Algeria",                "DZ", "en.wikipedia.org", "Algeria national football team", 44.0),
    Team("Cape Verde",             "CV", "en.wikipedia.org", "Cape Verde national football team", 0.58),
    Team("Egypt",                  "EG", "en.wikipedia.org", "Egypt national football team", 106.0),
    Team("Ghana",                  "GH", "en.wikipedia.org", "Ghana national football team", 34.0),
    Team("Ivory Coast",            "CI", "en.wikipedia.org", "Ivory Coast national football team", 28.0),
    Team("Morocco",                "MA", "en.wikipedia.org", "Morocco national football team", 37.0),
    Team("Senegal",                "SN", "en.wikipedia.org", "Senegal national football team", 17.5),
    Team("South Africa",           "ZA", "en.wikipedia.org", "South Africa national football team", 60.0),
    Team("Tunisia",                "TN", "en.wikipedia.org", "Tunisia national football team", 12.0),
    # AFC
    Team("Australia",              "AU", "en.wikipedia.org", "Australia men's national soccer team", 26.0),
    Team("Iran",                   "IR", "en.wikipedia.org", "Iran national football team", 91.0),
    Team("Japan",                  "JP", "en.wikipedia.org", "Japan national football team", 125.0),
    Team("Jordan",                 "JO", "en.wikipedia.org", "Jordan national football team", 10.0),
    Team("Uzbekistan",             "UZ", "en.wikipedia.org", "Uzbekistan national football team", 34.0),
    Team("Qatar",                  "QA", "en.wikipedia.org", "Qatar national football team", 3.2),
    Team("Saudi Arabia",           "SA", "en.wikipedia.org", "Saudi Arabia national football team", 36.0),
    Team("South Korea",            "KR", "en.wikipedia.org", "South Korea national football team", 51.0),
    # CONMEBOL
    Team("Argentina",              "AR", "en.wikipedia.org", "Argentina national football team", 47.0),
    Team("Brazil",                 "BR", "en.wikipedia.org", "Brazil national football team", 215.0),
    Team("Colombia",               "CO", "en.wikipedia.org", "Colombia national football team", 52.0),
    Team("Ecuador",                "EC", "en.wikipedia.org", "Ecuador national football team", 18.5),
    Team("Paraguay",               "PY", "en.wikipedia.org", "Paraguay national football team", 7.0),
    Team("Uruguay",                "UY", "en.wikipedia.org", "Uruguay national football team", 3.4),
    # OFC
    Team("New Zealand",            "NZ", "en.wikipedia.org", "New Zealand national football team", 5.2),
    # Inter-confederation playoff winners
    Team("DR Congo",               "CD", "en.wikipedia.org", "DR Congo national football team", 99.0),
    Team("Iraq",                   "IQ", "en.wikipedia.org", "Iraq national football team", 43.0),
]

# --- Google Trends ---
SNAPSHOT_TERM = "FIFA World Cup"     # shared term, same meaning everywhere
# Runnable now (run-up interest). During/after the event use "2026-06-11 2026-07-19".
TRENDS_TIMEFRAME = "today 3-m"
HL, TZ = "en-US", 0

# --- Wikipedia window ---
# Defaults to the most recent ~90 days so it returns data today. For tournament
# interest, set the window to ("20260611", "20260719") once the event is underway
# (see _date_windows below to switch from rolling to fixed dates).
WINDOW_DAYS = 90
USER_AGENT = "WorldCupInterest/1.0 (research; contact: you@example.com)"  # put a real contact
WIKI_API = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article"
WIKI_DELAY = 1.0   # seconds between articles (Wikimedia tightened limits in 2026)


def _norm(s: pd.Series) -> pd.Series:
    """Normalise to 0-1 as a share of the leader (top value -> 1.0)."""
    m = s.max()
    return s / m if m and m > 0 else s * 0.0


def _date_windows():
    """Recent window + an equal-length baseline immediately before it.

    To analyse the tournament itself instead, replace the body with fixed dates:
        return ("20260611", "20260719"), ("20260311", "20260418")
    """
    end = dt.date.today() - dt.timedelta(days=1)          # yesterday (today is partial)
    w_start = end - dt.timedelta(days=WINDOW_DAYS)
    b_end = w_start - dt.timedelta(days=1)
    b_start = b_end - dt.timedelta(days=WINDOW_DAYS)
    fmt = lambda d: d.strftime("%Y%m%d")
    return (fmt(w_start), fmt(end)), (fmt(b_start), fmt(b_end))


# ==========================================================================
# Google Trends -- one comparable cross-country query
# ==========================================================================

_URLLIB3_HINT = ("    Most likely the urllib3 2.x clash -- fix with:  "
                 "pip install 'urllib3<2'\n"
                 "    Skipping Trends; the Wikipedia signal below is unaffected.")


def trends_snapshot() -> dict[str, float]:
    try:
        pt = TrendReq(hl=HL, tz=TZ, timeout=(10, 25), retries=2, backoff_factor=0.5)
    except (TypeError, AttributeError) as exc:
        print(f"  ! pytrends failed to initialise ({type(exc).__name__}: {exc}).\n"
              + _URLLIB3_HINT)
        return {}
    for attempt in range(1, 6):
        try:
            pt.build_payload([SNAPSHOT_TERM], cat=0, timeframe=TRENDS_TIMEFRAME,
                             geo="", gprop="")
            df = pt.interest_by_region(resolution="COUNTRY", inc_low_vol=True,
                                       inc_geo_code=True)
            return dict(zip(df["geoCode"], df[SNAPSHOT_TERM]))
        except (TypeError, AttributeError) as exc:
            # Deterministic library/version incompatibility -- retrying won't help.
            print(f"  ! pytrends incompatibility ({type(exc).__name__}: {exc}).\n"
                  + _URLLIB3_HINT)
            return {}
        except Exception as exc:
            wait = min(60, 2 ** attempt) + random.uniform(0, 1.5)
            print(f"  [trends retry {attempt}/5] {type(exc).__name__} (throttling?); "
                  f"waiting {wait:.1f}s")
            time.sleep(wait)
    print("  ! Google Trends snapshot throttled; continuing with Wikipedia only.")
    return {}


# ==========================================================================
# Wikipedia -- open Pageviews API, no key
# ==========================================================================

def wiki_daily(project: str, article: str, start: str, end: str,
               max_tries: int = 6) -> list | None:
    """
    Daily view items for an article over [start, end] in a SINGLE request.
    Honours HTTP 429/503 Retry-After with exponential back-off (Wikimedia's 2026
    rate-limit guidance). Returns the `items` list, or None if unavailable.
    """
    title = quote(article.replace(" ", "_"), safe="")
    url = f"{WIKI_API}/{project}/all-access/user/{title}/daily/{start}/{end}"
    for attempt in range(1, max_tries + 1):
        try:
            r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
            if r.status_code == 404:
                print(f"  ! no data: {article} (check the exact title)")
                return None
            if r.status_code in (429, 503):
                ra = r.headers.get("Retry-After", "")
                wait = int(ra) if ra.isdigit() else min(120, 5 * 2 ** (attempt - 1))
                print(f"  [wiki {r.status_code}] {article}: waiting {wait}s "
                      f"({attempt}/{max_tries})")
                time.sleep(wait + random.uniform(0, 1.0))
                continue
            r.raise_for_status()
            return r.json().get("items", [])
        except requests.RequestException as exc:
            wait = min(60, 5 * attempt)
            print(f"  [wiki error] {article}: {exc} -- retry in {wait}s")
            time.sleep(wait)
    print(f"  ! gave up on {article} after {max_tries} attempts (rate limit).")
    return None


def _sum_between(items: list, lo: str, hi: str) -> int:
    """Sum daily views whose YYYYMMDD timestamp falls in [lo, hi]."""
    return sum(it["views"] for it in items if lo <= it["timestamp"][:8] <= hi)


# ==========================================================================
# Build the per-country interest table
# ==========================================================================

def main() -> None:
    (w_start, w_end), (b_start, b_end) = _date_windows()
    print(f"Wikipedia window:   {w_start}-{w_end}")
    print(f"Wikipedia baseline: {b_start}-{b_end}\n")

    print("Fetching Google Trends cross-country snapshot...")
    snap = trends_snapshot()

    print("Fetching Wikipedia pageviews (open API)...")
    rows = []
    for team in TEAMS:
        # One request per article spanning baseline+window, split locally (halves calls).
        items = wiki_daily(team.wiki_project, team.wiki_article, b_start, w_end)
        win = base = None
        if items is not None:
            win = _sum_between(items, w_start, w_end)
            base = _sum_between(items, b_start, b_end)
        lift = (win / base) if (win is not None and base) else None
        rows.append({
            "country": team.country,
            "population": team.population,
            "trends_interest": snap.get(team.geo[:2]),     # comparable 0-100
            "wiki_window_views": win,
            "wiki_baseline_views": base,
            "wiki_lift": round(lift, 3) if lift is not None else None,
        })
        time.sleep(WIKI_DELAY)  # pace requests; Wikimedia limits tightened in 2026

    df = pd.DataFrame(rows)

    # Normalise each signal to 0-1 (share of the leader).
    df["trends_norm"] = _norm(df["trends_interest"].fillna(0))
    # Use the World-Cup-driven UPLIFT as the Wikipedia interest signal: it controls
    # for how popular each team's article is at baseline. Swap to wiki_window_views
    # if you want raw attention instead.
    df["wiki_norm"] = _norm(df["wiki_lift"].fillna(0))
    df["wiki_window_views_per_capita"] = df["wiki_window_views"] / df["population"]
    df["combined"] = (df["trends_norm"] + df["wiki_norm"]) / 2
    df["combined_per_capita"] = df["combined"] / df["population"]
    df["divergence"] = df["wiki_norm"] - df["trends_norm"]
    df["trends_rank"] = df["trends_norm"].rank(ascending=False, method="min")
    df["wiki_rank"] = df["wiki_norm"].rank(ascending=False, method="min")
    df = df.sort_values("combined", ascending=False)

    df.to_csv("worldcup_open_interest.csv", index=False)
    print("\nRelative interest per country (two open signals):")
    show = ["country", "population", "trends_interest", "wiki_window_views",
            "wiki_window_views_per_capita", "wiki_lift", "trends_norm",
            "wiki_norm", "combined", "combined_per_capita", "divergence"]
    print(df[show].round(3).to_string(index=False))
    print("\nSaved -> worldcup_open_interest.csv")


if __name__ == "__main__":
    main()


# --------------------------------------------------------------------------
# Reading the output / notes
# --------------------------------------------------------------------------
# * trends_interest: 0-100, comparable across countries (one COUNTRY-resolution
#   query). Measures search interest in the term "FIFA World Cup".
# * wiki_lift: window views / baseline views of the national-team article -- the
#   World-Cup-driven uplift, which controls for baseline article popularity.
# * combined / combined_per_capita / divergence: each signal is scaled to its
#   leader, then averaged. combined is the raw average; combined_per_capita
#   normalizes by country population to surface smaller nations with high
#   per-person engagement. divergence > 0 -> Wikipedia uplift outruns search
#   interest; < 0 -> the reverse. Agreement is expected; the gaps are where to look.
#
# Caveats worth keeping in mind:
# * Both signals skew toward connected populations, so they under-measure
#   high-passion, low-connectivity regions. FIFA's post-tournament TV audience
#   reports are the closest open "ground truth" to validate against.
# * The national-team article is country-specific; to measure tournament interest
#   directly instead, point wiki_article at the localized "2026 FIFA World Cup"
#   article per language (note: same-language countries then share one article
#   and can't be separated).
# * Run it now for run-up interest; set TRENDS_TIMEFRAME to "2026-06-11
#   2026-07-19" and the Wikipedia window to fixed tournament dates during/after.
