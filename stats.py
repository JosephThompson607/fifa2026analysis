"""
Calcul de statistiques par équipe pour la Coupe du Monde FIFA 2026 (phase de groupes).

Pour chaque équipe on calcule :

  - distance_totale_km      : somme des trajets aller-retour camp -> stade -> camp
                              pour tous ses matchs (distance Haversine en km)
  - temperature_totale_c    : somme des températures moyennes de juin des stades joués
  - altitude_totale_m       : somme des altitudes des stades joués
  - humidite_totale_pct     : somme des humidités moyennes de juin des stades joués
  - jours_de_repos_total    : somme des jours STRICTEMENT entre deux matchs consécutifs
                              (le temps avant le premier match n'est pas compté)
  - jours_de_repos_ecart_type : écart-type (population) des intervalles de repos
  - changements_fuseau      : somme sur tous les matchs de |UTC_stade - UTC_camp|
  - matchs_mieux_reposes    : nombre de matchs où l'équipe arrive avec strictement
                              plus de jours de repos que son adversaire
                              (un match sans match précédent est compté comme
                              "à égalité" et n'est PAS comptabilisé)

Sortie :
  - tableau imprimé dans la console
  - fichier CSV `team_stats.csv`
"""

from __future__ import annotations

import csv
import statistics
from dataclasses import dataclass, asdict
from math import asin, cos, radians, sin, sqrt
from pathlib import Path
from typing import Dict, List, Optional

from models import Dataset, Game, Team, load_dataset


EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance grand cercle entre deux points (km)."""
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlam = radians(lon2 - lon1)
    a = sin(dphi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(dlam / 2) ** 2
    return 2 * EARTH_RADIUS_KM * asin(sqrt(a))


@dataclass
class TeamStats:
    team: str
    matches: int
    distance_totale_km: float
    temperature_totale_c: float
    altitude_totale_m: float
    humidite_totale_pct: float
    jours_de_repos_total: int
    jours_de_repos_ecart_type: float
    changements_fuseau: int
    matchs_mieux_reposes: int


def _days_rest_before(team: str, target_game: Game,
                      sorted_games: List[Game]) -> Optional[int]:
    """
    Nombre de jours strictement entre le match précédent de `team`
    et `target_game`. None si `target_game` est le premier match.
    """
    previous = None
    for g in sorted_games:
        if g is target_game:
            break
        if team in (g.team_1, g.team_2):
            previous = g
    if previous is None:
        return None
    return (target_game.date - previous.date).days - 1


def compute_team_stats(team: Team, ds: Dataset) -> TeamStats:
    games = ds.games_of(team.name)

    distance_totale = 0.0
    temperature_totale = 0.0
    altitude_totale = 0.0
    humidite_totale = 0.0
    changements_fuseau = 0

    for g in games:
        stadium = ds.stadiums[g.stadium_name]
        # aller-retour camp -> stade -> camp
        leg = haversine_km(team.camp.lat, team.camp.long,
                           stadium.lat, stadium.long)
        distance_totale += 2 * leg

        temperature_totale += stadium.temperature_c_june
        altitude_totale += stadium.altitude_m
        humidite_totale += stadium.humidity_pct_june

        changements_fuseau += abs(stadium.utc_offset - (team.camp.utc_offset or 0))

    # Jours de repos : intervalles entre matchs consécutifs.
    # Le délai avant le 1er match est ignoré.
    intervalles = [
        (games[i + 1].date - games[i].date).days - 1
        for i in range(len(games) - 1)
    ]
    jours_repos_total = sum(intervalles)
    ecart_type = statistics.pstdev(intervalles) if len(intervalles) >= 2 else 0.0

    # Matchs où l'équipe est mieux reposée que l'adversaire.
    # Si l'équipe n'a pas encore joué, on ne la compte pas.
    matchs_mieux_reposes = 0
    sorted_all = sorted(ds.games, key=lambda x: x.date)
    for g in games:
        my_rest = _days_rest_before(team.name, g, sorted_all)
        opp_name = g.team_2 if g.team_1 == team.name else g.team_1
        opp_rest = _days_rest_before(opp_name, g, sorted_all)
        if my_rest is None:
            continue                       # 1er match : pas de comparaison
        if opp_rest is None:
            # adversaire joue son 1er match -> il est "totalement reposé".
            # On ne peut pas dire qu'on est mieux reposé qu'eux.
            continue
        if my_rest > opp_rest:
            matchs_mieux_reposes += 1

    return TeamStats(
        team=team.name,
        matches=len(games),
        distance_totale_km=round(distance_totale, 2),
        temperature_totale_c=round(temperature_totale, 2),
        altitude_totale_m=round(altitude_totale, 2),
        humidite_totale_pct=round(humidite_totale, 2),
        jours_de_repos_total=jours_repos_total,
        jours_de_repos_ecart_type=round(ecart_type, 3),
        changements_fuseau=changements_fuseau,
        matchs_mieux_reposes=matchs_mieux_reposes,
    )


def compute_all(ds: Dataset) -> List[TeamStats]:
    return [compute_team_stats(t, ds) for t in ds.teams.values()]


def export_csv(stats: List[TeamStats], out_path: Path) -> None:
    fieldnames = list(asdict(stats[0]).keys())
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in stats:
            writer.writerow(asdict(s))


def print_table(stats: List[TeamStats]) -> None:
    fieldnames = list(asdict(stats[0]).keys())
    widths = {f: max(len(f), max(len(str(getattr(s, f))) for s in stats))
              for f in fieldnames}
    header = " | ".join(f.ljust(widths[f]) for f in fieldnames)
    print(header)
    print("-" * len(header))
    for s in sorted(stats, key=lambda x: x.team):
        print(" | ".join(str(getattr(s, f)).ljust(widths[f]) for f in fieldnames))


if __name__ == "__main__":
    here = Path(__file__).parent
    ds = load_dataset(here / "data_fifa.xlsx")
    stats = compute_all(ds)
    print_table(stats)
    out = here / "team_stats.csv"
    export_csv(stats, out)
    print(f"\nCSV écrit : {out}")
