"""
Réassignation optimale des camps d'entraînement (Coupe du Monde 2026).

On utilise le pool de camps disponibles listé dans la feuille `camps` du fichier
`data_fifa.xlsx` (62 sites), et on assigne à chacune des 48 équipes un camp,
de manière à minimiser la distance totale parcourue par toutes les équipes
sur leur calendrier de phase de groupes.

Pré-calcul
----------
Pour chaque couple (camp i, équipe j), on précalcule :

    cost[i, j] = Σ_{m ∈ matchs(j)}  2 · Haversine(camp_i, stade(m))

Modèle Gurobi
-------------
Variables :
    x_{i,j} ∈ {0,1}, vaut 1 si le camp i est attribué à l'équipe j

Contraintes :
    Σ_i x_{i,j}  = 1     ∀ j  (chaque équipe a exactement un camp)
    Σ_j x_{i,j} ≤ 1     ∀ i  (chaque camp est utilisé au plus une fois)

Objectif :
    min  Σ_{i,j} cost[i,j] · x_{i,j} ou autre

Sorties
-------
- Console : récap de la distance totale, comparaison au baseline (affectation
  actuelle issue de la feuille `teams`), et liste des équipes dont le camp change.
- CSV `camp_assignment.csv` : pour chaque équipe, camp optimal, distance,
  camp actuel, distance actuelle, gain.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import gurobipy as gp
from gurobipy import GRB

from models import Camp, Dataset, Team, load_dataset
from stats import haversine_km


# ---------------------------------------------------------------------------
# Pré-calcul des coûts
# ---------------------------------------------------------------------------


def precompute_costs(ds: Dataset) -> Tuple[List[Camp], List[Team], List[List[float]]]:
    """
    Renvoie (camps, teams, cost) où :
      - camps  : liste ordonnée des camps disponibles
      - teams  : liste ordonnée des équipes
      - cost   : matrice |camps| × |teams|, cost[i][j] = distance totale
                 si on assigne le camp i à l'équipe j
    """
    camps = list(ds.camps)
    teams = list(ds.teams.values())

    # On précalcule pour chaque équipe la liste de ses stades (avec lat/lon).
    team_stadiums: Dict[str, List[Tuple[float, float]]] = {}
    for t in teams:
        coords = []
        for g in ds.games_of(t.name):
            s = ds.stadiums[g.stadium_name]
            coords.append((s.lat, s.long))
        team_stadiums[t.name] = coords

    cost = [[0.0] * len(teams) for _ in range(len(camps))]
    for i, camp in enumerate(camps):
        for j, team in enumerate(teams):
            total = 0.0
            for lat_s, lon_s in team_stadiums[team.name]:
                total += 2.0 * haversine_km(camp.lat, camp.long, lat_s, lon_s)
            cost[i][j] = total

    return camps, teams, cost


# ---------------------------------------------------------------------------
# Modèle Gurobi
# ---------------------------------------------------------------------------


def _allowed_pairs(cost: List[List[float]], top_k: int) -> List[Tuple[int, int]]:
    """
    Pour chaque équipe j, garde uniquement les `top_k` camps de moindre coût.
    On augmente top_k automatiquement si l'union ne couvre pas assez de camps
    pour assurer une affectation faisable.
    """
    n_camps = len(cost)
    n_teams = len(cost[0])
    k = min(top_k, n_camps)

    while True:
        allowed: List[Tuple[int, int]] = []
        used_camps = set()
        for j in range(n_teams):
            order = sorted(range(n_camps), key=lambda i: cost[i][j])
            chosen = order[:k]
            for i in chosen:
                allowed.append((i, j))
                used_camps.add(i)
        if len(used_camps) >= n_teams or k == n_camps:
            return allowed
        k = min(k + 5, n_camps)


def solve(camps: List[Camp], teams: List[Team], cost: List[List[float]],
          top_k: int = 400) -> Tuple[Dict[str, str], float, gp.Model]:
    """
    Construit et résout le modèle d'assignation.

    Pour rester sous la limite (2000 variables) de la licence Gurobi gratuite,
    on n'instancie qu'un sous-ensemble des couples (camp, équipe) : pour chaque
    équipe, ses `top_k` camps les moins coûteux. L'optimum global est conservé
    tant que `top_k` est suffisamment grand (en pratique 40 sur ce dataset).

    Renvoie (assignement {team_name: camp_site}, distance_totale, modèle).
    """
    n_camps = len(camps)
    n_teams = len(teams)
    assert n_camps >= n_teams, (
        f"Pas assez de camps ({n_camps}) pour couvrir {n_teams} équipes"
    )

    pairs = _allowed_pairs(cost, top_k)
    print(f"Variables instanciées : {len(pairs)} "
          f"(sur {n_camps * n_teams} couples possibles)")

    m = gp.Model("camp_assignment")
    m.Params.OutputFlag = 1

    # x[i, j] = 1 si camp i affecté à équipe j  (modèle creux)
    x = m.addVars(pairs, vtype=GRB.BINARY, name="x")

    # Chaque équipe reçoit exactement un camp
    for j in range(n_teams):
        m.addConstr(
            gp.quicksum(x[i, j] for (i, jj) in pairs if jj == j) == 1,
            name=f"team_{j}_one_camp",
        )

    # Chaque camp est utilisé au plus une fois
    for i in range(n_camps):
        terms = [x[i, j] for (ii, j) in pairs if ii == i]
        if terms:
            m.addConstr(gp.quicksum(terms) <= 1,
                        name=f"camp_{i}_max_one_team")

    # Objectif : minimiser la distance totale
    m.setObjective(
        gp.quicksum(cost[i][j] * x[i, j] for (i, j) in pairs),
        GRB.MINIMIZE,
    )

    m.optimize()

    if m.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Statut Gurobi non optimal : {m.Status}")

    assignment: Dict[str, str] = {}
    for (i, j) in pairs:
        if x[i, j].X > 0.5:
            assignment[teams[j].name] = camps[i].site

    return assignment, m.ObjVal, m


# ---------------------------------------------------------------------------
# Baseline (affectation actuelle issue de la feuille teams)
# ---------------------------------------------------------------------------


def baseline_distance(ds: Dataset) -> Dict[str, float]:
    """
    Distance totale actuelle pour chaque équipe, basée sur le camp inscrit
    dans la feuille `teams`.
    """
    out: Dict[str, float] = {}
    for t in ds.teams.values():
        total = 0.0
        for g in ds.games_of(t.name):
            s = ds.stadiums[g.stadium_name]
            total += 2.0 * haversine_km(t.camp.lat, t.camp.long, s.lat, s.long)
        out[t.name] = total
    return out


# ---------------------------------------------------------------------------
# Export & reporting
# ---------------------------------------------------------------------------


def export_results(ds: Dataset, camps: List[Camp], teams: List[Team],
                   cost: List[List[float]],
                   assignment: Dict[str, str], out_csv: Path) -> None:
    camp_by_site = {c.site: (i, c) for i, c in enumerate(camps)}
    team_index = {t.name: j for j, t in enumerate(teams)}

    baseline = baseline_distance(ds)
    rows = []
    for t in teams:
        new_site = assignment[t.name]
        i_new, camp_new = camp_by_site[new_site]
        j = team_index[t.name]
        d_new = cost[i_new][j]
        d_old = baseline[t.name]
        rows.append({
            "team": t.name,
            "current_camp": t.camp.site,
            "current_city": t.camp.city,
            "current_distance_km": round(d_old, 2),
            "optimal_camp": camp_new.site,
            "optimal_city": camp_new.city,
            "optimal_distance_km": round(d_new, 2),
            "gain_km": round(d_old - d_new, 2),
            "changed": camp_new.site != t.camp.site,
        })

    rows.sort(key=lambda r: -r["gain_km"])

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

# ---------------------------------------------------------------------------
# Boxplot de comparaison des distances par équipe
# ---------------------------------------------------------------------------

def plot_distance_boxplot(ds: Dataset, camps: List[Camp], teams: List[Team],
                          cost: List[List[float]],
                          assignment: Dict[str, str], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    baseline = baseline_distance(ds)
    camp_index = {c.site: i for i, c in enumerate(camps)}

    actuel = [baseline[t.name] for t in teams]
    optimise = [cost[camp_index[assignment[t.name]]][j]
                for j, t in enumerate(teams)]

    fig, ax = plt.subplots(figsize=(7, 6))
    bp = ax.boxplot([actuel, optimise], labels=["Actuel", "Optimisé"],
                    patch_artist=True,
                    boxprops=dict(facecolor="#cfe2ff", edgecolor="#1f4e8a"),
                    medianprops=dict(color="#b3261e", linewidth=2),
                    flierprops=dict(marker="o", markerfacecolor="#b3261e",
                                    markersize=5, markeredgecolor="none"))

    # Annotation des outliers avec le nom des équipes
    for series, values in zip(bp["fliers"], [actuel, optimise]):
        xs = series.get_xdata()
        ys = series.get_ydata()
        for x, y in zip(xs, ys):
            name = next(t.name for t, v in zip(teams, values) if abs(v - y) < 1e-6)
            ax.annotate(name, xy=(x, y), xytext=(x + 0.05, y),
                        fontsize=8, va="center")

    ax.set_ylabel("Distance totale parcourue (km)")
    ax.set_title("Répartition de la distance par équipe\n(48 équipes)")
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Boxplot écrit : {out_path}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    here = Path(__file__).parent
    ds = load_dataset(here / "data_fifa.xlsx")

    print(f"{len(ds.camps)} camps disponibles, {len(ds.teams)} équipes")

    camps, teams, cost = precompute_costs(ds)
    assignment, total_opt, _ = solve(camps, teams, cost)

    baseline = baseline_distance(ds)
    total_base = sum(baseline.values())

    print()
    print("=" * 60)
    print(f"Distance totale (actuelle)  : {total_base:>12,.1f} km")
    print(f"Distance totale (optimisée) : {total_opt:>12,.1f} km")
    print(f"Gain                        : {total_base - total_opt:>12,.1f} km"
          f"  ({100 * (total_base - total_opt) / total_base:.1f} %)")
    print("=" * 60)

    n_changed = sum(1 for t in teams if assignment[t.name] != t.camp.site)
    print(f"\n{n_changed} équipes sur {len(teams)} changent de camp.")

    print("\nTop 10 plus gros gains :")
    print(f"  {'équipe':<25}{'ancien camp':<40}{'nouveau camp':<40}{'gain (km)':>10}")
    rows = []
    for t in teams:
        new = assignment[t.name]
        if new == t.camp.site:
            continue
        d_old = baseline[t.name]
        i_new = next(i for i, c in enumerate(camps) if c.site == new)
        j = teams.index(t)
        d_new = cost[i_new][j]
        rows.append((t.name, t.camp.site, new, d_old - d_new))
    rows.sort(key=lambda r: -r[3])
    for name, old, new, gain in rows[:10]:
        print(f"  {name:<25}{old[:38]:<40}{new[:38]:<40}{gain:>10,.1f}")

    out_csv = here / "camp_assignment_sum.csv"
    export_results(ds, camps, teams, cost, assignment, out_csv)
    plot_distance_boxplot(ds, camps, teams, cost, assignment, here / "boxplot_distance_sum.png")
    print(f"\nCSV écrit : {out_csv}")


if __name__ == "__main__":
    main()
