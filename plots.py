"""
Boxplots des statistiques par équipe.

Pour chaque mesure produite par `stats.compute_all`, on génère :
  - une figure dédiée (`box_<mesure>.png`)
  - une figure récapitulative `boxplots_all.png` (toutes les mesures côte à côte)

Les équipes au-delà des moustaches sont annotées par leur nom.
"""

from __future__ import annotations

from dataclasses import asdict, fields
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt

from models import load_dataset
from stats import TeamStats, compute_all


# Champs à tracer : tout sauf `team` et `matches`
EXCLUDE = {"team", "matches"}

# Libellés et unités pour des titres lisibles
LABELS = {
    "distance_totale_km": ("Distance totale parcourue", "km"),
    "temperature_totale_c": ("Température cumulée des matchs", "°C"),
    "altitude_totale_m": ("Altitude cumulée des stades", "m"),
    "humidite_totale_pct": ("Humidité cumulée", "%"),
    "jours_de_repos_total": ("Jours de repos cumulés", "jours"),
    "jours_de_repos_ecart_type": ("Écart-type des jours de repos", "jours"),
    "changements_fuseau": ("Changements de fuseau cumulés", "h"),
    "matchs_mieux_reposes": ("Matchs mieux reposés que l'adversaire", "matchs"),
}


def _measure_fields() -> List[str]:
    return [f.name for f in fields(TeamStats) if f.name not in EXCLUDE]


def _annotate_outliers(ax, values, names, q1, q3):
    """Ajoute le nom des équipes outliers à côté du point."""
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    for v, n in zip(values, names):
        if v < lo or v > hi:
            ax.annotate(n, xy=(1, v), xytext=(1.08, v),
                        fontsize=8, va="center")


def plot_single(stats: List[TeamStats], field_name: str, out_dir: Path) -> Path:
    values = [getattr(s, field_name) for s in stats]
    names = [s.team for s in stats]
    label, unit = LABELS.get(field_name, (field_name, ""))

    fig, ax = plt.subplots(figsize=(5, 6))
    bp = ax.boxplot(values, vert=True, patch_artist=True,
                    boxprops=dict(facecolor="#cfe2ff", edgecolor="#1f4e8a"),
                    medianprops=dict(color="#b3261e", linewidth=2),
                    flierprops=dict(marker="o", markerfacecolor="#b3261e",
                                    markersize=5, markeredgecolor="none"))

    # Annotations d'équipes hors moustaches
    q1, q3 = bp["whiskers"][0].get_ydata()[0], bp["whiskers"][1].get_ydata()[0]
    _annotate_outliers(ax, values, names, q1, q3)

    ax.set_title(label)
    ax.set_ylabel(unit)
    ax.set_xticks([1])
    ax.set_xticklabels([f"n = {len(values)} équipes"])
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    fig.tight_layout()
    path = out_dir / f"box_{field_name}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_all(stats: List[TeamStats], out_dir: Path) -> Path:
    """Une seule figure avec un sous-graphe par mesure."""
    measures = _measure_fields()
    ncols = 4
    nrows = (len(measures) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4.2 * nrows))
    axes = axes.flatten()

    for ax, field_name in zip(axes, measures):
        values = [getattr(s, field_name) for s in stats]
        names = [s.team for s in stats]
        label, unit = LABELS.get(field_name, (field_name, ""))
        bp = ax.boxplot(values, vert=True, patch_artist=True,
                        boxprops=dict(facecolor="#cfe2ff", edgecolor="#1f4e8a"),
                        medianprops=dict(color="#b3261e", linewidth=2),
                        flierprops=dict(marker="o", markerfacecolor="#b3261e",
                                        markersize=4, markeredgecolor="none"))
        q1 = bp["whiskers"][0].get_ydata()[0]
        q3 = bp["whiskers"][1].get_ydata()[0]
        _annotate_outliers(ax, values, names, q1, q3)
        ax.set_title(label, fontsize=10)
        ax.set_ylabel(unit, fontsize=9)
        ax.set_xticks([])
        ax.grid(axis="y", linestyle=":", alpha=0.5)

    # cache les axes restants
    for ax in axes[len(measures):]:
        ax.axis("off")

    fig.suptitle("Distribution des indicateurs par équipe — phase de groupes 2026",
                 fontsize=13, y=1.0)
    fig.tight_layout()
    path = out_dir / "boxplots_all.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


if __name__ == "__main__":
    here = Path(__file__).parent
    ds = load_dataset(here / "data_fifa.xlsx")
    stats = compute_all(ds)

    out_dir = here / "plots"
    out_dir.mkdir(exist_ok=True)

    paths = [plot_single(stats, name, out_dir) for name in _measure_fields()]
    summary = plot_all(stats, here)

    print("Boxplots individuels :")
    for p in paths:
        print(" -", p)
    print("Récapitulatif :", summary)
