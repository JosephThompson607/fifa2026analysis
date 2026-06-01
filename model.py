import pandas as pd
import gurobipy as gp
from gurobipy import GRB
from itertools import product


TIME_SLOTS = ["Morning", "Noon", "Evening", "Night"]


def load_data(path="data_fifa.xlsx"):
    teams_df    = pd.read_excel(path, sheet_name="teams")
    games_df    = pd.read_excel(path, sheet_name="games")
    stadiums_df = pd.read_excel(path, sheet_name="stadiums")
    return teams_df, games_df, stadiums_df


def build_intermediate_nodes(games_df, stadiums_df):
    """All combinations (date, stadium, time_slot): 17 days × 16 stadiums × 4 slots = 1,088."""
    dates    = sorted(games_df["date"].unique())
    stadiums = stadiums_df["stadium"].tolist()
    return list(product(dates, stadiums, TIME_SLOTS))


def build_and_solve(teams_df, games_df, stadiums_df):
    """
    Min-cost flow — schedule assignment:
      - n source nodes     (one per team, supply = 1)
      - 1,088 intermediate nodes  (date × stadium × time_slot)
      - 1 super-sink       (demand = n)

    Decision variable:
      y[g, v] = 1  if game g is assigned to slot v

    Arc groups that define valid (game, slot) pairs:
      1. Source → intermediate : day-1 games, date(v) <= last_date - 8
      2. Intermediate → intermediate : calendar ordering per team (+4 days between matchdays)
      3. Intermediate → sink  : day-3 games, date(v) >= first_date + 8
    """
    teams = teams_df["team"].tolist()
    nodes = build_intermediate_nodes(games_df, stadiums_df)

    dates       = pd.to_datetime(games_df["date"])
    first_date  = dates.min()
    last_date   = dates.max()
    src_cutoff  = last_date  - pd.Timedelta(days=8)   # arc group 1 upper bound
    sink_cutoff = first_date + pd.Timedelta(days=8)   # arc group 3 lower bound

    n = len(teams)
    m = len(nodes)
    G = len(games_df)

    # day number of each slot (1 = first_date, 17 = last_date)
    node_daynum = {
        v: (pd.to_datetime(nodes[v][0]) - first_date).days + 1
        for v in range(m)
    }
    node_date = {v: pd.to_datetime(nodes[v][0]) for v in range(m)}

    # Games as (team_1, team_2, matchday, group) indexed 0..G-1
    games = list(games_df[["team_1", "team_2", "day", "group"]].itertuples(index=False, name=None))

    # For each (team, matchday) → game index
    team_day_to_game = {}
    for g, (t1, t2, d, grp) in enumerate(games):
        team_day_to_game[(t1, d)] = g
        team_day_to_game[(t2, d)] = g

    # Games per (group, matchday): (grp, d) → [g_idx, ...]
    group_day_games = {}
    for g, (t1, t2, d, grp) in enumerate(games):
        group_day_games.setdefault((grp, d), []).append(g)

    group_day3 = {grp: idxs for (grp, d), idxs in group_day_games.items() if d == 3}

    # Valid slots per game (arc group 1 and 3 restrict day-1 and day-3)
    def valid_slots_for(matchday):
        if matchday == 1:
            return [v for v in range(m) if node_date[v] <= src_cutoff]
        elif matchday == 3:
            return [v for v in range(m) if node_date[v] >= sink_cutoff]
        else:
            return list(range(m))

    game_slots = {g: valid_slots_for(games[g][2]) for g in range(G)}

    # Reverse index: slot → list of games that can go there
    slot_games = {v: [] for v in range(m)}
    for g, slots in game_slots.items():
        for v in slots:
            slot_games[v].append(g)

    print(f"Teams             : {n}")
    print(f"Games             : {G}")
    print(f"Intermediate nodes: {m}  (17 × 16 × 4)")
    print(f"src_cutoff        : {src_cutoff.date()}  (last_date - 8,  arc group 1)")
    print(f"sink_cutoff       : {sink_cutoff.date()}  (first_date + 8, arc group 3)\n")

    # --- Model ---------------------------------------------------------------
    model = gp.Model("fifa_flow")
    model.Params.LogToConsole = 0

    # y[g, v] = 1 if game g assigned to slot v
    y = model.addVars(
        [(g, v) for g in range(G) for v in game_slots[g]],
        vtype=GRB.BINARY, name="y"
    )

    # Arc group 1 — each game assigned to exactly one slot
    model.addConstrs(
        (gp.quicksum(y[g, v] for v in game_slots[g]) == 1
         for g in range(G)),
        name="assign"
    )

    # Each slot hosts at most one game
    model.addConstrs(
        (gp.quicksum(y[g, v] for g in slot_games[v]) <= 1
         for v in range(m) if slot_games[v]),
        name="capacity"
    )

    # At most one game per (date, stadium) across all time slots
    date_stadium_slots = {}
    for v, (date, stadium, _) in enumerate(nodes):
        date_stadium_slots.setdefault((date, stadium), []).append(v)

    model.addConstrs(
        (gp.quicksum(y[g, v] for v in date_stadium_slots[ds] for g in slot_games[v] if slot_games[v]) <= 1
         for ds in date_stadium_slots),
        name="stadium_day"
    )

    # Arc group 2 — calendar ordering: consecutive matchdays of same team ≥ 4 days apart
    # sum_v daynum(v)*y[g2,v]  >=  sum_v daynum(v)*y[g1,v]  +  4
    for team in teams:
        for d in [1, 2]:
            g1 = team_day_to_game.get((team, d))
            g2 = team_day_to_game.get((team, d + 1))
            if g1 is None or g2 is None:
                continue
            model.addConstr(
                gp.quicksum(node_daynum[v] * y[g2, v] for v in game_slots[g2])
                - gp.quicksum(node_daynum[v] * y[g1, v] for v in game_slots[g1])
                >= 4,
                name=f"order_{team}_d{d}"
            )

    # Simultaneity — day-3 games in the same group must share (date, time_slot)
    # For each (date, time_slot): sum y[g1,v] == sum y[g2,v]
    date_slot_nodes = {}
    for v in range(m):
        key = (nodes[v][0], nodes[v][2])   # (date, time_slot)
        date_slot_nodes.setdefault(key, []).append(v)

    for grp, grp_games in group_day3.items():
        g1, g2 = grp_games
        for (date, slot), vs in date_slot_nodes.items():
            vs1 = [v for v in vs if v in set(game_slots[g1])]
            vs2 = [v for v in vs if v in set(game_slots[g2])]
            if not vs1 and not vs2:
                continue
            model.addConstr(
                gp.quicksum(y[g1, v] for v in vs1)
                == gp.quicksum(y[g2, v] for v in vs2),
                name=f"simul_{grp}_{date}_{slot}"
            )

    # Proximity — day-1 and day-2 same-group games must be within 1 calendar day
    # and NOT at the exact same (date, time_slot)
    for d in [1, 2]:
        for grp in set(grp for grp, _ in group_day_games):
            pair = group_day_games.get((grp, d), [])
            if len(pair) < 2:
                continue
            g1, g2 = pair

            # |date_num(g1) - date_num(g2)| <= 1
            lhs1 = gp.quicksum(node_daynum[v] * y[g1, v] for v in game_slots[g1])
            lhs2 = gp.quicksum(node_daynum[v] * y[g2, v] for v in game_slots[g2])
            model.addConstr(lhs1 - lhs2 <= 1, name=f"prox_{grp}_d{d}_a")
            model.addConstr(lhs2 - lhs1 <= 1, name=f"prox_{grp}_d{d}_b")

            # Not same (date, time_slot) — even across different stadiums
            for (date, slot), vs in date_slot_nodes.items():
                vs1 = [v for v in vs if v in set(game_slots[g1])]
                vs2 = [v for v in vs if v in set(game_slots[g2])]
                if not vs1 or not vs2:
                    continue
                model.addConstr(
                    gp.quicksum(y[g1, v] for v in vs1)
                    + gp.quicksum(y[g2, v] for v in vs2) <= 1,
                    name=f"notsameslot_{grp}_d{d}_{date}_{slot}"
                )

    # Objective: placeholder — update with travel / cost data
    model.setObjective(gp.LinExpr(), GRB.MINIMIZE)

    model.optimize()

    if model.Status == GRB.OPTIMAL:
        print(f"Status  : Optimal\n")

        # Collect results
        results = []
        for g, (t1, t2, d, grp) in enumerate(games):
            for v in game_slots[g]:
                if y[g, v].X > 0.5:
                    date, stadium, slot = nodes[v]
                    results.append((grp, d, t1, t2, date, stadium, slot))
                    break

        results.sort(key=lambda r: (r[0], r[1]))   # sort by group, then day

        print(f"{'Group':<6} {'Game':<40} {'Day':>3}  {'Date':<12} {'Stadium':<30} {'Slot'}")
        print("-" * 105)
        prev_grp = None
        for grp, d, t1, t2, date, stadium, slot in results:
            if prev_grp and grp != prev_grp:
                print()
            print(f"{grp:<6} {t1} vs {t2:<30} {d:>3}  {date:<12} {stadium:<30} {slot}")
            prev_grp = grp
    else:
        print(f"Model status: {model.Status}")

    return model


if __name__ == "__main__":
    teams_df, games_df, stadiums_df = load_data()
    build_and_solve(teams_df, games_df, stadiums_df)
