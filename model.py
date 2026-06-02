import pandas as pd
import gurobipy as gp
from gurobipy import GRB


def load_data(path="data_fifa.xlsx",
              camp_csv="flight_data_analysis/flight_distances_camps_stadiums.csv",
              weather_csv="historic_weather_data/stadium_hourly_filtered.csv"):
    teams_df    = pd.read_excel(path, sheet_name="teams")
    games_df    = pd.read_excel(path, sheet_name="games")
    stadiums_df = pd.read_excel(path, sheet_name="stadiums")
    dist_df     = pd.read_excel(path, sheet_name="stadium_distances", index_col=0)

    camp_raw     = pd.read_csv(camp_csv)[["camp_team", "stadium_name", "stad_camp_direct_distance_km"]]
    camp_dist_df = (camp_raw
                    .groupby(["camp_team", "stadium_name"])["stad_camp_direct_distance_km"]
                    .min()
                    .unstack(level="stadium_name"))

    weather_df = pd.read_csv(weather_csv, low_memory=False)

    return teams_df, games_df, stadiums_df, dist_df, camp_dist_df, weather_df


def build_intermediate_nodes(weather_df):
    """
    Nodes are (date, stadium, hour_utc) rows from the weather CSV.
    Returns:
        nodes      : list of (date_str, stadium, hour_utc) tuples
        node_temp  : dict {node_idx: apparent_temperature}
    """
    weather_df = weather_df.copy()
    weather_df["date"] = "2026-06-" + weather_df["day"].astype(str).str.zfill(2)

    nodes     = []
    node_temp = {}

    for _, row in weather_df.iterrows():
        v = len(nodes)
        nodes.append((row["date"], row["stadium"], int(row["hour_utc"])))
        node_temp[v] = row["apparent_temperature"]

    return nodes, node_temp


def build_and_solve(teams_df, games_df, stadiums_df, dist_df, camp_dist_df, weather_df):
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
    teams              = teams_df["team"].tolist()
    nodes, node_temp   = build_intermediate_nodes(weather_df)
    stad_elev          = stadiums_df.set_index("stadium")["elevation"].to_dict()
    camp_elev          = teams_df.set_index("team")["camp_elevation"].to_dict()
    camp_temp_avg      = teams_df.set_index("team")["camp_avg_temperature"].to_dict()

    dates       = pd.to_datetime(games_df["date"])
    first_date  = dates.min()
    last_date   = dates.max()
    src_cutoff  = last_date  - pd.Timedelta(days=8)   # arc group 1 upper bound
    sink_cutoff = first_date + pd.Timedelta(days=8)   # arc group 3 lower bound

    n        = len(teams)
    m        = len(nodes)
    G        = len(games_df)
    stadiums = stadiums_df["stadium"].tolist()

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

    # Host constraint: games involving a host team must be played in that country
    HOSTS = {"Mexico": "MEX", "Canada": "CAN", "USA": "USA"}
    stad_country = stadiums_df.set_index("stadium")["country"].to_dict()
    node_country = {v: stad_country[nodes[v][1]] for v in range(m)}

    for g, (t1, t2, d, grp) in enumerate(games):
        required_countries = {HOSTS[t] for t in (t1, t2) if t in HOSTS}
        if required_countries:
            game_slots[g] = [v for v in game_slots[g]
                             if node_country[v] in required_countries]

    # Reverse index: slot → list of games that can go there
    slot_games = {v: [] for v in range(m)}
    for g, slots in game_slots.items():
        for v in slots:
            slot_games[v].append(g)

    print(f"Teams             : {n}")
    print(f"Games             : {G}")
    print(f"Intermediate nodes: {m}  (from weather data)")
    print(f"src_cutoff        : {src_cutoff.date()}  (last_date - 8,  arc group 1)")
    print(f"sink_cutoff       : {sink_cutoff.date()}  (first_date + 8, arc group 3)\n")

    # --- Model ---------------------------------------------------------------
    model = gp.Model("fifa_flow")
    model.Params.LogToConsole = 0
    model.Params.TimeLimit    = 300

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

    # Stadium cooldown — a stadium cannot host two games within 3 days of each other
    # For each (stadium, date_1, date_2) where 0 < date_2 - date_1 < 3:
    #   games_at_s_d1 + games_at_s_d2 <= 1
    dates_sorted = sorted({nodes[v][0] for v in range(m)})
    date_to_ts   = {d: pd.to_datetime(d) for d in dates_sorted}

    for s in stadiums_df["stadium"]:
        for i, d1 in enumerate(dates_sorted):
            for d2 in dates_sorted[i + 1:]:
                gap = (date_to_ts[d2] - date_to_ts[d1]).days
                if gap >= 3:
                    break   # dates are sorted, no need to check further
                vs1 = [v for v in date_stadium_slots.get((d1, s), []) if slot_games[v]]
                vs2 = [v for v in date_stadium_slots.get((d2, s), []) if slot_games[v]]
                if not vs1 or not vs2:
                    continue
                model.addConstr(
                    gp.quicksum(y[g, v] for v in vs1 for g in slot_games[v])
                    + gp.quicksum(y[g, v] for v in vs2 for g in slot_games[v]) <= 1,
                    name=f"cooldown_{s}_{d1}_{d2}"
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

    # --- Flow formulation with arc costs ---------------------------------------
    stadiums       = dist_df.index.tolist()
    node_stad      = {v: nodes[v][1] for v in range(m)}
    game_stad_slots = {
        (g, s): [v for v in game_slots[g] if node_stad[v] == s]
        for g in range(G) for s in stadiums
    }

    # Consecutive game transitions per team: (team, matchday, g1, g2)
    team_consec = [
        (team, d, team_day_to_game[(team, d)], team_day_to_game[(team, d + 1)])
        for team in teams for d in [1, 2]
        if (team, d) in team_day_to_game and (team, d + 1) in team_day_to_game
    ]

    # Stadium diversity — each team must play in at least 2 different stadiums
    for team in teams:
        for s in stadiums:
            team_games = [team_day_to_game[(team, d)] for d in [1, 2, 3]
                          if (team, d) in team_day_to_game]
            slots_at_s = [game_stad_slots[g, s] for g in team_games]
            if not any(slots_at_s):
                continue
            model.addConstr(
                gp.quicksum(y[g, v] for g, vs in zip(team_games, slots_at_s) for v in vs) <= 2,
                name=f"diversity_{team}_{s}"
            )

    # Flow variable f[team, d, s1, s2]: flow on arc s1→s2 for team at transition d
    # Arc cost = dist(s1, s2).  LP-integral since supplies/demands are 0/1.
    f = model.addVars(
        [(team, d, s1, s2)
         for team, d, g1, g2 in team_consec
         for s1 in stadiums if game_stad_slots[g1, s1]
         for s2 in stadiums if game_stad_slots[g2, s2]],
        lb=0, ub=1, name="f"
    )

    # Flow conservation — outflow from s1 = whether game g1 is at s1
    for team, d, g1, g2 in team_consec:
        for s1 in stadiums:
            if not game_stad_slots[g1, s1]:
                continue
            model.addConstr(
                gp.quicksum(f[team, d, s1, s2]
                            for s2 in stadiums if (team, d, s1, s2) in f)
                == gp.quicksum(y[g1, v] for v in game_stad_slots[g1, s1]),
                name=f"fout_{team}_{d}_{s1}"
            )

    # Flow conservation — inflow to s2 = whether game g2 is at s2
    for team, d, g1, g2 in team_consec:
        for s2 in stadiums:
            if not game_stad_slots[g2, s2]:
                continue
            model.addConstr(
                gp.quicksum(f[team, d, s1, s2]
                            for s1 in stadiums if (team, d, s1, s2) in f)
                == gp.quicksum(y[g2, v] for v in game_stad_slots[g2, s2]),
                name=f"fin_{team}_{d}_{s2}"
            )

    # Source arc costs: camp → day-1 game stadium
    # f_src[team, s] = 1 if team's day-1 game is at stadium s
    valid_src_arcs = [
        (team, s)
        for team in teams
        for s in stadiums
        if team_day_to_game.get((team, 1)) is not None
        and game_stad_slots[team_day_to_game[(team, 1)], s]
        and team in camp_dist_df.index
        and s in camp_dist_df.columns
        and pd.notna(camp_dist_df.loc[team, s])
    ]

    f_src = model.addVars(valid_src_arcs, lb=0, ub=1, name="f_src")

    for team, s in valid_src_arcs:
        g1 = team_day_to_game[(team, 1)]
        model.addConstr(
            f_src[team, s]
            == gp.quicksum(y[g1, v] for v in game_stad_slots[g1, s]),
            name=f"fsrc_{team}_{s}"
        )

    # Objective: camp → day-1 + day-1 → day-2 + day-2 → day-3
    def via_camp_cost(team, s1, s2):
        """Distance s1 → camp → s2 for a given team. Falls back to direct if missing."""
        if (team in camp_dist_df.index
                and s1 in camp_dist_df.columns
                and s2 in camp_dist_df.columns
                and pd.notna(camp_dist_df.loc[team, s1])
                and pd.notna(camp_dist_df.loc[team, s2])):
            return camp_dist_df.loc[team, s1] + camp_dist_df.loc[team, s2]
        return dist_df.loc[s1, s2]   # fallback

    # --- Tolerance between phases (allow obj to degrade by this % in next phase)
    phase_tol = 0.01   # 1% — adjust as needed

    # Obj 1: direct stadium-to-stadium distance (no camp)
    obj_stadium = gp.quicksum(
        dist_df.loc[s1, s2] * f[team, d, s1, s2]
        for team, d, g1, g2 in team_consec
        for s1 in stadiums for s2 in stadiums
        if (team, d, s1, s2) in f
    )

    # Obj 2: full via-camp travel — camp→s1 (source) + s1→camp→s2 (inter-game)
    obj_camp = (
        gp.quicksum(
            camp_dist_df.loc[team, s] * f_src[team, s]
            for team, s in f_src
        )
        + gp.quicksum(
            via_camp_cost(team, s1, s2) * f[team, d, s1, s2]
            for team, d, g1, g2 in team_consec
            for s1 in stadiums for s2 in stadiums
            if (team, d, s1, s2) in f
        )
    )

    # Precompute temperature shock coefficients: |node_temp - avg_camp_temp| per (game, slot)
    obj_temp = gp.quicksum(
        (abs(node_temp[v] - camp_temp_avg.get(t1, 0))
         + abs(node_temp[v] - camp_temp_avg.get(t2, 0))) * y[g, v]
        for g, (t1, t2, d, grp) in enumerate(games)
        for v in game_slots[g]
    )

    # Elevation gain penalty: max(0, stad_elev - camp_elev) per team per game
    # Penalizes going from lower-altitude camp to higher-altitude stadium
    obj_elev = gp.quicksum(
        (max(0, stad_elev.get(nodes[v][1], 0) - camp_elev.get(t1, 0))
         + max(0, stad_elev.get(nodes[v][1], 0) - camp_elev.get(t2, 0))) * y[g, v]
        for g, (t1, t2, d, grp) in enumerate(games)
        for v in game_slots[g]
    )

    def phase_report(unit="km"):
        status_str = "Optimal" if model.Status == GRB.OPTIMAL else "Time limit"
        print(f"  Status : {status_str}")
        print(f"  Runtime: {model.Runtime:.1f}s")
        if model.SolCount > 0:
            print(f"  UB     : {model.ObjVal:,.2f} {unit}")
            print(f"  LB     : {model.ObjBound:,.2f} {unit}")
            print(f"  Gap    : {model.MIPGap * 100:.2f}%")
        else:
            print("  No solution found.")

    # --- Phase 1: stadium distances (priority 1 — highest) ------------------
    print("Phase 1 — Stadium-to-stadium distances:")
    model.setObjective(obj_stadium, GRB.MINIMIZE)
    model.optimize()
    phase_report("km")

    if model.SolCount == 0:
        print("No feasible solution found in phase 1.")
        return model

    model.addConstr(
        obj_stadium <= model.ObjVal * (1 + phase_tol), name="fix_obj1"
    )

    # --- Phase 2: camp distances (priority 2) --------------------------------
    print("\nPhase 2 — Camp-to-stadium distances:")
    model.setObjective(obj_camp, GRB.MINIMIZE)
    model.optimize()
    phase_report("km")

    if model.SolCount == 0:
        print("No feasible solution found in phase 2.")
        return model

    model.addConstr(
        obj_camp <= model.ObjVal * (1 + phase_tol), name="fix_obj2"
    )

    # --- Phase 3: temperature shock (priority 3) ----------------------------
    print("\nPhase 3 — Temperature shock (°C):")
    model.setObjective(obj_temp, GRB.MINIMIZE)
    model.optimize()
    phase_report("°C")

    if model.SolCount == 0:
        print("No feasible solution found in phase 3.")
        return model

    model.addConstr(
        obj_temp <= model.ObjVal * (1 + phase_tol), name="fix_obj3"
    )

    # --- Phase 4: elevation gain penalty (priority 4 — lowest) --------------
    print("\nPhase 4 — Elevation gain (m):")
    model.setObjective(obj_elev, GRB.MINIMIZE)
    model.optimize()
    phase_report("m")

    if model.SolCount == 0:
        print("No feasible solution found in phase 4.")
        return model

    print()

    # Collect and print results
    results = []
    for g, (t1, t2, d, grp) in enumerate(games):
        for v in game_slots[g]:
            if y[g, v].X > 0.5:
                date, stadium, slot = nodes[v]
                results.append((grp, d, t1, t2, date, stadium, slot))
                break

    results.sort(key=lambda r: (r[0], r[1]))

    print(f"{'Group':<6} {'Game':<40} {'Day':>3}  {'Date':<12} {'Stadium':<30} {'UTC Time'}")
    print("-" * 105)
    prev_grp = None
    for grp, d, t1, t2, date, stadium, slot in results:
        if prev_grp and grp != prev_grp:
            print()
        print(f"{grp:<6} {t1} vs {t2:<30} {d:>3}  {date:<12} {stadium:<30} {slot}")
        prev_grp = grp

    return model


if __name__ == "__main__":
    teams_df, games_df, stadiums_df, dist_df, camp_dist_df, weather_df = load_data()

    nodes, node_temp = build_intermediate_nodes(weather_df)
    stad_elev        = stadiums_df.set_index("stadium")["elevation"].to_dict()
    pd.DataFrame([
        {"node_id": v, "date": nodes[v][0], "stadium": nodes[v][1],
         "hour_utc": nodes[v][2], "apparent_temperature": node_temp[v],
         "elevation": stad_elev[nodes[v][1]]}
        for v in range(len(nodes))
    ]).to_csv("intermediate_nodes.csv", index=False)
    print(f"Exported {len(nodes)} nodes to intermediate_nodes.csv")

    build_and_solve(teams_df, games_df, stadiums_df, dist_df, camp_dist_df, weather_df)
