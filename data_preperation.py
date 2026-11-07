import pandas as pd

df = pd.read_csv("ball_by_ball.csv")

d = df.copy()

# ---------- 1. Batting columns ----------
legal = d[d["is_legal_ball"]]

bat = legal.groupby(["batter", "phase"]).agg(
    bat_innings=("match_id", "nunique"),
    bat_balls=("runs_batter", "size"),
    bat_runs=("runs_batter", "sum"),
    bat_outs=("batter_out", "sum"),
    bat_4s=("runs_batter", lambda x: (x == 4).sum()),
    bat_6s=("runs_batter", lambda x: (x == 6).sum()),
).reset_index().rename(columns={"batter": "player"})

bat["bat_avg"] = bat.bat_runs / bat.bat_outs.where(bat.bat_outs > 0)
bat["bat_sr"] = bat.bat_runs / bat.bat_balls * 100

# Runs above average
phase_sr = legal.groupby("phase").apply(
    lambda g: g.runs_batter.sum() / len(g),
    include_groups=False
)




# ---------- 2. Bowling columns ----------
bowl = d.groupby(["bowler", "phase"]).agg(
    bowl_balls=("is_legal_ball", "sum"),
    bowl_runs=("runs_conceded", "sum"),
    bowl_wkts=("bowler_wicket", "sum"),
).reset_index().rename(columns={"bowler": "player"})

bowl["bowl_avg"] = (
    bowl.bowl_runs / bowl.bowl_wkts.where(bowl.bowl_wkts > 0)
)

bowl["bowl_economy"] = bowl.bowl_runs / bowl.bowl_balls * 6


# ---------- 3. 4-wicket and 5-wicket hauls ----------
per_match = d.groupby(
    ["bowler", "phase", "match_id"]
).bowler_wicket.sum().reset_index()

hauls = per_match.groupby(["bowler", "phase"]).agg(
    bowl_4fers=("bowler_wicket", lambda x: (x == 4).sum()),
    bowl_5fers=("bowler_wicket", lambda x: (x >= 5).sum()),
).reset_index().rename(columns={"bowler": "player"})

bowl = bowl.merge(hauls, on=["player", "phase"], how="left")


# ---------- 4. Matches played ----------
a = d[["batter", "phase", "match_id"]].rename(
    columns={"batter": "player"}
)

b = d[["bowler", "phase", "match_id"]].rename(
    columns={"bowler": "player"}
)

matches = (
    pd.concat([a, b])
    .drop_duplicates()
    .groupby(["player", "phase"])
    .size()
    .rename("matches")
    .reset_index()
)


# ---------- 5. Merge everything ----------
out = matches.merge(
    bat, on=["player", "phase"], how="left"
).merge(
    bowl, on=["player", "phase"], how="left"
)


# ---------- 6. Fill missing count columns ----------
count_cols = [
    "bat_innings", "bat_balls", "bat_runs", "bat_outs",
    "bat_4s", "bat_6s", 
    "bowl_balls", "bowl_runs", "bowl_wkts",
    "bowl_4fers", "bowl_5fers"
]

out[count_cols] = out[count_cols].fillna(0)


# ---------- 7. Arrange final columns ----------
order = [
    "player", "phase", "matches",
    "bat_innings", "bat_balls", "bat_runs", "bat_outs",
    "bat_avg", "bat_sr", "bat_4s", "bat_6s",
    "bowl_balls", "bowl_wkts",
    "bowl_avg", "bowl_runs", "bowl_economy",
    "bowl_4fers", "bowl_5fers"
]

out = out[order]

out["phase"] = pd.Categorical(
    out["phase"],
    ["Opening", "Mid", "Death"]
)

out = out.sort_values(["player", "phase"])

out.round(2).to_csv("player_summary.csv", index=False)

print(
    "Done! player_summary.csv has",
    len(out),
    "rows and",
    out.player.nunique(),
    "players"
)