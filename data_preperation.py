
import pandas as pd

df = pd.read_csv("ball_by_ball.csv")

overall = df.copy()
overall["phase"] = "Overall"
d = pd.concat([df, overall], ignore_index=True)

# ---------- 2. Batting columns ----------
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

bat["bat_runs_above_avg"] = (
    bat.bat_runs - bat.bat_balls * bat.phase.map(phase_sr)
)


# ---------- 3. Bowling columns ----------
bowl = d.groupby(["bowler", "phase"]).agg(
    bowl_balls=("is_legal_ball", "sum"),
    bowl_runs=("runs_conceded", "sum"),
    bowl_wkts=("bowler_wicket", "sum"),
).reset_index().rename(columns={"bowler": "player"})

bowl["bowl_overs"] = (
    (bowl.bowl_balls // 6).astype(str)
    + "."
    + (bowl.bowl_balls % 6).astype(str)
)

bowl["bowl_avg"] = (
    bowl.bowl_runs / bowl.bowl_wkts.where(bowl.bowl_wkts > 0)
)

bowl["bowl_economy"] = bowl.bowl_runs / bowl.bowl_balls * 6


# ---------- 4. 4-wicket and 5-wicket hauls ----------
per_match = d.groupby(
    ["bowler", "phase", "match_id"]
).bowler_wicket.sum().reset_index()

hauls = per_match.groupby(["bowler", "phase"]).agg(
    bowl_4fers=("bowler_wicket", lambda x: (x == 4).sum()),
    bowl_5fers=("bowler_wicket", lambda x: (x >= 5).sum()),
).reset_index().rename(columns={"bowler": "player"})

bowl = bowl.merge(hauls, on=["player", "phase"], how="left")


# ---------- 5. Runs saved ----------
phase_rpb = d.groupby("phase").apply(
    lambda g: g.runs_conceded.sum() / g.is_legal_ball.sum(),
    include_groups=False
)

bowl["bowl_runs_saved"] = (
    bowl.bowl_balls * bowl.phase.map(phase_rpb) - bowl.bowl_runs
)


# ---------- 6. Matches played ----------
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


# ---------- 7. Merge everything ----------
out = matches.merge(
    bat, on=["player", "phase"], how="left"
).merge(
    bowl, on=["player", "phase"], how="left"
)


# ---------- 8. Fill missing count columns ----------
count_cols = [
    "bat_innings", "bat_balls", "bat_runs", "bat_outs",
    "bat_4s", "bat_6s", "bat_runs_above_avg",
    "bowl_balls", "bowl_runs", "bowl_wkts",
    "bowl_4fers", "bowl_5fers", "bowl_runs_saved"
]

out[count_cols] = out[count_cols].fillna(0)

out["bowl_overs"] = out["bowl_overs"].fillna("0.0")


# ---------- 9. Add is_batting and is_bowling ----------



# ---------- 10. Arrange final columns ----------
order = [
    "player", "phase", "matches",
    "bat_innings", "bat_balls", "bat_runs", "bat_outs",
    "bat_avg", "bat_sr", "bat_4s", "bat_6s",
    "bat_runs_above_avg",
    "bowl_overs", "bowl_balls", "bowl_wkts",
    "bowl_avg", "bowl_runs", "bowl_economy",
    "bowl_4fers", "bowl_5fers", "bowl_runs_saved"
]

out = out[order]

out["phase"] = pd.Categorical(
    out["phase"],
    ["Opening", "Mid", "Death", "Overall"]
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
