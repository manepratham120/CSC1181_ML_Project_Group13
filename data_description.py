"""
IPT player performance by phase (Opening / Mid / Death)
Run:  python ipt_phase_analysis.py ipt_json.zip
Creates:  ball_by_ball.csv, batter_by_phase.csv, bowler_by_phase.csv
"""
import json, sys, zipfile
import pandas as pd

ZIP_PATH = sys.argv[1] if len(sys.argv) > 1 else "ipt_json.zip"

# ---------- STEP 1: read every match into one ball-by-ball table ----------
rows = []
with zipfile.ZipFile(ZIP_PATH) as z:
    for name in z.namelist():
        if not name.endswith(".json"):
            continue
        d = json.loads(z.read(name))
        info = d["info"]
        match_id = name.split("/")[-1][:-5]
        for inn_no, inn in enumerate(d.get("innings", []), 1):
            if inn.get("super_over"):          # skip super overs
                continue
            for ov in inn["overs"]:
                for dl in ov["deliveries"]:
                    extras = dl.get("extras", {})
                    wk = dl.get("wickets", [])
                    rows.append({
                        "match_id": match_id,
                        "date": info["dates"][0],
                        "season": info.get("season"),
                        "venue": info.get("venue"),
                        "innings": inn_no,
                        "batting_team": inn["team"],
                        "over": ov["over"],                 # 0-19
                        "batter": dl["batter"],
                        "bowler": dl["bowler"],
                        "runs_batter": dl["runs"]["batter"],
                        "runs_total": dl["runs"]["total"],
                        "extras_type": next(iter(extras), None),
                        "is_wicket": len(wk) > 0,
                        "dismissal_kind": wk[0]["kind"] if wk else None,
                        "player_out": wk[0]["player_out"] if wk else None,
                    })
df = pd.DataFrame(rows)

# ---------- STEP 2: derived columns ----------
df["phase"] = pd.cut(df["over"], [-1, 5, 14, 19],
                     labels=["Opening", "Mid", "Death"]).astype(str)
df["is_legal_ball"] = ~df["extras_type"].isin(["wides", "noballs"])
# runs charged to bowler = all runs except byes / leg-byes
df["runs_conceded"] = df["runs_total"].where(
    ~df["extras_type"].isin(["byes", "legbyes"]), df["runs_batter"])
df["batter_out"] = df["player_out"] == df["batter"]
df["bowler_wicket"] = df["is_wicket"] & ~df["dismissal_kind"].isin(
    ["run out", "retired hurt", "retired out", "obstructing the field"])
df.to_csv("ball_by_ball.csv", index=False)
print("Deliveries:", len(df), "| Matches:", df.match_id.nunique())

# ---------- STEP 3: batter-by-phase table ----------
legal = df[df["is_legal_ball"]]
bat = legal.groupby(["batter", "phase"]).agg(
    balls_faced=("runs_batter", "size"),
    runs=("runs_batter", "sum"),
    dismissals=("batter_out", "sum"),
    fours=("runs_batter", lambda x: (x == 4).sum()),
    sixes=("runs_batter", lambda x: (x == 6).sum()),
    dots=("runs_batter", lambda x: (x == 0).sum()),
    innings=("match_id", "nunique"),
).reset_index()
bat["strike_rate"] = bat.runs / bat.balls_faced * 100
bat["batting_average"] = bat.runs / bat.dismissals.replace(0, pd.NA)
bat["boundary_pct"] = (bat.fours + bat.sixes) / bat.balls_faced * 100
bat["dot_ball_pct"] = bat.dots / bat.balls_faced * 100
phase_sr = legal.groupby("phase").apply(
    lambda g: g.runs_batter.sum() / len(g) * 100, include_groups=False)
bat["runs_above_average"] = bat.runs - bat.balls_faced * bat.phase.map(phase_sr) / 100
bat.round(2).to_csv("batter_by_phase.csv", index=False)

# ---------- STEP 4: bowler-by-phase table ----------
bowl = df.groupby(["bowler", "phase"]).agg(
    balls_bowled=("is_legal_ball", "sum"),
    runs_conceded=("runs_conceded", "sum"),
    wickets=("bowler_wicket", "sum"),
    wides_noballs=("is_legal_ball", lambda x: (~x).sum()),
    matches=("match_id", "nunique"),
).reset_index()
dots = df[df.is_legal_ball & (df.runs_conceded == 0)].groupby(
    ["bowler", "phase"]).size().rename("dots").reset_index()
bowl = bowl.merge(dots, how="left").fillna({"dots": 0})
bowl["economy"] = bowl.runs_conceded / bowl.balls_bowled * 6
bowl["bowling_strike_rate"] = bowl.balls_bowled / bowl.wickets.replace(0, pd.NA)
bowl["dot_ball_pct"] = bowl.dots / bowl.balls_bowled * 100
phase_rpb = df.groupby("phase").apply(
    lambda g: g.runs_conceded.sum() / g.is_legal_ball.sum(), include_groups=False)
bowl["runs_saved"] = bowl.balls_bowled * bowl.phase.map(phase_rpb) - bowl.runs_conceded
bowl.round(2).to_csv("bowler_by_phase.csv", index=False)

# ---------- STEP 5: leaderboards (minimum balls so small samples don't mislead) ----------
MIN_BAT = {"Opening": 150, "Mid": 150, "Death": 60}
MIN_BOWL = {"Opening": 120, "Mid": 200, "Death": 80}
for ph in ["Opening", "Mid", "Death"]:
    print(f"\n=== {ph}: top batters (min {MIN_BAT[ph]} balls) ===")
    print(bat[(bat.phase == ph) & (bat.balls_faced >= MIN_BAT[ph])]
          .nlargest(5, "runs_above_average")
          [["batter", "balls_faced", "runs", "strike_rate", "runs_above_average"]]
          .round(1).to_string(index=False))
    print(f"\n=== {ph}: top bowlers (min {MIN_BOWL[ph]} balls) ===")
    print(bowl[(bowl.phase == ph) & (bowl.balls_bowled >= MIN_BOWL[ph])]
          .nlargest(5, "runs_saved")
          [["bowler", "balls_bowled", "wickets", "economy", "runs_saved"]]
          .round(1).to_string(index=False))