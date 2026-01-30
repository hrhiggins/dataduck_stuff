import numpy as np
import pandas as pd
import time
from wakepy import keep
import tensorflow as tf
import json
import os

from train_model import (
    import_data_from_file,
    convert_to_time_series,
    preprocess_data,
    codes_dict,
    NUM_CODES,
    get_team_id
)

def add_team_context(df, teams_present, num_teams, num_codes):
    block_size = num_teams + num_codes

    active_teams = []
    goal_teams = []

    for vec, goal_flag in zip(df["features"], df["goal_event"]):
        active = None
        scorer = None

        for i, team in enumerate(teams_present):
            start = i * block_size
            team_onehot = vec[start : start + num_teams]
            event_vec = vec[start + num_teams : start + block_size]

            # If this team has any event at this timestep
            if event_vec.sum() > 0:
                active = team

            # If this team scored
            goal_index = codes_dict["goal"] - 1
            if goal_flag == 1 and event_vec[goal_index] == 1:
                scorer = team

        active_teams.append(active)
        goal_teams.append(scorer)

    df["active_team"] = active_teams
    df["goal_team"] = goal_teams
    return df



def assign_defending_team(df, teams_present):
    df = df.copy()
    teamA, teamB = teams_present

    def get_defender(active):
        if active == teamA:
            return teamB
        if active == teamB:
            return teamA
        return None

    df["defending_team"] = df["active_team"].apply(get_defender)
    return df

def run_sliding_inference(model, df, window_seconds, step_seconds):
    seq_len = int(window_seconds / step_seconds)
    feature_dim = len(df["features"].iloc[0])

    results = []

    for idx in range(len(df)):
        if idx < seq_len:
            continue

        window = df["features"].iloc[idx-seq_len:idx].tolist()
        X = np.array(window, dtype=np.float32).reshape(1, seq_len, feature_dim)

        goal_prob, xg = model.predict(X, verbose=0)

        row = df.iloc[idx].copy()
        row["pred_goal_prob"] = float(goal_prob[0][0])
        row["pred_xg"] = float(xg[0][0])
        results.append(row)

    return pd.DataFrame(results)

def compute_attacking_profile(df, team):
    df_team = df[df.active_team == team]

    return {
        "avg_danger": df_team.pred_goal_prob.mean(),
        "avg_xg": df_team.pred_xg.mean(),
        "danger_after_scoring": df_team[df_team.goal_team == team].pred_goal_prob.mean(),
        "num_samples": len(df_team)
    }


def compute_defensive_profile(df, team):
    df_team = df[df.defending_team == team]

    return {
        "avg_danger_conceded": df_team.pred_goal_prob.mean(),
        "avg_xg_conceded": df_team.pred_xg.mean(),
        "danger_conceded_after_opponent_goal":
            df_team[df_team.goal_team.notna()].pred_goal_prob.mean(),
        "num_samples": len(df_team)
    }


def compute_profiles_for_all_teams(df, all_teams):
    attacking = {}
    defensive = {}

    for team in all_teams:
        attacking[team] = compute_attacking_profile(df, team)
        defensive[team] = compute_defensive_profile(df, team)

    return attacking, defensive


def main():
    samples = 1
    list_of_files = import_data_from_file()

    model = tf.keras.models.load_model("temp/optuna/temp/trial_saves/best_model.keras")

    all_predictions = []
    all_teams = set()

    print("Processing all games...")

    for game_id, file in enumerate(list_of_files):
        print(f"Loading game {game_id}: {file}")

        file_df = pd.read_xml(file, xpath=".//instance")
        events_df = preprocess_data(file_df)

        # teams_present must come from the event-level dataframe
        teams_present = sorted(events_df["team"].unique())
        if len(teams_present) != 2:
            print(f"Warning: game {game_id} has {len(teams_present)} teams, expected 2")

        all_teams.update(teams_present)

        # now convert to time series
        df = convert_to_time_series(events_df, samples)

        num_teams = len(teams_present)
        num_codes = NUM_CODES

        df = add_team_context(df, teams_present, num_teams, num_codes)
        df = assign_defending_team(df, teams_present)

        pred_df = run_sliding_inference(
            model=model,
            df=df,
            window_seconds=40,
            step_seconds=1.0
        )

        pred_df["game_id"] = game_id
        all_predictions.append(pred_df)

    print("Combining all predictions...")
    df_all = pd.concat(all_predictions, ignore_index=True)

    print("Computing profiles...")
    attacking_profiles, defensive_profiles = compute_profiles_for_all_teams(
        df_all, sorted(all_teams)
    )

    os.makedirs("profiles", exist_ok=True)

    with open("profiles/attacking_profiles.json", "w") as f:
        json.dump(attacking_profiles, f, indent=4)

    with open("profiles/defensive_profiles.json", "w") as f:
        json.dump(defensive_profiles, f, indent=4)

    print("Profiles saved in profiles/ directory")
    print("Done.")

if __name__ == "__main__":
   with keep.running():
       start = time.time()
       main()
       end = time.time()
       print(f"The program took {end - start} seconds to run")
