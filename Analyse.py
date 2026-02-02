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
    NUM_CODES
)
from windowing import WarmupCosine
from run_study import UncertaintyWeights


def add_team_context(df, teams_present, num_teams, num_codes):
    block_size = num_teams + num_codes

    active_teams = []
    goal_teams = []

    for vec, goal_flag in zip(df["features"], df["goal_event"]):
        active = None
        scorer = None

        for i, team in enumerate(teams_present):
            start = i * block_size
            event_vec = vec[start + num_teams : start + block_size]

            # Any event for this team at this timestep
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
    if len(teams_present) != 2:
        df["defending_team"] = None
        return df

    teamA, teamB = teams_present

    def get_defender(active):
        if active == teamA:
            return teamB
        if active == teamB:
            return teamA
        return None

    df["defending_team"] = df["active_team"].apply(get_defender)
    return df


def run_sliding_inference(model, df, window_seconds, step_seconds, global_len=80):
    seq_len = int(window_seconds / step_seconds)
    feature_dim = len(df["features"].iloc[0])

    # Build global sequence once per game (downsampled)
    T = len(df)
    idx = np.linspace(0, T - 1, global_len).astype(int)
    global_features = np.array([df["features"].iloc[i] for i in idx], dtype=np.float32)
    X_global = global_features.reshape(1, global_len, feature_dim)

    results = []

    for t in range(len(df)):
        if t < seq_len:
            continue

        window = df["features"].iloc[t - seq_len : t].tolist()
        X_local = np.array(window, dtype=np.float32).reshape(1, seq_len, feature_dim)

        preds = model.predict(
            {
                "local_input": X_local,
                "global_input": X_global,
            },
            verbose=0,
        )

        goal_prob = preds[0][0][0]
        xg = preds[1][0][0]

        row = df.iloc[t].copy()
        row["pred_goal_prob"] = float(goal_prob)
        row["pred_xg"] = float(xg)
        results.append(row)

    return pd.DataFrame(results)


def compute_attacking_profile(df, team):
    df_team = df[df.active_team == team]
    if len(df_team) == 0:
        return {
            "avg_danger": None,
            "avg_xg": None,
            "danger_after_scoring": None,
            "num_samples": 0,
        }

    return {
        "avg_danger": float(df_team.pred_goal_prob.mean()),
        "avg_xg": float(df_team.pred_xg.mean()),
        "danger_after_scoring": float(
            df_team[df_team.goal_team == team].pred_goal_prob.mean()
        )
        if (df_team.goal_team == team).any()
        else None,
        "num_samples": int(len(df_team)),
    }


def compute_defensive_profile(df, team):
    df_team = df[df.defending_team == team]
    if len(df_team) == 0:
        return {
            "avg_danger_conceded": None,
            "avg_xg_conceded": None,
            "danger_conceded_after_opponent_goal": None,
            "num_samples": 0,
        }

    return {
        "avg_danger_conceded": float(df_team.pred_goal_prob.mean()),
        "avg_xg_conceded": float(df_team.pred_xg.mean()),
        "danger_conceded_after_opponent_goal": float(
            df_team[df_team.goal_team.notna()].pred_goal_prob.mean()
        )
        if df_team.goal_team.notna().any()
        else None,
        "num_samples": int(len(df_team)),
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

    # Load the FULL model (.keras) with custom objects
    model = tf.keras.models.load_model(
        "temp/optuna/best_model.keras",
        custom_objects={
            "WarmupCosine": WarmupCosine,
            "UncertaintyWeights": UncertaintyWeights,
        },
    )

    all_predictions = []
    all_teams = set()

    print("Processing all games...")

    for game_id, file in enumerate(list_of_files):
        print(f"Loading game {game_id}: {file}")

        file_df = pd.read_xml(file, xpath=".//instance")
        events_df = preprocess_data(file_df)

        teams_present = sorted(events_df["team"].unique())
        all_teams.update(teams_present)

        df = convert_to_time_series(events_df, samples)

        num_teams = len(teams_present)
        num_codes = NUM_CODES

        df = add_team_context(df, teams_present, num_teams, num_codes)
        df = assign_defending_team(df, teams_present)

        pred_df = run_sliding_inference(
            model=model,
            df=df,
            window_seconds=40,   # must match training
            step_seconds=1.0,
            global_len=80,       # must match training
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