import os
import sys
import time
import pandas as pd
from wakepy import keep
import numpy as np


codes_dict = {"ball_loss_1" : 1, "ball_loss_2" : 2, "ball_loss_3" : 3, "ball_loss_4" : 4, "ball_win_0" : 5,
              "ball_win_1" : 6, "ball_win_2" : 7, "ball_win_3" : 8, "ball_win_4" : 9, "ball_win_5" : 10,
              "ball_restart_0" : 11, "ball_restart_1" : 12, "ball_restart_2" : 13, "ball_restart_3" : 14,
              "ball_restart_4" : 15, "ball_restart_5" : 16, "ball_possession" : 17, "circle_penetration" : 18,
              "circle_penetration_against" : 19, "goal" : 20, "goal_attempt" : 21, "goal_attempt_against" : 22,
              "penalty_corner" : 23, "penalty_corner_against" : 24, "penalty_stroke" : 25,
              "cause_of_penalty_corner" : 26, "cause_of_penalty_corner_against" : 27, "cause_of_penalty_stroke" : 28,
              "green_card" : 29, "yellow_card" : 30, "red_card" : 31, "umpiring" : 32, "start" : 33, "end" : 34,
              "pause" : 35, "press" : 36, "build_up" : 37, "high_ball" : 38, "long_corner" : 39, "shootout" : 40,
              "man_up" : 41, "man_down" : 42, "no_keeper" : 43}

NUM_CODES = len(codes_dict)


teams_dict = {}
next_team_id = 1

def get_team_id(team_name):
    """Assign a unique ID to each team dynamically."""
    global next_team_id

    if not isinstance(team_name, str):
        return None

    team_name = team_name.lower().strip()

    # Ensure team name is NOT an event code
    if team_name in codes_dict:
        return None

    if team_name not in teams_dict:
        teams_dict[team_name] = next_team_id
        next_team_id += 1

    return teams_dict[team_name]


def codes_to_vector(code_list):
    vector = np.zeros(NUM_CODES, dtype=np.float32)
    for code in code_list:
        if code is None:
            continue
        vector[code - 1] = 1.0
    return vector


def import_data_from_file():
    if len(sys.argv) < 2:
        raise ImportError("Not enough arguments passed")

    all_files = []
    for i in range(1, len(sys.argv)):
        directory = sys.argv[i]

        if not os.path.isdir(directory):
            raise NotADirectoryError(f"{directory} is not a valid directory")

        for f in os.listdir(directory):
            full_path = os.path.join(directory, f)
            if os.path.isfile(full_path):
                all_files.append(full_path)

    return all_files


def preprocess_data(df):
    df = df.copy()

    df[["team", "code"]] = df["code"].astype(str).str.split(
        pat=" ", n=1, expand=True
    )

    # Assign dynamic team IDs
    df["team"] = df["team"].apply(get_team_id)

    # Drop rows where team was actually an event code
    df = df[df["team"].notna()]
    df["team"] = df["team"].astype(int)

    df = df[df["code"].notna()]
    df["code"] = df["code"].astype(str).str.strip().str.lower()

    df["code"] = df["code"].map(codes_dict)
    df = df[df["code"].notna()]
    df["code"] = df["code"].astype(int)

    start_time = df["start"].min()
    df["start"] -= start_time
    df["end"] -= start_time

    df["start"] = df["start"].round(1)
    df["end"] = df["end"].round(1)

    return df


def convert_to_time_series(df, sample_rate):
    df = df.copy()

    start_time = df["start"].min()
    end_time = df["end"].max()

    n_steps = int((end_time - start_time) * sample_rate) + 1
    time_values = [step / sample_rate for step in range(n_steps)]

    teams_present = sorted(df["team"].unique())
    num_teams = len(teams_dict)

    team_lists = {}
    for team in teams_present:
        team_lists[team] = [[] for _ in range(n_steps)]

    for _, row in df.iterrows():
        code = int(row["code"])
        team = int(row["team"])
        start_idx = int(row["start"] * sample_rate)
        end_idx = int(row["end"] * sample_rate)
        for idx in range(start_idx, end_idx + 1):
            team_lists[team][idx].append(code)

    data = {}
    data["time"] = time_values

    for team in teams_present:
        column_name = f"team_{team}"
        vectors = []
        for lst in team_lists[team]:
            team_onehot = np.zeros(num_teams, dtype=np.float32)
            team_onehot[team - 1] = 1.0
            event_vector = codes_to_vector(lst)
            vectors.append(np.concatenate([team_onehot, event_vector]))
        data[column_name] = vectors

    feature_vectors = []
    for i in range(n_steps):
        combined = []
        for team in teams_present:
            combined.append(data[f"team_{team}"][i])
        feature_vectors.append(np.concatenate(combined))

    # Extract goal event
    goal_code = codes_dict["goal"]
    goal_offset = goal_code - 1
    block_size = num_teams + NUM_CODES

    goal_indices = []
    for block_i in range(len(teams_present)):
        base = block_i * block_size
        goal_idx = base + num_teams + goal_offset
        goal_indices.append(goal_idx)

    goal_events = []
    cleaned_features = []

    for vec in feature_vectors:
        goal_flag = 0
        for idx in goal_indices:
            if idx < len(vec) and vec[idx] == 1.0:
                goal_flag = 1
                break

        goal_events.append(goal_flag)

        valid_goal_indices = [idx for idx in goal_indices if idx < len(vec)]
        vec_no_goal = np.delete(vec, valid_goal_indices)
        cleaned_features.append(vec_no_goal)

    data["features"] = cleaned_features
    data["goal_event"] = goal_events

    time_series = pd.DataFrame(data)
    df = time_series[["time", "features", "goal_event"]].copy()

    return df


def main():
    samples = 2
    list_of_files = import_data_from_file()

    game_dfs = []

    for game_id, file in enumerate(list_of_files):
        file_df = pd.read_xml(file, xpath=".//instance")
        df = preprocess_data(file_df)
        df = convert_to_time_series(df, samples)

        df["game_id"] = game_id
        game_dfs.append(df)

    all_games_df = pd.concat(game_dfs, ignore_index=True)
    all_games_df.to_csv("time_series.csv", index=False)





if __name__ == "__main__":
    with keep.running():
        start = time.time()
        main()
        end = time.time()
        print(f"The program took {end - start} seconds to run")

