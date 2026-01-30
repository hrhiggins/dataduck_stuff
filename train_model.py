import os
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"

# Stop annoying Tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import optuna
from multiprocessing import Pool
import multiprocessing as mp
import pandas as pd
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
import time
import numpy as np
from wakepy import keep
from keras.models import load_model
import shutil
from run_study import objective
import tensorflow as tf

# If running on linux turn on:
#mp.set_start_method("spawn", force=True)
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy("mixed_float16")

import warnings
#warnings.filterwarnings("ignore", category=UserWarning)
#warnings.filterwarnings("ignore", category=FutureWarning)



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


def get_tf_device():
    gpus = tf.config.list_logical_devices("GPU")
    if gpus:
        return "/GPU:0"
    else:
        return "/CPU:0"



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
    num_teams = len(teams_present)
    team_index_map = {team: idx for idx, team in enumerate(teams_present)}

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
            team_onehot[team_index_map[team]] = 1.0
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
    # --- Extract key events (goal, penalty corner, penalty stroke, circle entry) ---

    # Event codes
    goal_code = codes_dict["goal"]
    pc_code = codes_dict["penalty_corner"]
    ps_code = codes_dict["penalty_stroke"]
    ce_code = codes_dict["circle_penetration"]

    # Offsets inside each team block
    goal_offset = goal_code - 1
    pc_offset = pc_code - 1
    ps_offset = ps_code - 1
    ce_offset = ce_code - 1

    block_size = num_teams + NUM_CODES

    # Compute indices for each team block
    def make_indices(offset):
        return [
            block_i * block_size + num_teams + offset
            for block_i in range(num_teams)
        ]

    goal_indices = make_indices(goal_offset)
    pc_indices = make_indices(pc_offset)
    ps_indices = make_indices(ps_offset)
    ce_indices = make_indices(ce_offset)

    # Output arrays
    goal_events = []
    pc_events = []
    ps_events = []
    ce_events = []
    cleaned_features = []

    for vec in feature_vectors:
        # Detect events
        goal_flag = any(idx < len(vec) and vec[idx] == 1.0 for idx in goal_indices)
        pc_flag = any(idx < len(vec) and vec[idx] == 1.0 for idx in pc_indices)
        ps_flag = any(idx < len(vec) and vec[idx] == 1.0 for idx in ps_indices)
        ce_flag = any(idx < len(vec) and vec[idx] == 1.0 for idx in ce_indices)

        goal_events.append(int(goal_flag))
        pc_events.append(int(pc_flag))
        ps_events.append(int(ps_flag))
        ce_events.append(int(ce_flag))

        # Remove goal indices from features (keep PC, PS, CE in features)
        valid_goal_indices = [idx for idx in goal_indices if idx < len(vec)]
        vec_no_goal = np.delete(vec, valid_goal_indices)
        cleaned_features.append(vec_no_goal)

    # Add to dataframe
    data["features"] = cleaned_features
    data["goal_event"] = goal_events
    data["penalty_corner_event"] = pc_events
    data["penalty_stroke_event"] = ps_events
    data["circle_entry_event"] = ce_events

    time_series = pd.DataFrame(data)
    df = time_series[
        ["time", "features", "goal_event",
         "penalty_corner_event", "penalty_stroke_event", "circle_entry_event"]
    ].copy()

    return df

# https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/004_distributed.html
# https://optuna.readthedocs.io/en/stable/faq.html#id2
def run_study(device, time_snapshot, number_of_trials, number_of_processes, df):
    os.makedirs("temp/optuna/journals", exist_ok=True)
    study = optuna.create_study(directions=["minimize"], study_name="expected_goals",
                                storage=JournalStorage(JournalFileBackend(file_path=f"temp/optuna/journals/journal{time_snapshot}.log")),
                                load_if_exists=True,
                                pruner=optuna.pruners.HyperbandPruner(min_resource=3, max_resource="auto",
                                                                      reduction_factor=3))

    with tf.device(device):
        study.optimize(lambda trial: objective(trial, df), n_trials=(number_of_trials//number_of_processes))
    return study

def new_temp_dirs():
    # Remove old temporary data
    shutil.rmtree("temp/optuna/temp", ignore_errors=True)
    # Create directories for new data
    os.makedirs("temp/optuna/temp", exist_ok=True)
    os.makedirs("temp/optuna/temp/trial_saves", exist_ok=True)


# https://superfastpython.com/multiprocessing-pool-python/#How_to_Configure_the_Multiprocessing_Pool
# https://superfastpython.com/multiprocessing-pool-num-workers/#Need_to_Configure_the_Number_of_Worker_Processes
# https://optuna.readthedocs.io/en/stable/reference/generated/optuna.load_study.html
def main():
    new_temp_dirs()

    samples = 1
    list_of_files = import_data_from_file()

    game_dfs = []

    device = get_tf_device()
    print("Using device:", device)

    for game_id, file in enumerate(list_of_files):
        file_df = pd.read_xml(file, xpath=".//instance")
        df = preprocess_data(file_df)
        df = convert_to_time_series(df, samples)

        df["game_id"] = game_id
        game_dfs.append(df)
    print("all games gathered")

    training_data = pd.concat(game_dfs, ignore_index=True)
    time_snapshot = time.time()

    # Values work best:
    number_of_trials = 24
    if device == "/CPU:0":
        number_of_processes = 8

        arguments = [(device, time_snapshot, number_of_trials, number_of_processes, training_data.copy()),
                 (device, time_snapshot, number_of_trials, number_of_processes, training_data.copy()),
                 (device, time_snapshot, number_of_trials, number_of_processes, training_data.copy()),
                 (device, time_snapshot, number_of_trials, number_of_processes, training_data.copy()),
                 (device, time_snapshot, number_of_trials, number_of_processes, training_data.copy()),
                 (device, time_snapshot, number_of_trials, number_of_processes, training_data.copy()),
                 (device, time_snapshot, number_of_trials, number_of_processes, training_data.copy()),
                 (device, time_snapshot, number_of_trials, number_of_processes, training_data.copy())]

        with Pool(processes=number_of_processes) as pool:
            pool.starmap(run_study, arguments)
    else:
        number_of_processes = 1
        run_study(device, time_snapshot, number_of_trials, number_of_processes, training_data.copy())

    # Access the data of the completed study
    completed_study = optuna.load_study(study_name="expected_goals", storage=JournalStorage(JournalFileBackend(file_path=f"temp/optuna/journals/journal{time_snapshot}.log")))
    best_trial = completed_study.best_trial
    best_parameters = best_trial.params
    print(f"Best trial was trial {completed_study.best_trial.number} with RMSE: {completed_study.best_value}")
    best_model_name = completed_study.best_trial.user_attrs.get("model_name")
    best_model = load_model(f"temp/optuna/temp/trial_saves/{best_model_name}.keras")
    print(f"temp/optuna/temp/trial_saves/{best_model_name}.keras")
    return best_model


if __name__ == "__main__":
   with keep.running():
       start = time.time()
       main()
       end = time.time()
       print(f"The program took {end - start} seconds to run")
