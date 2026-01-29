import os
import pickle

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
from preprocessing import pad_data_set
from run_study import objective
from preprocessing import smooth_data
from preprocessing import normalise_test
from preprocessing import get_data_columns
from wakepy import keep
from optuna.artifacts import FileSystemArtifactStore, download_artifact
from keras.models import load_model
import shutil

# If running on linux turn on:
# mp.set_start_method("spawn", force=True)


def get_and_split_data():
    # Get data from arguments
    if len(sys.argv) < 3:
        raise ImportError("Not enough arguments passed")

    training_data_file_path = sys.argv[1]
    testing_data_file_path = sys.argv[2]
    training_data = get_data_from_file(training_data_file_path)
    testing_data = get_data_from_file(testing_data_file_path)

    return training_data, testing_data

# https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/004_distributed.html
# https://optuna.readthedocs.io/en/stable/faq.html#id2
def run_study(time_snapshot, training_data, validation_data, number_of_trials, number_of_processes, artifact_store):
    study = optuna.create_study(directions=["minimize"], study_name="rul",
                                storage=JournalStorage(JournalFileBackend(file_path=f"temp/optuna/journals/journal{time_snapshot}.log")),
                                load_if_exists=True,
                                pruner=optuna.pruners.HyperbandPruner(min_resource=5, max_resource="auto",
                                                                      reduction_factor=4))

    study.optimize(lambda trial: objective(trial, training_data, validation_data, artifact_store), n_trials=(number_of_trials//number_of_processes))
    return study

# https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/003_attributes.html
def predict_results(testing_data, study, artifact_store):

    best_trial = study.best_trial
    best_parameters = best_trial.params
    print(f"Best trial was trial {study.best_trial.number} with RMSE: {study.best_value}")

    sequence_length = best_parameters["sequence_length"]
    include_smooth = best_parameters["include_smooth"]
    sigma = best_parameters["sigma"]
    batch_size = best_parameters["batch_size"]

    # Get artifact IDs
    best_model_name = study.best_trial.user_attrs.get("model_name")
    dropped_columns_artifact_id = study.best_trial.user_attrs.get("dropped_columns_artifact_id")
    normalising_scaler_artifact_id = study.best_trial.user_attrs.get("normalising_scaler_artifact_id")

    # Download saved artifacts
    download_artifact(artifact_store=artifact_store, file_path="temp/optuna/temp/best_attributes/dropped_columns.pkl", artifact_id=dropped_columns_artifact_id)
    download_artifact(artifact_store=artifact_store, file_path="temp/optuna/temp/best_attributes/normalising_scaler.pkl", artifact_id=normalising_scaler_artifact_id)

    # Access saved artifacts
    rul_model = load_model(f"temp/optuna/temp/trial_saves/{best_model_name}.keras")

    with open(f"temp/optuna/temp/best_attributes/dropped_columns.pkl", "rb") as f:
        dropped_columns = pickle.load(f)
    with open(f"temp/optuna/temp/best_attributes/normalising_scaler.pkl", "rb") as f:
        normalising_scaler = pickle.load(f)

    testing_data = testing_data.drop(columns=dropped_columns)

    if include_smooth:
        testing_data = smooth_data(testing_data, sigma)

    testing_data = pad_data_set(testing_data, sequence_length)


    testing_data = normalise_test(testing_data, normalising_scaler)

    testing_data = testing_data.sort_values(by=["engine_id", "cycle"]).reset_index(drop=True)

    data_cols = get_data_columns(testing_data)

    results_list = []

    for engine_id, engine in testing_data.groupby("engine_id"):

        # DEBUG TEST: inspect engine 1 vs others
        #print("\n--- ENGINE", engine_id, "---")
        #print("Shape:", subset.shape)
        #print("Columns:", list(subset.columns))
        #print("Head:\n", subset.head())

        # Only interested in last window
        last_window = engine.iloc[-sequence_length:]

        last_window_values = last_window[data_cols].values
        last_window = np.expand_dims(last_window_values, axis=0)

        predicted_result = rul_model.predict(last_window, batch_size=batch_size)
        predicted_rul = predicted_result[0][0]
        print(f"Predicted RUL for engine {engine_id}: {predicted_rul}")

        results_list.append({'engine' : engine_id, 'predicted_rul' : predicted_rul})

    results_df = pd.DataFrame(results_list)
    results_df.to_csv(r"temp/results.csv", index=False)


def new_temp_dirs():
    # Remove old temporary data
    shutil.rmtree("temp/optuna/temp", ignore_errors=True)
    # Create directories for new data
    os.makedirs("temp/optuna/temp", exist_ok=True)
    os.makedirs("temp/optuna/temp/trial_saves", exist_ok=True)
    os.makedirs("temp/optuna/temp/artifacts", exist_ok=True)
    os.makedirs("temp/optuna/temp/best_attributes", exist_ok=True)


# https://superfastpython.com/multiprocessing-pool-python/#How_to_Configure_the_Multiprocessing_Pool
# https://superfastpython.com/multiprocessing-pool-num-workers/#Need_to_Configure_the_Number_of_Worker_Processes
# https://optuna.readthedocs.io/en/stable/reference/generated/optuna.load_study.html
def main():
    new_temp_dirs()

    artifact_store = FileSystemArtifactStore(base_path="temp/optuna/temp/artifacts")
    training_data, testing_data = get_and_split_data()
    time_snapshot = time.time()

    training_data, validation_data = validate_train_split_data(training_data, validation_ratio=0.8)

    # Values work best:
    number_of_trials = 32
    number_of_processes = 8

    arguments = [(time_snapshot, training_data.copy(), validation_data.copy(), number_of_trials, number_of_processes, artifact_store),
                 (time_snapshot, training_data.copy(), validation_data.copy(), number_of_trials, number_of_processes, artifact_store),
                 (time_snapshot, training_data.copy(), validation_data.copy(), number_of_trials, number_of_processes, artifact_store),
                 (time_snapshot, training_data.copy(), validation_data.copy(), number_of_trials, number_of_processes, artifact_store),
                 (time_snapshot, training_data.copy(), validation_data.copy(), number_of_trials, number_of_processes, artifact_store),
                 (time_snapshot, training_data.copy(), validation_data.copy(), number_of_trials, number_of_processes, artifact_store),
                 (time_snapshot, training_data.copy(), validation_data.copy(), number_of_trials, number_of_processes, artifact_store),
                 (time_snapshot, training_data.copy(), validation_data.copy(), number_of_trials, number_of_processes, artifact_store),]

    with Pool(processes=number_of_processes) as pool:
        pool.starmap(run_study, arguments)

    # Access the data of the completed study
    completed_study = optuna.load_study(study_name="rul", storage=JournalStorage(JournalFileBackend(file_path=f"temp/optuna/journals/journal{time_snapshot}.log")))

    predict_results(testing_data, completed_study, artifact_store)


def get_data_from_file(file_path):
    col_names = (['engine_id', 'cycle', 'op_setting_1', 'op_setting_2', 'op_setting_3', ]
                 + [f'sensor_{i}' for i in range(1, 22)])

    data_set = pd.read_csv(file_path, sep=r"\s+", header=None)
    data_set = data_set.dropna(axis=1, how='all')

    if len(data_set.columns) != 26:
        raise ValueError("The data provided does not have 26 columns.")

    data_set.columns = col_names

    return data_set


def validate_train_split_data(training_data, validation_ratio):
    all_engines_list = sorted(training_data["engine_id"].unique())
    np.random.shuffle(all_engines_list)

    split_point = int(len(all_engines_list) * validation_ratio)

    training_engines = all_engines_list[:split_point]
    validation_engines = all_engines_list[split_point:]

    training_df = training_data[training_data["engine_id"].isin(training_engines)]
    validation_df = training_data[training_data["engine_id"].isin(validation_engines)]

    return training_df, validation_df


if __name__ == "__main__":
    with keep.running():
        start = time.time()
        main()
        end = time.time()
        print(f"The program took {end - start} seconds to run")
