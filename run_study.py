from optuna_integration import TFKerasPruningCallback
from keras.optimizers import Adam, RMSprop
from preprocessing import add_rul_labels
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from preprocessing import remove_non_relevant_data
from preprocessing import pad_data_set
from preprocessing import smooth_data
from preprocessing import get_data_columns
from preprocessing import normalise
from keras.models import Sequential
from keras.layers import LSTM, Dense, GRU, Masking, Input
import numpy as np
from optuna.artifacts import upload_artifact
import pickle

PREPROCESSING_CACHE = {}


# https://docs.databricks.com/aws/en/machine-learning/automl-hyperparam-tuning/optuna
# https://medium.com/@mihaitimoficiuc/predicting-jet-engine-failures-with-nasas-c-mapss-dataset-and-lstm-a-practical-guide-to-85b9513ea9ed
# https://www.marktechpost.com/2025/11/17/a-coding-guide-to-implement-advanced-hyperparameter-optimization-with-optuna-using-pruning-multi-objective-search-early-stopping-and-deep-visual-analysis/
# https://hiya31.medium.com/a-guide-to-lstm-hyperparameter-tuning-for-optimal-model-training-064f5c7f099d
# https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/012_artifact_tutorial.html
# https: // optuna.readthedocs.io / en / stable / faq.html  # how-to-save-machine-learning-models-trained-in-objective-functions
# https://www.geeksforgeeks.org/machine-learning/save-and-load-models-in-tensorflow/

def objective(trial, training_data, validation_data, artifact_store):

    # Get optuna to figure out which value to pick
    # Define all the parameters
    rul_cap                         = trial.suggest_int("rul_cap", 115, 135)
    #healthy_cycles                 = trial.suggest_int('healthy_cycles', 20, 40)
    #fault_threshold                = trial.suggest_float('fault_threshold', 0.01, 0.1)
    variance_threshold              = trial.suggest_float('variance_threshold', 0.01, 0.05)
    correlation_threshold           = trial.suggest_float('correlation_threshold', 0.15, 0.30)
    sigma                           = trial.suggest_float('sigma', 0.1, 0.5)
    model_type_name                 = trial.suggest_categorical("model_type", ["LSTM", "GRU"])
    if model_type_name == "LSTM":
        model_type = LSTM
    else:
        model_type = GRU
    sequence_length                 = trial.suggest_int("sequence_length", 20, 60)
    include_smooth                  = trial.suggest_categorical("include_smooth", [True, False])
    #k = trial.suggest_int("k", 1, 20)
    unit_layer_1                    = trial.suggest_int("lstm_layer_1", 50, 100)
    dropout_layer_1                 = trial.suggest_float("dropout_layer_1", 0.2, 0.5, step=0.1)
    recurrent_dropout_layer_1       = trial.suggest_float("recurrent_dropout_layer_1", 0.0, 0.5)
    unit_layer_2                    = trial.suggest_int("lstm_layer_2", 20, 50)
    dropout_layer_2                 = trial.suggest_float("dropout_layer_2", 0.2, 0.5, step=0.1)
    recurrent_dropout_layer_2       = trial.suggest_float("recurrent_dropout_layer_2", 0.0, 0.5)
    dense                           = trial.suggest_int("dense", 16, 48)
    activation                      = trial.suggest_categorical("activation", ["relu", "tanh", "elu"])
    optimizer_type                  = trial.suggest_categorical("optimizer_type", ["adam", "rmsprop"])
    if optimizer_type == "adam":
        selected_optimizer = Adam
    else:
        selected_optimizer = RMSprop
    learning_rate                   = trial.suggest_float("learning_rate", 0.0001, 0.001)
    loss                            = "mse"
    batch_size                      = trial.suggest_categorical("batch_size", [32, 64, 128])
    epochs                          = trial.suggest_int("epochs", 50, 100)

    training_data = pad_data_set(training_data, sequence_length)
    validation_data = pad_data_set(validation_data, sequence_length)



    training_df, rul_training_df = convert_to_windowed(training_data, sequence_length)
    validation_df, rul_validation_df = convert_to_windowed(validation_data, sequence_length)

    # Train
    # num_samples = training_df.shape[0]
    # training_df = training_df.reshape(num_samples, -1)
    # fault_model = KNeighborsClassifier(n_neighbors=k)
    # fault_model.fit(training_df, rul_training_df)
    # Validate
    # num_samples = validation_df.shape[0]
    # validation_df = validation_df.reshape(num_samples, -1)
    # predicted = fault_model.predict(validation_df)
    # f1 = f1_score(rul_validation_df, predicted)

    num_cols = training_df.shape[2]

    rul_model = build_rul_model(sequence_length, num_cols, unit_layer_1, dropout_layer_1, unit_layer_2, dropout_layer_2,
                            dense, activation, selected_optimizer, learning_rate, loss, model_type, recurrent_dropout_layer_1, recurrent_dropout_layer_2)

    # https://github.com/optuna/optuna-examples/blob/main/tfkeras/tfkeras_integration.py
    callbacks = TFKerasPruningCallback(trial, "val_loss")

    # Train model.
    history = rul_model.fit(training_df, rul_training_df, validation_data=(validation_df, rul_validation_df),
                            epochs=epochs, batch_size=batch_size, callbacks=callbacks)

    final_mse = history.history['val_loss'][-1]

    # Use RMS as the optuna objective
    rmse = np.sqrt(final_mse)


    model_name = f"trial_{trial.number}_model"
    dropped_columns_name = f"trial_{trial.number}_dropped_columns"
    normalising_scaler_name = f"trial_{trial.number}_normalising_scaler"

    rul_model.save(f"temp/optuna/temp/trial_saves/{model_name}.keras")

    # So they can be accessed after training
    with open(f"temp/optuna/temp/trial_saves/{dropped_columns_name}.pkl", "wb") as file:
        pickle.dump(dropped_columns, file)
    with open(f"temp/optuna/temp/trial_saves/{normalising_scaler_name}.pkl", "wb") as file:
        pickle.dump(normalising_scaler, file)

    # Upload files as artifacts
    dropped_columns_artifact_id = upload_artifact(artifact_store=artifact_store, file_path=f"temp/optuna/temp/trial_saves/{dropped_columns_name}.pkl", study_or_trial=trial)
    normalising_scaler_artifact_id = upload_artifact(artifact_store=artifact_store, file_path=f"temp/optuna/temp/trial_saves/{normalising_scaler_name}.pkl", study_or_trial=trial)

    # Store IDs to get later if needed
    trial.set_user_attr("model_name", model_name)
    trial.set_user_attr("dropped_columns_artifact_id", dropped_columns_artifact_id)
    trial.set_user_attr("normalising_scaler_artifact_id", normalising_scaler_artifact_id)

    return rmse


# https://www.geeksforgeeks.org/deep-learning/long-short-term-memory-lstm-rnn-in-tensorflow/
# https://stackoverflow.com/questions/75410827/how-does-masking-work-in-tensorflow-keras
def build_rul_model(sequence_length, num_cols, unit_layer_1, dropout_layer_1, unit_layer_2, dropout_layer_2, dense,
                    activation, optimizer_type, learning_rate, loss, model_type, recurrent_dropout_layer_1, recurrent_dropout_layer_2):

    model = Sequential()
    model.add(Input(shape=(sequence_length, num_cols)))
    model.add(Masking(mask_value=0.0))
    model.add(model_type(unit_layer_1, return_sequences=True, dropout=dropout_layer_1, recurrent_dropout=recurrent_dropout_layer_1))
    model.add(model_type(unit_layer_2, return_sequences=False, dropout=dropout_layer_2, recurrent_dropout=recurrent_dropout_layer_2))
    model.add(Dense(dense, activation=activation))
    model.add(Dense(1))

    model.compile(optimizer=optimizer_type(learning_rate=learning_rate, clipnorm=1.0), loss=loss, metrics=['mse', 'mae'])

    return model


# https://medium.com/@mihaitimoficiuc/predicting-jet-engine-failures-with-nasas-c-mapss-dataset-and-lstm-a-practical-guide-to-85b9513ea9ed
def convert_to_windowed(data_set, sequence_length):
    data_set = data_set.sort_values(["game_id", "time"]).reset_index(drop=True)

    data_windows_list = []
    goal_windows_list = []

    for engine_id, engine in data_set.groupby("game_id"):
        engine = engine.reset_index(drop=True)
        data_cols = get_data_columns(engine)

        # To avoid going into padded rows (where cycle less than 0):
        non_padded_rows = engine.loc[engine["time"] > 0]
        rul_values = non_padded_rows["gao"].reset_index(drop=True)
        data_col_values = non_padded_rows[data_cols]
        num_windows = len(rul_values)

        for i in range(num_windows):

            # Skip if window would be too big
            if i + sequence_length > len(data_col_values):
                continue

            # Slice the data to only show that window
            data_window = data_col_values[i: i + sequence_length]
            rul_window = rul_values[i + sequence_length - 1]

            data_windows_list.append(data_window)
            rul_windows_list.append(rul_window)

    return np.array(data_windows_list), np.array(rul_windows_list)

