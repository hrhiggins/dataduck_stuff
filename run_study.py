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

def objective(trial, training_data):

    # Get optuna to figure out which value to pick
    # Define all the parameters
    model_type_name                 = trial.suggest_categorical("model_type", ["LSTM", "GRU"])
    if model_type_name == "LSTM":
        model_type = LSTM
    else:
        model_type = GRU
    window_time                     = trial.suggest_int("window_time", 15, 60)
    horizon_time                    = trial.suggest_int("horizon_time", 3, 8)
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

    training_data = pad_data_set(training_data, window_time)

    data_set, goal_danger_df, expected_goals, game_ids= build_xg_windows(df=training_data, window_seconds=window_time, step_seconds=0.5, horizon_seconds=horizon_time)

    num_cols = training_df.shape[2]

    model = build_model(window_time, num_cols, unit_layer_1, dropout_layer_1, unit_layer_2, dropout_layer_2,
                            dense, activation, selected_optimizer, learning_rate, loss, model_type, recurrent_dropout_layer_1, recurrent_dropout_layer_2)

    # https://github.com/optuna/optuna-examples/blob/main/tfkeras/tfkeras_integration.py
    callbacks = TFKerasPruningCallback(trial, "val_loss")

    # Train model.
    history = model.fit(training_df, rul_training_df, validation_data=(validation_df, rul_validation_df),
                            epochs=epochs, batch_size=batch_size, callbacks=callbacks)

    final_mse = history.history['val_loss'][-1]

    # Use RMS as the optuna objective
    rmse = np.sqrt(final_mse)


    model_name = f"trial_{trial.number}_model"

    model.save(f"temp/optuna/temp/trial_saves/{model_name}.keras")

    # Store IDs to get later if needed
    trial.set_user_attr("model_name", model_name)

    return rmse


# https://www.geeksforgeeks.org/deep-learning/long-short-term-memory-lstm-rnn-in-tensorflow/
# https://stackoverflow.com/questions/75410827/how-does-masking-work-in-tensorflow-keras
def build_model(sequence_length, num_cols, unit_layer_1, dropout_layer_1, unit_layer_2, dropout_layer_2, dense,
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


def build_xg_windows(df, window_seconds, step_seconds, horizon_seconds, feature_col="features",
                     goal_col="goal_event", game_col="game_id"):


    # Convert seconds → timesteps
    window = int(window_seconds / step_seconds)
    horizon = int(horizon_seconds / step_seconds)

    X = []
    y_goal = []
    y_xg = []
    game_ids = []

    # Validate feature dimensions
    lengths = df[feature_col].apply(lambda x: x.shape[0])
    if lengths.nunique() != 1:
        raise ValueError(f"Inconsistent feature vector lengths detected:\n{lengths.value_counts()}")

    feature_dim = lengths.iloc[0]

    # Process each game separately
    for game_id, game_df in df.groupby(game_col):
        features = game_df[feature_col].tolist()
        goals = game_df[goal_col].tolist()

        num_steps = len(features)

        # Slide window across the game
        for i in range(num_steps - window - horizon):
            window_feats = features[i:i+window]

            # Safety check: all vectors must match
            if any(f.shape[0] != feature_dim for f in window_feats):
                continue  # skip corrupted window

            window = np.stack(window_feats)
            future = goals[i+window: i+window+horizon]

            X.append(window)
            y_goal.append(1 if any(future) else 0)
            y_xg.append(sum(future))  # simple xG target
            game_ids.append(game_id)

    return (np.array(X, dtype=np.float32), np.array(y_goal, dtype=np.float32), np.array(y_xg, dtype=np.float32),
            np.array(game_ids))
