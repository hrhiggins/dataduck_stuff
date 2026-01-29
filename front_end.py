import numpy as np
import pandas as pd
import time
from wakepy import keep
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout

def build_lstm_model(window_size, feature_dim):

    inputs = Input(shape=(window_size, feature_dim))

    # Shared temporal encoder
    x = LSTM(128, return_sequences=False)(inputs)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.2)(x)

    # Goal probability head
    goal_prob = Dense(1, activation="sigmoid", name="goal_prob")(x)

    # Expected goals head
    xg = Dense(1, activation="linear", name="xg")(x)

    model = Model(inputs, [goal_prob, xg])

    model.compile(
        optimizer="adam",
        loss={
            "goal_prob": "binary_crossentropy",
            "xg": "mse"
        },
        metrics={
            "goal_prob": ["accuracy", "AUC"],
            "xg": ["mse"]
        }
    )

    return model


def build_windows(df, window_seconds, step_seconds, horizon_seconds):
    """
    df must contain: time, features, goal_event, game_id
    """

    # Convert seconds → timesteps
    WINDOW = int(window_seconds / step_seconds)      # 45 / 0.5 = 90
    HORIZON = int(horizon_seconds / step_seconds)    # 5 / 0.5 = 10

    X = []
    y_goal = []
    y_xg = []
    game_ids = []

    for game_id, game_df in df.groupby("game_id"):
        features = game_df["features"].tolist()
        goals = game_df["goal_event"].tolist()

        num_steps = len(features)

        for i in range(num_steps - WINDOW - HORIZON):
            window = np.stack(features[i:i+WINDOW])
            future = goals[i+WINDOW : i+WINDOW+HORIZON]

            X.append(window)
            y_goal.append(1 if any(future) else 0)
            y_xg.append(sum(future))  # simple xG target
            game_ids.append(game_id)

    X = np.array(X, dtype=np.float32)
    y_goal = np.array(y_goal, dtype=np.float32)
    y_xg = np.array(y_xg, dtype=np.float32)
    game_ids = np.array(game_ids)

    return X, y_goal, y_xg, game_ids


def parse_feature_string(s):
    s = s.strip()[1:-1]  # remove [ and ]
    if not s:
        return np.array([], dtype=np.float32)
    return np.fromstring(s, sep=' ', dtype=np.float32)


def main():
    data = pd.read_csv("time_series.csv")

    # Convert feature strings back to numpy arrays
    data["features"] = data["features"].apply(parse_feature_string)

    # Build windows
    x_data, y_goal, y_xg, game_ids = build_windows(
        data,
        window_seconds=45,
        step_seconds=0.5,
        horizon_seconds=5
    )

    # Determine feature dimension
    feature_dim = data["features"].iloc[0].shape[0]

    # Build model
    model = build_lstm_model(window_size=90, feature_dim=feature_dim)
    model.summary()


if __name__ == "__main__":
    with keep.running():
        start = time.time()
        main()
        end = time.time()
        print(f"The program took {end - start} seconds to run")

