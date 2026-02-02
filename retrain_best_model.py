import optuna
import tensorflow as tf
import numpy as np
import gc

from run_study import (
    UncertaintyLoss,
    build_dual_head_model,
)
from windowing import WindowGenerator, WarmupCosine
from train_model import preprocess_data, convert_to_time_series


# ------------------------------------------------------------
# Load your Optuna study
# ------------------------------------------------------------
study = optuna.load_study(
    study_name="expected_goals",
    storage="sqlite:///optuna_study.db"
)

best = study.best_trial.params
print("Best trial:", best)


# ------------------------------------------------------------
# Load your training data again
# ------------------------------------------------------------
import pandas as pd

df = pd.read_pickle("training_data.pkl")   # adjust if needed

unique_games = df["game_id"].unique()
train_games = unique_games[: int(0.8 * len(unique_games))]
val_games = unique_games[int(0.8 * len(unique_games)) :]

train_df = df[df["game_id"].isin(train_games)]
val_df = df[df["game_id"].isin(val_games)]


# ------------------------------------------------------------
# Windowing
# ------------------------------------------------------------
window_seconds = 40
horizon_seconds = 12
step_seconds = 1.0

window = int(window_seconds / step_seconds)
horizon = int(horizon_seconds / step_seconds)
global_len = 80

batch_size = 256

train_gen = WindowGenerator(
    df=train_df,
    window=window,
    horizon=horizon,
    step=1,
    batch_size=batch_size,
    global_len=global_len,
)

val_gen = WindowGenerator(
    df=val_df,
    window=window,
    horizon=horizon,
    step=1,
    batch_size=batch_size,
    global_len=global_len,
)

first_game_id = next(iter(train_gen.games))
feature_dim = train_gen.games[first_game_id][0].shape[1]


# ------------------------------------------------------------
# Rebuild the model using best hyperparameters
# ------------------------------------------------------------
model = build_dual_head_model(
    local_seq_len=window,
    global_seq_len=global_len,
    feature_dim=feature_dim,
    dropout1=best["dropout1"],
    dense_units=best["dense_units"],
    activation="tanh",
    num_heads=best["num_heads"],
    ff_dim=best["ff_dim"],
    key_dim=best["key_dim"],
    ff_activation=best["ff_activation"],
    pooling="max",
    xg_activation="linear",
    pos_dim=best["pos_dim"],
)

lr_schedule = WarmupCosine(
    base_lr=best["learning_rate"],
    warmup_steps=1000,
    total_steps=10000,
)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

loss = {
    "goal_prob": UncertaintyLoss(tf.keras.losses.BinaryCrossentropy(), "goal_prob"),
    "xg": UncertaintyLoss(tf.keras.losses.MeanSquaredError(), "xg"),
    "penalty_corner": UncertaintyLoss(tf.keras.losses.BinaryCrossentropy(), "penalty_corner"),
    "penalty_stroke": UncertaintyLoss(tf.keras.losses.BinaryCrossentropy(), "penalty_stroke"),
    "circle_entry": UncertaintyLoss(tf.keras.losses.BinaryCrossentropy(), "circle_entry"),
}

model.compile(
    optimizer=optimizer,
    loss=loss,
    metrics={"xg": ["mse"]},
)

print("Retraining best model…")

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=best["epochs"],
    verbose=1,
)

print("Saving corrected model…")
model.save("temp/optuna/best_model.keras")

print("Done. Model saved and ready for Analyse.py.")