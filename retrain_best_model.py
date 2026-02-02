import optuna
import tensorflow as tf
import numpy as np
import pandas as pd
import gc
import os

from run_study import UncertaintyLoss, build_dual_head_model
from windowing import WindowGenerator, WarmupCosine
from train_model import preprocess_data, convert_to_time_series
from optuna.storages.journal import JournalFileBackend
from optuna.storages import JournalStorage



# ============================================================
# Load and prepare training data
# ============================================================

def load_training_data():
    """
    Loads all XML files passed as command-line arguments,
    preprocesses them, converts to time series, and concatenates.
    """
    import sys
    import os

    if len(sys.argv) < 2:
        raise RuntimeError("Usage: python retrain_best_model.py <directory1> <directory2> ...")

    all_files = []
    for arg in sys.argv[1:]:
        if not os.path.isdir(arg):
            raise NotADirectoryError(f"{arg} is not a valid directory")

        for f in os.listdir(arg):
            full = os.path.join(arg, f)
            if os.path.isfile(full):
                all_files.append(full)

    game_dfs = []
    samples = 1

    for game_id, file in enumerate(all_files):
        print(f"Loading {file}")
        xml_df = pd.read_xml(file, xpath=".//instance")
        df = preprocess_data(xml_df)
        df = convert_to_time_series(df, samples)
        df["game_id"] = game_id
        game_dfs.append(df)

    print("All games loaded.")
    return pd.concat(game_dfs, ignore_index=True)


# ============================================================
# Build generators
# ============================================================

def make_generators(training_df):
    window_seconds = 40
    horizon_seconds = 12
    step_seconds = 1.0

    window = int(window_seconds / step_seconds)
    horizon = int(horizon_seconds / step_seconds)
    global_len = 80
    batch_size = 256

    unique_games = training_df["game_id"].unique()
    split = int(0.8 * len(unique_games))

    train_games = unique_games[:split]
    val_games = unique_games[split:]

    train_df = training_df[training_df["game_id"].isin(train_games)]
    val_df = training_df[training_df["game_id"].isin(val_games)]

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

    return train_gen, val_gen, feature_dim, window, global_len


# ============================================================
# Main retraining logic
# ============================================================

def main():
    # --------------------------------------------------------
    # Load Optuna study
    # --------------------------------------------------------

    study = optuna.load_study(
        study_name="expected_goals",
        storage=JournalStorage(JournalFileBackend("temp/optuna/journals/journal_optuna_search_trial.log")),
    )

    best = study.best_trial.params
    print("\nBest trial parameters:")
    for k, v in best.items():
        print(f"  {k}: {v}")

    # --------------------------------------------------------
    # Load training data
    # --------------------------------------------------------
    training_df = load_training_data()

    # --------------------------------------------------------
    # Build generators
    # --------------------------------------------------------
    train_gen, val_gen, feature_dim, window, global_len = make_generators(training_df)

    # --------------------------------------------------------
    # Rebuild model
    # --------------------------------------------------------
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

    # --------------------------------------------------------
    # Retrain best model
    # --------------------------------------------------------
    print("\nRetraining best model...\n")

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=best["epochs"],
        verbose=1,
    )

    # --------------------------------------------------------
    # Save final model
    # --------------------------------------------------------
    os.makedirs("temp/optuna", exist_ok=True)
    model.save("temp/optuna/best_model.keras")

    print("\nModel saved to temp/optuna/best_model.keras")
    print("Retraining complete.")


if __name__ == "__main__":
    main()