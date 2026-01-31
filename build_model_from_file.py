import optuna


def main():
    # Access the data of the completed study
    completed_study = optuna.load_study(study_name="expected_goals", storage=JournalStorage(JournalFileBackend(file_path=f"temp/optuna/journals/journal{time_snapshot}.log")))
    best_model_name = completed_study.best_trial.user_attrs.get("model_name")
    best_params = completed_study.best_trial.params

    # --- Windowing ---
    window_seconds = 60
    horizon_seconds = 12
    step_seconds = 1.0

    window = int(window_seconds / step_seconds)
    horizon = int(horizon_seconds / step_seconds)

    # --- Train/Val split ---
    unique_games = training_data["game_id"].unique()

    train_df = training_data[training_data["game_id"].isin(train_games)]

    train_gen = WindowGenerator(
        df=train_df,
        window=window,
        horizon=horizon,
        step=1,
        batch_size=batch_size,
        global_len=120
    )


    # Feature dimension
    first_game_id = next(iter(train_gen.games))
    feature_dim = train_gen.games[first_game_id][0].shape[1]
    global_len = train_gen.global_len



    # Rebuild model with best hyperparameters
    # (same call as in objective, but using best_params instead of trial.suggest_*)
    model = build_dual_head_model(
        local_seq_len=window,
        global_seq_len=120,
        feature_dim=feature_dim,  # you can recompute from training_data
        dropout1=best_params["dropout1"],
        dense_units=best_params["dense_units"],
        activation="tanh",
        optimizer_type=Adam,
        learning_rate=best_params["learning_rate"],
        num_heads=best_params["num_heads"],
        ff_dim=best_params["ff_dim"],
        key_dim=best_params["key_dim"],
        ff_activation=best_params["ff_activation"],
        pooling="max",
        xg_weight=best_params["xg_weight"],
        xg_activation="linear",
        pos_dim=best_params["pos_dim"],
        penalty_corner_weight=best_params["penalty_corner_weight"],
        penalty_stroke_weight=best_params["penalty_stroke_weight"],
        circle_entry_weight=best_params["circle_entry_weight"],
    )

    model.load_weights(f"temp/optuna/temp/trial_saves/{best_model_name}.weights.h5")
    best_model = model

