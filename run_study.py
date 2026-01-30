from optuna_integration import TFKerasPruningCallback
from keras.optimizers import Adam, RMSprop
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from keras.models import Sequential
from keras.layers import LSTM, Dense, GRU, Masking, Input
import numpy as np
from optuna.artifacts import upload_artifact
from sklearn.model_selection import train_test_split
from keras.layers import (
    Input, Dense, LayerNormalization, Dropout,
    MultiHeadAttention, Add
)
from keras.models import Model
import tensorflow as tf






def objective(trial, training_data):

    # Model type
    model_type_name = trial.suggest_categorical("model_type", ["LSTM", "GRU"])
    model_type = LSTM if model_type_name == "LSTM" else GRU
    num_heads = trial.suggest_int("num_heads", 2, 8)
    ff_dim = trial.suggest_int("ff_dim", 64, 256)
    # Window + horizon
    window_time = trial.suggest_int("window_time", 15, 60)
    horizon_time = trial.suggest_int("horizon_time", 3, 8)

    # LSTM/GRU layers
    units1 = trial.suggest_int("units1", 50, 100)
    dropout1 = trial.suggest_float("dropout1", 0.2, 0.5)
    rec_dropout1 = trial.suggest_float("rec_dropout1", 0.0, 0.5)

    units2 = trial.suggest_int("units2", 20, 50)
    dropout2 = trial.suggest_float("dropout2", 0.2, 0.5)
    rec_dropout2 = trial.suggest_float("rec_dropout2", 0.0, 0.5)

    # Dense layer
    dense_units = trial.suggest_int("dense_units", 16, 48)
    activation = trial.suggest_categorical("activation", ["relu", "tanh", "elu"])

    # Optimizer
    optimizer_name = trial.suggest_categorical("optimizer", ["adam", "rmsprop"])
    optimizer_type = Adam if optimizer_name == "adam" else RMSprop
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-3)

    # Training params
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    epochs = trial.suggest_int("epochs", 20, 60)

    # Build windows
    X, y_goal, y_xg, game_ids = build_xg_windows(
        df=training_data,
        window_seconds=window_time,
        step_seconds=0.5,
        horizon_seconds=horizon_time
    )

    # Split by game_id
    unique_games = np.unique(game_ids)
    train_games, val_games = train_test_split(unique_games, test_size=0.2, random_state=42)

    train_mask = np.isin(game_ids, train_games)
    val_mask = np.isin(game_ids, val_games)

    X_train, X_val = X[train_mask], X[val_mask]
    y_goal_train, y_goal_val = y_goal[train_mask], y_goal[val_mask]
    y_xg_train, y_xg_val = y_xg[train_mask], y_xg[val_mask]

    feature_dim = X.shape[2]

    # Build model
    model = build_dual_head_model(
        sequence_length=X.shape[1],
        feature_dim=feature_dim,
        model_type=model_type,
        units1=units1,
        dropout1=dropout1,
        rec_dropout1=rec_dropout1,
        units2=units2,
        dropout2=dropout2,
        rec_dropout2=rec_dropout2,
        dense_units=dense_units,
        activation=activation,
        optimizer_type=optimizer_type,
        learning_rate=learning_rate,
        num_heads=num_heads,
        ff_dim=ff_dim
    )

    callbacks = [
        TFKerasPruningCallback(trial, "val_xg_mse")
    ]

    history = model.fit(
        X_train,
        {"goal_prob": y_goal_train, "xg": y_xg_train},
        validation_data=(X_val, {"goal_prob": y_goal_val, "xg": y_xg_val}),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0
    )

    final_mse = history.history["val_xg_mse"][-1]
    rmse = np.sqrt(final_mse)

    model_name = f"trial_{trial.number}_dual_head"
    model.save(f"temp/optuna/temp/trial_saves/{model_name}.keras")
    trial.set_user_attr("model_name", model_name)

    return rmse


def build_xg_windows(
    df,
    window_seconds,
    step_seconds,
    horizon_seconds,
    feature_col="features",
    goal_col="goal_event",
    game_col="game_id"
):
    window = int(window_seconds / step_seconds)
    horizon = int(horizon_seconds / step_seconds)

    X, y_goal, y_xg, game_ids = [], [], [], []

    lengths = df[feature_col].apply(lambda x: x.shape[0])
    if lengths.nunique() != 1:
        raise ValueError(f"Inconsistent feature vector lengths:\n{lengths.value_counts()}")

    feature_dim = lengths.iloc[0]

    for game_id, game_df in df.groupby(game_col):
        features = game_df[feature_col].tolist()
        goals = game_df[goal_col].tolist()
        num_steps = len(features)

        for i in range(num_steps - window - horizon):
            window_feats = features[i:i + window]

            if any(f.shape[0] != feature_dim for f in window_feats):
                continue

            X.append(np.stack(window_feats))
            future = goals[i + window : i + window + horizon]

            y_goal.append(1 if any(future) else 0)
            y_xg.append(sum(future))
            game_ids.append(game_id)

    return (
        np.array(X, dtype=np.float32),
        np.array(y_goal, dtype=np.float32),
        np.array(y_xg, dtype=np.float32),
        np.array(game_ids)
    )


def transformer_encoder(x, num_heads, ff_dim, dropout):
    # Multi-head self-attention
    attn_output = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=ff_dim,
        dropout=dropout
    )(x, x)

    # Residual + norm
    x = Add()([x, attn_output])
    x = LayerNormalization(epsilon=1e-6)(x)

    # Feed-forward block
    ff_output = Dense(ff_dim, activation="relu")(x)
    ff_output = Dense(x.shape[-1])(ff_output)

    # Residual + norm
    x = Add()([x, ff_output])
    x = LayerNormalization(epsilon=1e-6)(x)

    return x


def build_dual_head_model(
    sequence_length,
    feature_dim,
    model_type,  # ignored but kept for compatibility
    units1,
    dropout1,
    rec_dropout1,
    units2,
    dropout2,
    rec_dropout2,
    dense_units,
    activation,
    optimizer_type,
    learning_rate,
    num_heads,
    ff_dim
):
    inputs = Input(shape=(sequence_length, feature_dim))

    # Positional encoding (trainable)
    pos_encoding = tf.range(start=0, limit=sequence_length, delta=1)
    pos_encoding = tf.keras.layers.Embedding(
        input_dim=sequence_length,
        output_dim=feature_dim
    )(pos_encoding)

    x = inputs + pos_encoding

    # Transformer blocks
    x = transformer_encoder(
        x,
        num_heads=num_heads,
        ff_dim=ff_dim,
        dropout=dropout1
    )

    x = transformer_encoder(
        x,
        num_heads=num_heads,
        ff_dim=ff_dim,
        dropout=dropout2
    )

    # Global pooling
    x = tf.keras.layers.GlobalAveragePooling1D()(x)

    # Dense layer
    x = Dense(dense_units, activation=activation)(x)

    # Outputs
    goal_prob = Dense(1, activation="sigmoid", name="goal_prob")(x)
    xg = Dense(1, activation="linear", name="xg")(x)

    model = Model(inputs, [goal_prob, xg])

    model.compile(
        optimizer=optimizer_type(learning_rate=learning_rate),
        loss={
            "goal_prob": "binary_crossentropy",
            "xg": "mse"
        },
        metrics={
            "goal_prob": ["accuracy"],
            "xg": ["mse"]
        }
    )

    return model