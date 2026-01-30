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
import gc


def objective(trial, X, y_goal, y_xg, game_ids):
    tf.keras.backend.clear_session()

    # Model type
    #model_type_name = trial.suggest_categorical("model_type", ["LSTM", "GRU"])
    #model_type = LSTM if model_type_name == "LSTM" else GRU
    num_heads = trial.suggest_int("num_heads", 2, 4)
    ff_dim = trial.suggest_int("ff_dim", 64, 128)
    dropout1 = trial.suggest_float("dropout1", 0.0, 0.2)
    dropout2 = trial.suggest_float("dropout2", 0.0, 0.2)

    # Dense layer
    dense_units = trial.suggest_int("dense_units", 16, 48)
    activation = trial.suggest_categorical("activation", ["relu", "tanh"])

    # Optimizer
    optimizer_name = trial.suggest_categorical("optimizer", ["adam"])
    optimizer_type = Adam if optimizer_name == "adam" else RMSprop
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 5e-4)

    # Training params
    batch_size = trial.suggest_categorical("batch_size", [32, 64])
    epochs = trial.suggest_int("epochs", 10, 30)

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
        dropout1=dropout1,
        dropout2=dropout2,
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

    # Save only if this trial is the best so far
    if trial.study.best_trial is None or rmse < trial.study.best_value:
        model_name = f"best_model"
        model.save(f"temp/optuna/temp/trial_saves/{model_name}.keras")
        trial.set_user_attr("model_name", model_name)

    del model, history
    gc.collect()
    tf.keras.backend.clear_session()

    return rmse




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
    dropout1,
    dropout2,
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