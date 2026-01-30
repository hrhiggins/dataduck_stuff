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
from windowing import WindowGenerator

def objective(trial, training_data):
    tf.keras.backend.clear_session()

    #num_heads = trial.suggest_int("num_heads", 2, 3)
    num_heads = trial.suggest_categorical("num_heads", [2])
    ff_dim = trial.suggest_int("ff_dim", 16, 48)
    dropout1 = trial.suggest_float("dropout1", 0.0, 0.15)
    dropout2 = trial.suggest_float("dropout2", 0.0, 0.15)

    dense_units = trial.suggest_int("dense_units", 8, 24)
    activation = trial.suggest_categorical("activation", ["relu", "tanh"])

    optimizer_name = trial.suggest_categorical("optimizer", ["adam"])
    optimizer_type = Adam if optimizer_name == "adam" else RMSprop
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 5e-4)

    #batch_size = trial.suggest_categorical("batch_size", [32, 64])
    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])
    epochs = trial.suggest_int("epochs", 10, 20)

    window_seconds = 40
    horizon_seconds = 12
    step_seconds = 1.0

    window = int(window_seconds / step_seconds)
    horizon = int(horizon_seconds / step_seconds)

    unique_games = training_data["game_id"].unique()
    train_games, val_games = train_test_split(unique_games, test_size=0.2, random_state=42)

    train_df = training_data[training_data["game_id"].isin(train_games)]
    val_df = training_data[training_data["game_id"].isin(val_games)]

    train_gen = WindowGenerator(
        df=train_df,
        window=window,
        horizon=horizon,
        step=1,
        batch_size=batch_size
    )

    val_gen = WindowGenerator(
        df=val_df,
        window=window,
        horizon=horizon,
        step=1,
        batch_size=batch_size
    )

    # Feature dimension from cached data
    first_game_id = next(iter(train_gen.games))
    feature_dim = train_gen.games[first_game_id][0].shape[1]

    model = build_dual_head_model(
        sequence_length=window,
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
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks,
        validation_freq=3
    )

    final_mse = history.history["val_xg_mse"][-1]
    rmse = np.sqrt(final_mse)

    if trial.number == 0 or rmse < trial.study.best_value:
        model_name = "best_model"
        model.save(f"temp/optuna/temp/trial_saves/{model_name}.keras")
        trial.set_user_attr("model_name", model_name)

    del model, history, train_gen, val_gen
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

    #x = transformer_encoder(
    #    x,
    #    num_heads=num_heads,
    #    ff_dim=ff_dim,
    #    dropout=dropout2
    #)

    # Global pooling
    x = tf.keras.layers.GlobalAveragePooling1D()(x)

    # Dense layer
    x = Dense(dense_units, activation=activation)(x)

    # Outputs
    goal_prob = Dense(1, activation="sigmoid", name="goal_prob")(x)
    xg = Dense(1, activation="sigmoid", name="xg")(x)

    model = Model(inputs, [goal_prob, xg])

    model.compile(
        optimizer=optimizer_type(learning_rate=learning_rate),
        loss={
            "goal_prob": "binary_crossentropy",
            "xg": "mse"
        },
        loss_weights={
            "goal_prob": 1.0,
            "xg": 0.3
        },
        metrics={
            "xg": ["mse"]
        }
    )

    return model