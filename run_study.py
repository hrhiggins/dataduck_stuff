from optuna_integration import TFKerasPruningCallback
from keras.optimizers import Adam, RMSprop
from sklearn.model_selection import train_test_split
from keras.layers import (
    Input, Dense, LayerNormalization, Dropout,
    MultiHeadAttention, Add, Embedding
)
from keras.models import Model
import tensorflow as tf
import numpy as np
import gc
from windowing import WindowGenerator


def objective(trial, training_data):
    tf.keras.backend.clear_session()

    # --- Hyperparameters ---
    num_heads = trial.suggest_categorical("num_heads", [2])
    ff_dim = trial.suggest_int("ff_dim", 18, 28)
    key_dim = trial.suggest_int("key_dim", 8, 32)

    dropout1 = trial.suggest_float("dropout1", 0.05, 0.15)

    dense_units = trial.suggest_int("dense_units", 22, 34)
    activation = trial.suggest_categorical("activation", ["tanh"])

    ff_activation = trial.suggest_categorical("ff_activation", ["relu", "gelu", "tanh"])
    pooling = trial.suggest_categorical("pooling", ["avg", "max"])
    xg_weight = trial.suggest_float("xg_weight", 0.1, 0.6)
    xg_activation = trial.suggest_categorical("xg_activation", ["sigmoid", "linear"])

    use_second_block = trial.suggest_categorical("use_second_block", [True, False])

    pos_dim = trial.suggest_int("pos_dim", 8, 64)

    optimizer_name = trial.suggest_categorical("optimizer", ["adam"])
    optimizer_type = Adam if optimizer_name == "adam" else RMSprop
    learning_rate = trial.suggest_float("learning_rate", 1.5e-4, 3.5e-4)

    batch_size = trial.suggest_categorical("batch_size", [256, 512])
    epochs = trial.suggest_int("epochs", 8, 14)
    penalty_stroke_weight = trial.suggest_float("penalty_corner_weight", 0.5, 2.0)
    penalty_corner_weight = trial.suggest_float("penalty_stroke_weight", 0.2, 0.8)
    circle_entry_weight = trial.suggest_float("circle_entry_weight", 0.05, 0.3)

    # --- Windowing ---
    window_seconds = 40
    horizon_seconds = 12
    step_seconds = 1.0

    window = int(window_seconds / step_seconds)
    horizon = int(horizon_seconds / step_seconds)

    # --- Train/Val split ---
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

    # Feature dimension
    first_game_id = next(iter(train_gen.games))
    feature_dim = train_gen.games[first_game_id][0].shape[1]

    # --- Build model ---
    model = build_dual_head_model(
        sequence_length=window,
        feature_dim=feature_dim,
        dropout1=dropout1,
        dense_units=dense_units,
        activation=activation,
        optimizer_type=optimizer_type,
        learning_rate=learning_rate,
        num_heads=num_heads,
        ff_dim=ff_dim,
        key_dim=key_dim,
        ff_activation=ff_activation,
        pooling=pooling,
        xg_weight=xg_weight,
        xg_activation=xg_activation,
        use_second_block=use_second_block,
        pos_dim=pos_dim,
        penalty_corner_weight=penalty_corner_weight,
        penalty_stroke_weight=penalty_stroke_weight,
        circle_entry_weight=circle_entry_weight
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


# --- Transformer Encoder ---
def transformer_encoder(x, num_heads, ff_dim, key_dim, dropout, ff_activation):
    attn_output = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=key_dim,
        dropout=dropout
    )(x, x)

    x = Add()([x, attn_output])
    x = LayerNormalization(epsilon=1e-6)(x)

    ff_output = Dense(ff_dim, activation=ff_activation)(x)
    ff_output = Dense(x.shape[-1])(ff_output)

    x = Add()([x, ff_output])
    x = LayerNormalization(epsilon=1e-6)(x)

    return x


# --- Full Model ---
def build_dual_head_model(
    sequence_length,
    feature_dim,
    dropout1,
    dense_units,
    activation,
    optimizer_type,
    learning_rate,
    num_heads,
    ff_dim,
    key_dim,
    ff_activation,
    pooling,
    xg_weight,
    xg_activation,
    use_second_block,
    pos_dim,
    penalty_corner_weight,
    penalty_stroke_weight,
    circle_entry_weight
):
    inputs = Input(shape=(sequence_length, feature_dim))

    # Positional encoding
    pos_encoding = Embedding(
        input_dim=sequence_length,
        output_dim=pos_dim
    )(tf.range(sequence_length))

    pos_encoding = Dense(feature_dim)(pos_encoding)
    pos_encoding = tf.expand_dims(pos_encoding, axis=0)

    x = inputs + pos_encoding

    # Transformer block 1
    x = transformer_encoder(
        x,
        num_heads=num_heads,
        ff_dim=ff_dim,
        key_dim=key_dim,
        dropout=dropout1,
        ff_activation=ff_activation
    )

    # Optional second block
    if use_second_block:
        x = transformer_encoder(
            x,
            num_heads=num_heads,
            ff_dim=ff_dim,
            key_dim=key_dim,
            dropout=dropout1,
            ff_activation=ff_activation
        )

    # Pooling
    if pooling == "avg":
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
    else:
        x = tf.keras.layers.GlobalMaxPooling1D()(x)

    # Dense layer
    x = Dense(dense_units, activation=activation)(x)

    # Outputs
    goal_prob = Dense(1, activation="sigmoid", name="goal_prob")(x)
    xg = Dense(1, activation=xg_activation, name="xg")(x)
    penalty_corner = Dense(1, activation="sigmoid", name="penalty_corner")(x)
    penalty_stroke = Dense(1, activation="sigmoid", name="penalty_stroke")(x)
    circle_entry = Dense(1, activation="sigmoid", name="circle_entry")(x)

    model = Model(inputs, [goal_prob, xg, penalty_corner, penalty_stroke, circle_entry])

    model.compile(
        optimizer=optimizer_type(learning_rate=learning_rate),
        loss={
            "goal_prob": "binary_crossentropy",
            "xg": "mse",
            "penalty_corner": "binary_crossentropy",
            "penalty_stroke": "binary_crossentropy",
            "circle_entry": "binary_crossentropy"
        },
        loss_weights={
            "goal_prob": 1.0,
            "xg": xg_weight,
            "penalty_corner": penalty_corner_weight,
            "penalty_stroke": penalty_stroke_weight,
            "circle_entry": circle_entry_weight
        },
        metrics={
            "xg": ["mse"]
        }
    )

    return model