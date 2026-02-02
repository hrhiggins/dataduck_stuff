from optuna_integration import TFKerasPruningCallback
from keras.optimizers import Adam, RMSprop
from sklearn.model_selection import train_test_split
from keras.layers import (
    Input, Dense, LayerNormalization, Dropout,
    MultiHeadAttention, Add, Embedding, GlobalAveragePooling1D,
    GlobalMaxPooling1D, Concatenate
)
from keras.models import Model
import tensorflow as tf
import numpy as np
import gc

from windowing import WindowGenerator, WarmupCosine


# ============================================================
# Transformer blocks
# ============================================================

def transformer_encoder(x, num_heads, ff_dim, key_dim, dropout, ff_activation):
    attn_output = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=key_dim,
        dropout=dropout
    )(x, x)

    x = Add()([x, attn_output])
    x = LayerNormalization(epsilon=1e-6)(x)

    ff_output = Dense(ff_dim, activation=ff_activation)(x)
    ff_output = Dropout(dropout)(ff_output)
    ff_output = Dense(x.shape[-1])(ff_output)
    ff_output = Dropout(dropout)(ff_output)

    x = Add()([x, ff_output])
    x = LayerNormalization(epsilon=1e-6)(x)

    return x


def cross_attention_block(local_x, global_x, num_heads, key_dim, dropout):
    attn_output = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=key_dim,
        dropout=dropout
    )(local_x, global_x)

    x = Add()([local_x, attn_output])
    x = LayerNormalization(epsilon=1e-6)(x)
    return x


# ============================================================
# Uncertainty-based loss weighting (per-task Loss objects)
# ============================================================

class UncertaintyLoss(tf.keras.losses.Loss):
    def __init__(self, base_loss_fn, name_prefix, **kwargs):
        super().__init__(**kwargs)
        self.base_loss_fn = base_loss_fn
        self.log_var = tf.Variable(
            0.0,
            trainable=True,
            name=f"{name_prefix}_log_var",
            dtype=tf.float32,
        )

    def call(self, y_true, y_pred):
        base = self.base_loss_fn(y_true, y_pred)
        precision = tf.exp(-self.log_var)
        return precision * base + self.log_var


# ============================================================
# Dual-stream model (local + global, cross-attention)
# ============================================================

def build_dual_head_model(
    local_seq_len,
    global_seq_len,
    feature_dim,
    dropout1,
    dense_units,
    activation,
    num_heads,
    ff_dim,
    key_dim,
    ff_activation,
    pooling,
    xg_activation,
    pos_dim
):
    # ----- Inputs -----
    local_inputs = Input(shape=(local_seq_len, feature_dim), name="local_input")
    global_inputs = Input(shape=(global_seq_len, feature_dim), name="global_input")

    # ----- Positional encodings -----
    # Local
    local_pos = Embedding(
        input_dim=local_seq_len,
        output_dim=pos_dim
    )(tf.range(local_seq_len))
    local_pos = Dense(feature_dim)(local_pos)
    local_pos = tf.keras.ops.expand_dims(local_pos, axis=0)
    local_x = local_inputs + local_pos

    # Global
    global_pos = Embedding(
        input_dim=global_seq_len,
        output_dim=pos_dim
    )(tf.range(global_seq_len))
    global_pos = Dense(feature_dim)(global_pos)
    global_pos = tf.keras.ops.expand_dims(global_pos, axis=0)
    global_x = global_inputs + global_pos

    # ----- Local transformer blocks -----
    for _ in range(2):
        local_x = transformer_encoder(
            local_x,
            num_heads=num_heads,
            ff_dim=ff_dim,
            key_dim=key_dim,
            dropout=dropout1,
            ff_activation=ff_activation
        )
        local_x = Dropout(dropout1)(local_x)

    # ----- Global transformer blocks -----
    for _ in range(2):
        global_x = transformer_encoder(
            global_x,
            num_heads=num_heads,
            ff_dim=ff_dim,
            key_dim=key_dim,
            dropout=dropout1,
            ff_activation=ff_activation
        )
        global_x = Dropout(dropout1)(global_x)

    # ----- Cross-attention: local attends to global -----
    local_x = cross_attention_block(
        local_x,
        global_x,
        num_heads=num_heads,
        key_dim=key_dim,
        dropout=dropout1
    )

    # ----- Pooling -----
    if pooling == "avg":
        local_pooled = GlobalAveragePooling1D()(local_x)
        global_pooled = GlobalAveragePooling1D()(global_x)
    else:
        local_pooled = GlobalMaxPooling1D()(local_x)
        global_pooled = GlobalMaxPooling1D()(global_x)

    # ----- Fusion -----
    fused = Concatenate()([local_pooled, global_pooled])

    # Shared dense layer
    x = Dense(dense_units, activation=activation)(fused)

    # Goal probability head
    goal_prob = Dense(1, activation="sigmoid", name="goal_prob")(x)

    # xG head
    xg_hidden = Dense(16, activation="gelu")(x)
    xg = Dense(1, activation=xg_activation, name="xg")(xg_hidden)

    # Other event heads
    penalty_corner = Dense(1, activation="sigmoid", name="penalty_corner")(x)
    penalty_stroke = Dense(1, activation="sigmoid", name="penalty_stroke")(x)
    circle_entry = Dense(1, activation="sigmoid", name="circle_entry")(x)

    model = Model(
        inputs=[local_inputs, global_inputs],
        outputs=[goal_prob, xg, penalty_corner, penalty_stroke, circle_entry]
    )

    return model


# ============================================================
# Optuna objective
# ============================================================

def objective(trial, training_data):
    tf.keras.backend.clear_session()

    # --- Hyperparameters ---
    num_heads = trial.suggest_categorical("num_heads", [2, 4])
    key_dim = trial.suggest_int("key_dim", 24, 40)
    ff_dim = trial.suggest_int("ff_dim", 48, 96)

    dropout1 = trial.suggest_float("dropout1", 0.05, 0.15)

    dense_units = trial.suggest_int("dense_units", 64, 128)
    activation = "tanh"

    ff_activation = trial.suggest_categorical("ff_activation", ["relu", "gelu"])
    pooling = "max"

    xg_activation = "linear"

    pos_dim = trial.suggest_int("pos_dim", 32, 80)

    optimizer_name = trial.suggest_categorical("optimizer", ["adam"])
    optimizer_type = Adam if optimizer_name == "adam" else RMSprop

    learning_rate = trial.suggest_float("learning_rate", 8e-5, 2e-4)

    batch_size = 256
    epochs = trial.suggest_int("epochs", 8, 16)

    # --- Windowing ---
    window_seconds = 40
    horizon_seconds = 12
    step_seconds = 1.0

    window = int(window_seconds / step_seconds)
    horizon = int(horizon_seconds / step_seconds)

    unique_games = training_data["game_id"].unique()
    train_games, val_games = train_test_split(unique_games, test_size=0.2, random_state=42)

    train_df = training_data[training_data["game_id"].isin(train_games)]
    val_df = training_data[training_data["game_id"].isin(val_games)]

    global_len = 80

    train_gen = WindowGenerator(
        df=train_df,
        window=window,
        horizon=horizon,
        step=1,
        batch_size=batch_size,
        global_len=global_len
    )

    val_gen = WindowGenerator(
        df=val_df,
        window=window,
        horizon=horizon,
        step=1,
        batch_size=batch_size,
        global_len=global_len
    )

    # Feature dimension
    first_game_id = next(iter(train_gen.games))
    feature_dim = train_gen.games[first_game_id][0].shape[1]

    model = build_dual_head_model(
        local_seq_len=window,
        global_seq_len=global_len,
        feature_dim=feature_dim,
        dropout1=dropout1,
        dense_units=dense_units,
        activation=activation,
        num_heads=num_heads,
        ff_dim=ff_dim,
        key_dim=key_dim,
        ff_activation=ff_activation,
        pooling=pooling,
        xg_activation=xg_activation,
        pos_dim=pos_dim
    )

    lr_schedule = WarmupCosine(
        base_lr=learning_rate,
        warmup_steps=1000,
        total_steps=10000
    )
    optimizer = optimizer_type(learning_rate=lr_schedule)

    loss = {
        "goal_prob": UncertaintyLoss(
            tf.keras.losses.BinaryCrossentropy(from_logits=False),
            name_prefix="goal_prob",
        ),
        "xg": UncertaintyLoss(
            tf.keras.losses.MeanSquaredError(),
            name_prefix="xg",
        ),
        "penalty_corner": UncertaintyLoss(
            tf.keras.losses.BinaryCrossentropy(from_logits=False),
            name_prefix="penalty_corner",
        ),
        "penalty_stroke": UncertaintyLoss(
            tf.keras.losses.BinaryCrossentropy(from_logits=False),
            name_prefix="penalty_stroke",
        ),
        "circle_entry": UncertaintyLoss(
            tf.keras.losses.BinaryCrossentropy(from_logits=False),
            name_prefix="circle_entry",
        ),
    }

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics={
            "xg": ["mse"]
        }
    )

    callbacks = [
        TFKerasPruningCallback(trial, "val_xg_mse"),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_xg_mse",
            patience=3,
            restore_best_weights=True,
            verbose=0
        ),
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks,
        validation_freq=1,
        verbose=0
    )

    final_mse = history.history["val_xg_mse"][-1]
    rmse = np.sqrt(final_mse)

    if trial.number == 0 or rmse < trial.study.best_value:
        model_name = "best_model"
        model.save(f"temp/optuna/{model_name}.keras")
        trial.set_user_attr("model_name", model_name)

    del model, history, train_gen, val_gen
    gc.collect()
    tf.keras.backend.clear_session()

    return rmse