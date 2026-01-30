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


# ============================================================
# WindowGenerator: local + global inputs
# ============================================================
import numpy as np
import tensorflow as tf

class WindowGenerator(tf.keras.utils.Sequence):
    def __init__(
        self,
        df,
        window,
        horizon,
        step,
        batch_size,
        global_len=120,
        feature_col="features",
        goal_col="goal_event",
        pc_col="penalty_corner_event",
        ps_col="penalty_stroke_event",
        ce_col="circle_entry_event",
        game_col="game_id"
    ):
        super().__init__()  # important so Keras treats this as a Sequence

        self.window = window
        self.horizon = horizon
        self.step = step
        self.batch_size = batch_size
        self.global_len = global_len

        # Store per-game arrays
        self.games = {}
        for game_id, game_df in df.groupby(game_col):
            features = np.stack(game_df[feature_col].to_list()).astype(np.float32)
            goals = game_df[goal_col].to_numpy(dtype=np.float32)
            pcs = game_df[pc_col].to_numpy(dtype=np.float32)
            pss = game_df[ps_col].to_numpy(dtype=np.float32)
            ces = game_df[ce_col].to_numpy(dtype=np.float32)

            T = len(features)
            if T < 2:
                continue

            # Global sequence (downsampled over full game)
            idx = np.linspace(0, T - 1, self.global_len).astype(int)
            global_features = features[idx]

            self.games[game_id] = (
                features,
                goals,
                pcs,
                pss,
                ces,
                global_features,
            )

        # Precompute all window start positions
        self.index_game = []
        self.index_start = []

        for game_id, (features, goals, pcs, pss, ces, global_features) in self.games.items():
            T = len(features)
            max_start = T - window - horizon
            if max_start <= 0:
                continue

            starts = np.arange(0, max_start, step, dtype=np.int32)
            self.index_game.append(np.full_like(starts, game_id))
            self.index_start.append(starts)

        # Flatten
        if self.index_game:
            self.index_game = np.concatenate(self.index_game)
            self.index_start = np.concatenate(self.index_start)
        else:
            self.index_game = np.array([], dtype=np.int32)
            self.index_start = np.array([], dtype=np.int32)

        # Shuffle order
        self.order = np.arange(len(self.index_game))

    def __len__(self):
        return len(self.order) // self.batch_size

    def __getitem__(self, idx):
        batch_ids = self.order[idx * self.batch_size : (idx + 1) * self.batch_size]

        game_ids = self.index_game[batch_ids]
        starts = self.index_start[batch_ids]

        local_batch = []
        global_batch = []

        y_goal = []
        y_xg = []
        y_pc = []
        y_ps = []
        y_ce = []

        unique_games = np.unique(game_ids)

        for g in unique_games:
            mask = (game_ids == g)
            starts_g = starts[mask]

            features, goals, pcs, pss, ces, global_features = self.games[g]

            # Local window slices
            X_local = features[
                starts_g[:, None] + np.arange(self.window)[None, :]
            ]

            # Global sequence (same for all samples from this game)
            X_global = np.repeat(global_features[None, :, :], len(starts_g), axis=0)

            # Future slices
            future_g = goals[
                starts_g[:, None] + self.window + np.arange(self.horizon)[None, :]
            ]
            future_pc = pcs[
                starts_g[:, None] + self.window + np.arange(self.horizon)[None, :]
            ]
            future_ps = pss[
                starts_g[:, None] + self.window + np.arange(self.horizon)[None, :]
            ]
            future_ce = ces[
                starts_g[:, None] + self.window + np.arange(self.horizon)[None, :]
            ]

            # Labels: event occurs anywhere in horizon
            y_goal.append((future_g.sum(axis=1) > 0).astype(np.float32))
            y_xg.append((future_g.sum(axis=1) > 0).astype(np.float32))
            y_pc.append((future_pc.sum(axis=1) > 0).astype(np.float32))
            y_ps.append((future_ps.sum(axis=1) > 0).astype(np.float32))
            y_ce.append((future_ce.sum(axis=1) > 0).astype(np.float32))

            local_batch.append(X_local)
            global_batch.append(X_global)

        # Concatenate across games
        local_batch = np.concatenate(local_batch, axis=0)
        global_batch = np.concatenate(global_batch, axis=0)

        y_goal = np.concatenate(y_goal, axis=0)
        y_xg = np.concatenate(y_xg, axis=0)
        y_pc = np.concatenate(y_pc, axis=0)
        y_ps = np.concatenate(y_ps, axis=0)
        y_ce = np.concatenate(y_ce, axis=0)

        # IMPORTANT: return dict of inputs, not a list
        inputs = {
            "local_input": local_batch,
            "global_input": global_batch,
        }

        targets = {
            "goal_prob": y_goal,
            "xg": y_xg,
            "penalty_corner": y_pc,
            "penalty_stroke": y_ps,
            "circle_entry": y_ce,
        }

        return inputs, targets

    def on_epoch_end(self):
        np.random.shuffle(self.order)


# ============================================================
# Warmup + Cosine LR schedule
# ============================================================

class WarmupCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_lr, warmup_steps=2000, total_steps=20000):
        super().__init__()
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.cosine = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=base_lr,
            decay_steps=total_steps - warmup_steps,
            alpha=0.1
        )

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        return tf.cond(
            step < self.warmup_steps,
            lambda: self.base_lr * (step / self.warmup_steps),
            lambda: self.cosine(step - self.warmup_steps)
        )

    def get_config(self):
        return {
            "base_lr": self.base_lr,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
        }


# ============================================================
# Transformer encoder block (with residual dropout)
# ============================================================

def transformer_encoder(x, num_heads, ff_dim, key_dim, dropout, ff_activation):
    # Multi-head attention
    attn_output = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=key_dim,
        dropout=dropout
    )(x, x)

    x = Add()([x, attn_output])
    x = LayerNormalization(epsilon=1e-6)(x)

    # Feed-forward network with residual dropout
    ff_output = Dense(ff_dim, activation=ff_activation)(x)
    ff_output = Dropout(dropout)(ff_output)
    ff_output = Dense(x.shape[-1])(ff_output)
    ff_output = Dropout(dropout)(ff_output)

    x = Add()([x, ff_output])
    x = LayerNormalization(epsilon=1e-6)(x)

    return x


# ============================================================
# Dual-stream model: local + global transformers
# ============================================================

def build_dual_head_model(
    local_seq_len,
    global_seq_len,
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
    pos_dim,
    penalty_corner_weight,
    penalty_stroke_weight,
    circle_entry_weight
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
    local_pos = tf.expand_dims(local_pos, axis=0)
    local_x = local_inputs + local_pos

    # Global
    global_pos = Embedding(
        input_dim=global_seq_len,
        output_dim=pos_dim
    )(tf.range(global_seq_len))
    global_pos = Dense(feature_dim)(global_pos)
    global_pos = tf.expand_dims(global_pos, axis=0)
    global_x = global_inputs + global_pos

    # ----- Local transformer blocks -----
    for _ in range(3):
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
    for _ in range(3):
        global_x = transformer_encoder(
            global_x,
            num_heads=num_heads,
            ff_dim=ff_dim,
            key_dim=key_dim,
            dropout=dropout1,
            ff_activation=ff_activation
        )
        global_x = Dropout(dropout1)(global_x)

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

    # ----- Warmup + Cosine Decay -----
    lr_schedule = WarmupCosine(
        base_lr=learning_rate,
        warmup_steps=2000,
        total_steps=20000
    )

    model = Model(
        inputs=[local_inputs, global_inputs],
        outputs=[goal_prob, xg, penalty_corner, penalty_stroke, circle_entry]
    )

    model.compile(
        optimizer=optimizer_type(learning_rate=lr_schedule),
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


# ============================================================
# Optuna objective with tuned hyperparameters
# ============================================================

def objective(trial, training_data):
    tf.keras.backend.clear_session()

    # --- Hyperparameters ---
    num_heads = trial.suggest_categorical("num_heads", [2, 4])
    key_dim = trial.suggest_int("key_dim", 16, 48)
    ff_dim = trial.suggest_int("ff_dim", 32, 96)

    dropout1 = trial.suggest_float("dropout1", 0.05, 0.25)

    dense_units = trial.suggest_int("dense_units", 32, 128)
    activation = "tanh"

    ff_activation = trial.suggest_categorical("ff_activation", ["relu", "gelu"])
    pooling = "max"

    xg_weight = trial.suggest_float("xg_weight", 0.1, 0.7)
    xg_activation = "linear"

    pos_dim = trial.suggest_int("pos_dim", 16, 128)

    optimizer_name = trial.suggest_categorical("optimizer", ["adam"])
    optimizer_type = Adam if optimizer_name == "adam" else RMSprop

    learning_rate = trial.suggest_float("learning_rate", 8e-5, 2e-4)

    batch_size = 512
    epochs = trial.suggest_int("epochs", 10, 20)

    penalty_corner_weight = trial.suggest_float("penalty_corner_weight", 0.2, 1.0)
    penalty_stroke_weight = trial.suggest_float("penalty_stroke_weight", 0.5, 2.5)
    circle_entry_weight = trial.suggest_float("circle_entry_weight", 0.05, 0.4)

    # --- Windowing ---
    window_seconds = 60
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
        batch_size=batch_size,
        global_len=120
    )

    val_gen = WindowGenerator(
        df=val_df,
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

    # --- Build model ---
    model = build_dual_head_model(
        local_seq_len=window,
        global_seq_len=global_len,
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
        validation_freq=1,
        verbose=0
    )

    final_mse = history.history["val_xg_mse"][-1]
    rmse = np.sqrt(final_mse)

    # Save best model (weights only to avoid LR schedule serialization issues)
    if trial.number == 0 or rmse < trial.study.best_value:
        model_name = "best_model"
        model.save_weights(f"temp/optuna/temp/trial_saves/{model_name}.weights.h5")
        trial.set_user_attr("model_name", model_name)

    del model, history, train_gen, val_gen
    gc.collect()
    tf.keras.backend.clear_session()

    return rmse