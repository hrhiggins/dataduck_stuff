
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

