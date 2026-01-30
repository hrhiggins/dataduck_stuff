import numpy as np
import tensorflow as tf
import random

class WindowGenerator(tf.keras.utils.Sequence):
    def __init__(self, df, window, horizon, step, batch_size,
                 feature_col="features",
                 goal_col="goal_event",
                 pc_col="penalty_corner_event",
                 ps_col="penalty_stroke_event",
                 ce_col="circle_entry_event",
                 game_col="game_id"):

        self.window = window
        self.horizon = horizon
        self.step = step
        self.batch_size = batch_size

        # Store per-game arrays
        self.games = {}
        for game_id, game_df in df.groupby(game_col):
            features = np.stack(game_df[feature_col].to_list()).astype(np.float32)
            goals = game_df[goal_col].to_numpy(dtype=np.float32)
            pcs = game_df[pc_col].to_numpy(dtype=np.float32)
            pss = game_df[ps_col].to_numpy(dtype=np.float32)
            ces = game_df[ce_col].to_numpy(dtype=np.float32)

            self.games[game_id] = (features, goals, pcs, pss, ces)

        # Precompute all window start positions
        self.index_game = []
        self.index_start = []

        for game_id, (features, goals, pcs, pss, ces) in self.games.items():
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

        X_batch = []
        y_goal = []
        y_xg = []
        y_pc = []
        y_ps = []
        y_ce = []

        unique_games = np.unique(game_ids)
        for g in unique_games:
            mask = (game_ids == g)
            starts_g = starts[mask]

            features, goals, pcs, pss, ces = self.games[g]

            # Window slices
            X_g = features[
                starts_g[:, None] + np.arange(self.window)[None, :]
            ]

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

            X_batch.append(X_g)

        # Concatenate across games
        X_batch = np.concatenate(X_batch, axis=0)
        y_goal = np.concatenate(y_goal, axis=0)
        y_xg = np.concatenate(y_xg, axis=0)
        y_pc = np.concatenate(y_pc, axis=0)
        y_ps = np.concatenate(y_ps, axis=0)
        y_ce = np.concatenate(y_ce, axis=0)

        return (
            X_batch,
            {
                "goal_prob": y_goal,
                "xg": y_xg,
                "penalty_corner": y_pc,
                "penalty_stroke": y_ps,
                "circle_entry": y_ce,
            }
        )

    def on_epoch_end(self):
        np.random.shuffle(self.order)