import numpy as np
import tensorflow as tf
import random

class WindowGenerator(tf.keras.utils.Sequence):
    def __init__(self, df, window, horizon, step, batch_size,
                 feature_col="features", goal_col="goal_event", game_col="game_id"):

        self.window = window
        self.horizon = horizon
        self.step = step
        self.batch_size = batch_size

        # Store per-game arrays
        self.games = {}
        for game_id, game_df in df.groupby(game_col):
            features = np.stack(game_df[feature_col].to_list()).astype(np.float32)
            goals = game_df[goal_col].to_numpy(dtype=np.float32)
            self.games[game_id] = (features, goals)

        # Precompute all window start positions
        self.index_game = []
        self.index_start = []

        for game_id, (features, goals) in self.games.items():
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
        # Batch indices
        batch_ids = self.order[idx * self.batch_size : (idx + 1) * self.batch_size]

        # Resolve game IDs and start positions
        game_ids = self.index_game[batch_ids]
        starts = self.index_start[batch_ids]

        # Vectorised extraction
        X_batch = []
        y_goal = []
        y_xg = []

        # Group by game to slice efficiently
        unique_games = np.unique(game_ids)
        for g in unique_games:
            mask = (game_ids == g)
            starts_g = starts[mask]

            features, goals = self.games[g]

            # Window slices: shape (num_samples_g, window, feature_dim)
            X_g = features[
                starts_g[:, None] + np.arange(self.window)[None, :]
            ]

            # Future goals: shape (num_samples_g, horizon)
            future_g = goals[
                starts_g[:, None] + self.window + np.arange(self.horizon)[None, :]
            ]

            X_batch.append(X_g)
            y_goal.append((future_g.sum(axis=1) > 0).astype(np.float32))
            y_xg.append(future_g.sum(axis=1).astype(np.float32))

        # Concatenate across games
        X_batch = np.concatenate(X_batch, axis=0)
        y_goal = np.concatenate(y_goal, axis=0)
        y_xg = np.concatenate(y_xg, axis=0)

        return (
            X_batch,
            {
                "goal_prob": y_goal,
                "xg": y_xg,
            }
        )

    def on_epoch_end(self):
        np.random.shuffle(self.order)