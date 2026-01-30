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

        self.games = {}
        for game_id, game_df in df.groupby(game_col):
            features = np.stack(game_df[feature_col].to_list()).astype(np.float32)
            goals = game_df[goal_col].to_numpy(dtype=np.float32)
            self.games[game_id] = (features, goals)

        self.game_ids = list(self.games.keys())  # order will be shuffled
        self.windows_per_game = {}

        for game_id, (features, goals) in self.games.items():
            T = len(features)
            max_start = T - window - horizon
            if max_start <= 0:
                self.windows_per_game[game_id] = []
                continue

            starts = list(range(0, max_start, step))
            self.windows_per_game[game_id] = starts  # keep order, no shuffle

        # Build a flat index of (game_id, start)
        self._rebuild_index()

    def _rebuild_index(self):

        self.index = []
        for game_id in self.game_ids:
            for start in self.windows_per_game[game_id]:
                self.index.append((game_id, start))

    def __len__(self):
        return len(self.index) // self.batch_size

    def __getitem__(self, idx):
        batch_idx = self.index[idx * self.batch_size : (idx + 1) * self.batch_size]

        X_batch = []
        y_goal_batch = []
        y_xg_batch = []

        for game_id, start in batch_idx:
            features, goals = self.games[game_id]

            window_feats = features[start : start + self.window]
            future_goals = goals[start + self.window : start + self.window + self.horizon]

            X_batch.append(window_feats)
            y_goal_batch.append(1.0 if np.any(future_goals) else 0.0)
            y_xg_batch.append(np.sum(future_goals))

        return (
            np.array(X_batch, dtype=np.float32),
            {
                "goal_prob": np.array(y_goal_batch, dtype=np.float32),
                "xg": np.array(y_xg_batch, dtype=np.float32),
            }
        )

    def on_epoch_end(self):
        random.shuffle(self.game_ids)   # shuffle games
        self._rebuild_index()           # rebuild index in new game order