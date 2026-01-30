import numpy as np
import tensorflow as tf

class WindowGenerator(tf.keras.utils.Sequence):
    def __init__(self, df, window, horizon, step, batch_size,
                 feature_col="features", goal_col="goal_event", game_col="game_id"):

        self.window = window
        self.horizon = horizon
        self.step = step
        self.batch_size = batch_size

        # -----------------------------
        # CACHE PER-GAME DATA (FAST)
        # -----------------------------
        self.games = {}
        for game_id, game_df in df.groupby(game_col):
            # Convert features column (list of arrays) → (T, feature_dim) NumPy array
            features = np.stack(game_df[feature_col].to_list()).astype(np.float32)

            # Convert goals → NumPy array
            goals = game_df[goal_col].to_numpy(dtype=np.float32)

            self.games[game_id] = (features, goals)

        # -----------------------------
        # PRECOMPUTE ALL WINDOW STARTS
        # -----------------------------
        self.index = []
        for game_id, (features, goals) in self.games.items():
            T = len(features)
            max_start = T - window - horizon
            if max_start <= 0:
                continue

            for start in range(0, max_start, step):
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

            # FAST: pure NumPy slicing
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