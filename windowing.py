import numpy as np
import tensorflow as tf

class WindowGenerator(tf.keras.utils.Sequence):
    def __init__(self, df, window, horizon, step, batch_size, feature_col="features", goal_col="goal_event", game_col="game_id"):
        self.df = df
        self.window = window
        self.horizon = horizon
        self.step = step
        self.batch_size = batch_size
        self.feature_col = feature_col
        self.goal_col = goal_col
        self.game_col = game_col

        # Pre-group by game to avoid repeated work
        self.games = list(df.groupby(game_col))

        # Precompute total number of windows
        self.index = []
        for game_id, game_df in self.games:
            num_steps = len(game_df)
            for i in range(0, num_steps - window - horizon, step):
                self.index.append((game_id, i))

    def __len__(self):
        return len(self.index) // self.batch_size

    def __getitem__(self, idx):
        print(f"Building batch {idx}", flush=True)

        batch_idx = self.index[idx * self.batch_size : (idx + 1) * self.batch_size]

        X_batch = []
        y_goal_batch = []
        y_xg_batch = []

        for game_id, start in batch_idx:
            game_df = dict(self.games)[game_id]

            window_feats = game_df[self.feature_col].iloc[start : start + self.window]
            future_goals = game_df[self.goal_col].iloc[start + self.window : start + self.window + self.horizon]

            X_batch.append(np.stack(window_feats))
            y_goal_batch.append(1 if any(future_goals) else 0)
            y_xg_batch.append(sum(future_goals))

        return (
            np.array(X_batch, dtype=np.float32),
            {
                "goal_prob": np.array(y_goal_batch, dtype=np.float32),
                "xg": np.array(y_xg_batch, dtype=np.float32),
            }
        )