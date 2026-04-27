import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader


class WindowDatasetDF(Dataset):
    def __init__(self, df, lookback, forecast_horizon, features, target, **kwargs):
        self.df = df
        self.lookback = lookback
        self.horizon = forecast_horizon
        self.feat_cols = features
        self.target_col = target

        # Precompute valid start indices (windows without NaNs)
        self.valid_starts = []
        max_start = len(df) - lookback - forecast_horizon + 1
        for start in range(max_start):
            end = start + lookback
            horizon_end = end + forecast_horizon

            window_X = df.iloc[start:end][self.feat_cols]
            window_y = df.iloc[end:horizon_end][[self.target_col]]

            if not (window_X.isna().any().any() or window_y.isna().any().any()):
                self.valid_starts.append(start)

    def __len__(self):
        return len(self.valid_starts)

    def __getitem__(self, idx):
        start = self.valid_starts[idx]
        end = start + self.lookback
        horizon_end = end + self.horizon

        X_df = self.df.iloc[start:end][self.feat_cols].copy()
        y_df = self.df.iloc[end:horizon_end][[self.target_col]].copy()

        return X_df, y_df


def build_timeseries_dataloader_df(csv_path, config):
    print(f"Building DataLoader from {csv_path} with config: {config}")
    df = pd.read_csv(csv_path)

    dataset = WindowDatasetDF(
        df=df,
        **config
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda batch: batch[0]  # remove outer batch list → return (X_df, y_df)
    )

    return loader
