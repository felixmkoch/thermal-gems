from tqdm import tqdm
import torch
import numpy as np
from src.metrics.metrics import compute_metrics

import timesfm

def evaluate(
    loader,
    config
):

    torch.set_float32_matmul_precision("high")

    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")

    model.compile(
        timesfm.ForecastConfig(
            max_context=config["lookback"],
            max_horizon=config["forecast_horizon"],
            normalize_inputs=True,
            use_continuous_quantile_head=True,
            force_flip_invariance=True,
            infer_is_positive=True,
            fix_quantile_crossing=True,
        )
    )

    trues, preds = [], []

    target = config["target"]
    features = config["features"]

    covariate_columns = [s for s in features if s != target]

    target_index = features.index(target)

    for X, y in tqdm(loader):

        inputs = [X[target]]      # Only multivariate, we did not get the multivariate support running.

        point_forecast, quantile_forecast = model.forecast(
            horizon=config["forecast_horizon"],
            inputs=inputs
        )

        forecast_target = point_forecast[target_index]
        trues.append(np.array(y[target]))
        preds.append(np.array(forecast_target))
        

    result = compute_metrics(np.array(trues), np.array(preds))

    return result
