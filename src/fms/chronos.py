from chronos import BaseChronosPipeline, Chronos2Pipeline
from tqdm import tqdm
import numpy as np
from src.metrics.metrics import compute_metrics

def evaluate(
    loader,
    config
):
    
    pipeline: Chronos2Pipeline = BaseChronosPipeline.from_pretrained("amazon/chronos-2", device_map="cuda")

    trues, preds = [], []

    target = config["target"]

    for X, y in tqdm(loader):

        # Chronos-2 has mandatory item-id. We just mock it with a 0.
        if 'item_id' not in X:
            X['item_id'] = 0

        # Index for Timeseries data with Chronos.
        X["index"] = X.index

        pred = pipeline.predict_df(
            X,
            prediction_length=config["forecast_horizon"],
            target=target,
            timestamp_column="index"
        )

        trues.append(np.array(y[target]))
        preds.append(np.array(pred["predictions"]))

    result = compute_metrics(np.array(trues), np.array(preds))

    return result
