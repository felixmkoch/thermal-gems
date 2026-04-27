from tqdm import tqdm
import torch
import numpy as np
from src.metrics.metrics import compute_metrics
from toto.data.util.dataset import MaskedTimeseries
from toto.inference.forecaster import TotoForecaster
from toto.model.toto import Toto

def evaluate(
    loader,
    config
):

    toto = Toto.from_pretrained('Datadog/Toto-Open-Base-1.0')
    toto.to('cuda')  

    toto.compile() 

    forecaster = TotoForecaster(toto.model)

    trues, preds = [], []

    target = config["target"]
    features = config["features"]

    target_index = features.index(target)

    interval = 60 * 15 # 15 min intervals

    for X, y in tqdm(loader):

        array = X[features].values    # Here [f, s]
        array = array.T                             # Transpose to [s, f]

        input_series = torch.from_numpy(array).float().to("cuda")

        # Mock as described in othe Toto documentation
        timestamps = torch.arange(config["lookback"]).float()  # [0, 1, 2, ..., 95]
        timestamp_seconds = timestamps.unsqueeze(0).expand(len(features), config["lookback"])
        timestamp_seconds = timestamp_seconds.to("cuda")

        time_interval_seconds=torch.full((len(features),), interval).to("cuda")


        inputs = MaskedTimeseries(
            series=input_series,
            padding_mask=torch.full_like(input_series, True, dtype=torch.bool),
            id_mask=torch.zeros_like(input_series),
            timestamp_seconds=timestamp_seconds,
            time_interval_seconds=time_interval_seconds,
        )

        # Forecast has shape [1, f, s]
        forecast = forecaster.forecast(
            inputs,
            prediction_length=config["forecast_horizon"],
            num_samples=32,
            samples_per_batch=32,
            use_kv_cache=True,
        )

        # Only the first element and then the target column, because all features are forecasted.
        forecast_medians = forecast.median[0][target_index].to("cpu").numpy()

        trues.append(np.array(y[target]))
        preds.append(forecast_medians)
        

    result = compute_metrics(np.array(trues), np.array(preds))

    return result
