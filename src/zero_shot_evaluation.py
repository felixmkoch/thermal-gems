from src.data.get_loader import build_timeseries_dataloader_df
try:
    from src.fms.chronos import evaluate as evaluate_chronos
except Exception:
    pass

try:
    from src.fms.toto import evaluate as evaluate_toto
except Exception:
    pass

try:
    from src.fms.timesfm import evaluate as evaluate_timesfm
except Exception:
    pass

from src.time_series.time_series import evaluate as evaluate_time_series


def evaluate_zero_shot(
        model: str,
        config: dict,
        csv_path: str
):
    
    loader = build_timeseries_dataloader_df(
        csv_path=csv_path,
        config=config
    )

    if model == "chronos":
        results = evaluate_chronos(
            loader=loader,
            config=config
        )

    if model == "toto":
        results = evaluate_toto(
            loader=loader,
            config=config
        )

    if model == "timesfm":
        results = evaluate_timesfm(
            loader=loader,
            config=config
        )

    if model == "lstm" or model == "transformer":
        results = evaluate_time_series(
            loader=loader,
            config=config
        )
        

    
    return results