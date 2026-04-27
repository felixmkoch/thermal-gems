import os
import joblib
import torch
import numpy as np
from src.time_series.lstm import LSTM
from src.time_series.transformer import Transformer

try:
    from src.time_series.xlstm import XLSTM
except ImportError:
    XLSTM = None

try:
    from src.time_series.mamba import Mamba
except ImportError:
    Mamba = None

from src.metrics.metrics import compute_metrics


def evaluate(loader, config):
    config = config.copy()

    model_name = config.get("model_name") or config.get("model", {}).get("name")
    if not model_name:
        raise ValueError("config['model_name'] or config['model']['name'] is required.")

    dataset = config.get("dataset")
    if not dataset:
        raise ValueError("config['dataset'] is required.")

    model_name = model_name.lower()
    model_path = config.get("model_path") or get_local_model_path(dataset, model_name)
    feature_scaler_path, target_scaler_path = get_local_scaler_paths(dataset, model_name)

    model_dict = load_model(model_path, device=config.get("device"))
    feature_scaler = load_scaler(feature_scaler_path)
    target_scaler = load_scaler(target_scaler_path)

    model = model_dict["model"]
    model.eval()

    device = next(model.parameters()).device

    trues, preds = [], []

    with torch.no_grad():
        for X_df, y_df in loader:
            X_np = feature_scaler.transform(X_df.astype(np.float32))
            X_t = torch.tensor(X_np, dtype=torch.float32).unsqueeze(0).to(device)

            pred_scaled = model(X_t).squeeze(0).detach().cpu().numpy()
            pred = target_scaler.inverse_transform(pred_scaled)

            preds.append(pred.astype(np.float32).squeeze())
            trues.append(y_df.values.astype(np.float32).squeeze())

    result = compute_metrics(np.array(trues), np.array(preds))
    return result


def get_local_model_path(dataset, model_name):
    model_basename = f"{dataset}_{model_name}_gm"
    path = os.path.join("models", dataset, model_name, f"{model_basename}.pt")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Model checkpoint not found at: {path}")
    return path


def get_local_scaler_paths(dataset, model_name):
    model_basename = f"{dataset}_{model_name}_gm"
    model_dir = os.path.join("models", dataset, model_name)

    feature_scaler_path = os.path.join(model_dir, f"feature_scaler_{model_basename}.gz")
    target_scaler_path = os.path.join(model_dir, f"target_scaler_{model_basename}.gz")

    if not os.path.isfile(feature_scaler_path):
        raise FileNotFoundError(f"Feature scaler not found at: {feature_scaler_path}")
    if not os.path.isfile(target_scaler_path):
        raise FileNotFoundError(f"Target scaler not found at: {target_scaler_path}")

    return feature_scaler_path, target_scaler_path


def load_scaler(path):
    return joblib.load(path)


def load_model(path: str, device="cuda"):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if isinstance(device, str):
        if device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available but device='cuda' was requested. Please set device='cpu' or install CUDA.")
            device = torch.device("cuda")
        else:
            device = torch.device(device)

    from torch.optim.adam import Adam

    with torch.serialization.safe_globals([Adam]):
        checkpoint = torch.load(path, map_location=device, weights_only=False)

    cfg = checkpoint.get("config")
    if cfg is None:
        raise ValueError("Checkpoint has no config.")

    model_name = cfg["model"].get("name")
    if not model_name:
        raise ValueError("Model name missing in checkpoint config.")

    model_name = model_name.lower()
    check_model_params_exist(model_name, cfg["model"])
    model_class = get_model_from_string(model_name)
    if model_class is None:
        raise ValueError(f"Unsupported model type: {model_name}")

    model = model_class(
        num_features=len(cfg["data"]["feature_cols"]),
        seq_len=cfg["data"]["seq_len"],
        num_targets=len(cfg["data"]["target_cols"]),
        **cfg["model"]
    )

    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    return {
        "model": model,
        "config": cfg,
    }


def check_model_params_exist(s: str, d: dict):
    s = s.lower()
    model_keys = d.keys()

    if s == "lstm":
        required = ["forecast_horizon", "num_layers", "hidden_size", "optimizer", "lr", "dropout"]
    elif s == "transformer":
        required = ["forecast_horizon", "d_model", "n_head", "num_layers", "optimizer", "hidden_size", "dropout", "layer_norm_epsilon", "lr"]
    elif s == "xlstm":
        required = ["forecast_horizon", "embedding_dim", "context_length", "num_blocks", "num_heads"]
    elif s == "mamba":
        required = ["forecast_horizon", "d_model", "nhid", "num_layers"]
    else:
        required = []

    for key in required:
        if key not in model_keys:
            raise ValueError(f"Parameter for a new {s} model missing: {key}")


def get_model_from_string(s: str = ""):
    s = s.lower()
    if s == "lstm":
        return LSTM
    if s == "transformer":
        return Transformer
    if s == "xlstm":
        if XLSTM is None:
            raise ImportError("xlstm module is not available. Please install xlstm to use this model.")
        return XLSTM
    if s == "mamba":
        if Mamba is None:
            raise ImportError("mamba module is not available. Please install mamba-ssm to use this model.")
        return Mamba
    return None
