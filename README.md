# Thermal-GEMs: Generalized Models for Building Thermal Dynamics

This repository provides the code to reproduce the results of the paper **Thermal-GEMs: Generalized Models for Building Thermal Dynamics**.

## Datasets

The benchmark uses three building thermal dynamics datasets:

- **Ecobee**: Residential thermostat data with 14 features including temperature, humidity, and HVAC runtime measurements
- **IDEAL**: Residential building simulation data with 7 features including temperature, radiator states, and weather variables
- **EURO**: Commercial building simulation data with 5 features including thermal zone temperature and control signals

## Models

### Foundation Models
- [Toto](https://github.com/DataDog/toto): General-purpose time series foundation model
- [Chronos-2](https://github.com/amazon-science/chronos-forecasting): Pretrained forecasting model
- [TimesFM-2.5](https://github.com/google-research/timesfm): Google's time series foundation model

### Specialized Models
- **LSTM**: Long Short-Term Memory network
- **Transformer**: Multi-head attention-based architecture
- **xLSTM**: Extended LSTM with memory mixing
- **Mamba**: State-space model for efficient sequence processing

## Setup

### Environment

Due to conflicting dependencies between foundation models, each model requires its own conda environment (you can use other environments as well):

```bash
# Create base environment
conda create -n thermal-benchmark python=3.10
conda activate thermal-benchmark
```

### Foundation Model Setup

Foundation models require separate Docker environments due to dependency conflicts:

1. **Toto**: Use `Dockerfile_Toto`
2. **Chronos-2**: Use `Dockerfile_Chronos`
3. **TimesFM**: Use `Dockerfile_TimesFM` (requires manual installation)

### Specialized Model Setup

For LSTM, Transformer, xLSTM, and Mamba models, use the provided environment file:

```bash
pip install -r requirements.txt
```

The environment includes all necessary dependencies for training and evaluating specialized time series models.

## Usage

### Training Specialized Models

The training code for specialized models is available in the companion repository [thermal-training](https://github.com/your-org/thermal-training).

### Evaluation

To evaluate pretrained models:

1. Set the model and dataset in `eval_model.py`:
   ```python
   model = "transformer"  # Options: "lstm", "transformer", "xlstm", "mamba"
   dataset = "ecobee"     # Options: "ecobee", "ideal", "euro"
   device = "cuda"        # Options: "cuda", "cpu"
   ```

2. Run evaluation:
   ```bash
   python eval_model.py
   ```

Results are saved to `{dataset}_{model}_eval.csv`.

### Foundation Model Evaluation

For foundation models, use the provided Docker containers:

```bash
# Example for Toto
docker build -f Dockerfile_Toto -t thermal-toto .
docker run -v $(pwd):/workspace thermal-toto python eval_model.py
```

## Results

Evaluation results include standard time series forecasting metrics (MSE, RMSE, MAE) computed across all test sequences for each dataset-model combination.

## Citation

If you use this code in your research, please cite:

```
@article{koch2026thermal,
  title={Thermal-GEMs: Generalized Models for Building Thermal Dynamics},
  author={Koch, Felix and Raisch, Fabian and Tischler, Benjamin},
  booktitle={Proceedings of the 13th ACM International Conference on Systems for Energy-Efficient Buildings, Cities, and Transportation}
  year={2026}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
