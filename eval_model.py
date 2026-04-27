import os
from helper.utils import writerow, combination_exists_dict
from src.time_series.time_series import evaluate
from src.data.get_loader import build_timeseries_dataloader_df

#-------------------------------------------------------
#   FOR THE USER
#-------------------------------------------------------

model = "transformer"  # "transformer", "lstm", "xlstm", "mamba"
dataset = "ecobee"  # "ecobee", "ideal", "euro"
device = "cuda"  # "cuda" or "cpu" - set to "cuda" to force GPU, "cpu" for CPU only

config = {
    "lookback": 96,
    "forecast_horizon": 4,
    "limit": 35040,
    "dataset": dataset,
    "model_name": model,
    "device": device,
}

#-------------------------------------------------------
#   LOGIC
#-------------------------------------------------------

dataset_folder = f"data/{dataset}"

if dataset == "ecobee":
    config["features"] = ["Indoor_AverageTemperature", "Indoor_Humidity", "HeatingEquipmentStage1_RunTime", "HeatingEquipmentStage2_RunTime", "HeatingEquipmentStage3_RunTime", "CoolingEquipmentStage1_RunTime", "CoolingEquipmentStage2_RunTime", "HeatPumpsStage1_RunTime", "HeatPumpsStage2_RunTime", "Fan_RunTime", "Thermostat_Temperature", "Thermostat_DetectedMotion", "Outdoor_Temperature", "HGloHor"]
    config["target"] = "Indoor_AverageTemperature"
elif dataset == "ideal":
    config["features"] = ["temperature", "radiator-input", "radiator-output", "outdoor-temperature", "solar-radiation", "windspeed", "humidity"]
    config["target"] = "temperature"
elif dataset == "euro":
    config["features"] = ["weaBusHDifHor", "weaBusHDirNor", "weaBusTDryBul",  "u", "thermalZoneTAir"]
    config["target"] = "thermalZoneTAir"

#-------------------------------------------------------
#   EXECUTION
#-------------------------------------------------------

result_csv_name = f"{dataset}_{model}_eval.csv"

csv_files = [f for f in sorted(os.listdir(dataset_folder)) if f.lower().endswith(".csv")]

for idx, file in enumerate(csv_files):
    file_name = file.replace(".csv", "")
    file_path = os.path.join(dataset_folder, file)

    print(f"At file #{idx + 1} of {len(csv_files)}")

    result_row = {
        "timeseries": file_name,
        "model": model
    }

    if combination_exists_dict(output_csv_name=result_csv_name, d=result_row):
        print(f"Already have the combination method {model} for file {file_name} in folder {dataset_folder} --> Skipping")
        continue

    loader = build_timeseries_dataloader_df(csv_path=file_path, config=config)
    result = evaluate(loader=loader, config=config)

    result_row.update(result)

    writerow(
        output_csv_name=result_csv_name,
        csv_cols=result_row.keys(),
        result=result_row
    )
