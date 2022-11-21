import pandas as pd
from common.utils import timestep_csv_names
import json


def main():
    data = {
        "2020-10-01": {"start": "10-30.csv", "end": "16-30.csv"},
        "2020-10-02": {"start": "15-30.csv", "end": "18-0.csv"},
        "2020-09-19": {"start": "5-30.csv", "end": "7-30.csv"},
    }

    _timestep_csv_names = timestep_csv_names(time_step_minutes=30)
    valid_dataset = {}
    valid_dataset["TC_case"] = {}
    for sample_idx, date in enumerate(list(data.keys())):
        valid_dataset["TC_case"][f"sample{sample_idx}"] = {}
        start = data[date]["start"]
        end = data[date]["end"]
        start_idx, end_idx = _timestep_csv_names.index(start), _timestep_csv_names.index(end)
        for key, val in enumerate(_timestep_csv_names[start_idx : end_idx + 1]):
            valid_dataset["TC_case"][f"sample{sample_idx}"][str(key)] = {"date": date, "start": val}

    with open("./preprocess/src/less_than_40mmh_testdataset.json", "w") as f:
        json.dump(valid_dataset, f)


if __name__ == "__main__":
    main()
    # observation_point_file_path = "./common/meta-data/observation_point.json"
    # df = pd.read_csv("../data/rain_image/2020/10/2020-10-12/7-0.csv", index_col=0)
    # save_rain_image(df.to_numpy(), observation_point_file_path, "./test.png")
