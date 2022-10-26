import numpy as np
import pandas as pd
import torch
import os
from common.data_loader import test_data_loader
import json
from common.interpolate_by_gpr import interpolate_by_gpr
from evaluate.src.create_image import save_rain_image
from train.src.config import DEVICE
from train.src.obpoint_seq_to_seq import OBPointSeq2Seq


def main():
    test_data_paths = os.path.join("./data/preprocess", "meta_test.json")
    observation_point_file_path = "./common/meta-data/observation_point.json"
    # NOTE: test_data_loader loads all parameters tensor. So num_channels are maximum.
    test_dataset, features_dict = test_data_loader(
        test_data_paths,
        observation_point_file_path,
        scaling_method="min_max",
        use_dummy_data=False,
    )

    # NOTE: all parameter trained model (model) should be evaluate in the end so that combine models prediction can be executed.
    trained_model = torch.load(os.path.join("./data/train", "model.pth"))
    model = OBPointSeq2Seq(
       num_channels=trained_model["num_channels"],
       ob_point_count=trained_model["ob_point_count"],
       kernel_size=trained_model["kernel_size"],
       num_kernels=trained_model["num_kernels"],
       padding=trained_model["padding"],
       activation=trained_model["activation"],
       frame_size=trained_model["frame_size"],
       num_layers=trained_model["num_layers"],
       input_seq_length=trained_model["input_seq_length"],
       prediction_seq_length=trained_model["prediction_seq_length"],
       weights_initializer=trained_model["weights_initializer"],
       return_sequences=False,
       ) 
    model.load_state_dict(trained_model["model_state_dict"])
    model.to(DEVICE)
    model.float()

    test_case_name = "TC_case_2020-10-12_7-0_start"
    sample_test_dataset = test_dataset[test_case_name]

    pred = model(sample_test_dataset["input"].to(DEVICE))

    pred_rain = pred[0, 0, 0, ...] * 100
    with open(observation_point_file_path, "r") as f:
        ob_point_data = json.load(f)

    pred_df = pd.DataFrame(pred_rain.cpu().detach().numpy().copy(), index=list(ob_point_data.keys()), columns=["Pred_Value"])
    label_df = sample_test_dataset["label_df"][0]
    # label_df["Pred_Value"] = pred_rain.cpu().detach().numpy().copy()
    label_df = label_df.merge(pred_df, left_index=True, right_index=True)
    print(label_df[["hour-rain", "Pred_Value"]])

    interp_pred = interpolate_by_gpr(pred_df["Pred_Value"].to_numpy(), observation_point_file_path)
    pd.DataFrame(interp_pred).to_csv("./interp_pred.csv")
    save_rain_image(interp_pred, observation_point_file_path, "./pred.png")

    #save_rain_image(pred_df["Pred_Value"].to_numpy(), observation_point_file_path, "./pred.png")
    


if __name__ == "__main__":
    main()
