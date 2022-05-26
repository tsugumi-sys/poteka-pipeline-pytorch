#!/bin/bash
EXPERIMENT_NAME=feature_swap

PREPROCESS_PARAMS1="rain temperature humidity wind abs_wind station_pressure seaLevel_pressure"
PREPROCESS_PARAMS2="rain humidity wind abs_wind station_pressure seaLevel_pressure"
PREPROCESS_PARAMS3="rain temperature wind abs_wind station_pressure seaLevel_pressure"
PREPROCESS_PARAMS4="rain temperature humidity abs_wind station_pressure seaLevel_pressure"
PREPROCESS_PARAMS5="rain temperature humidity wind station_pressure seaLevel_pressure"
PREPROCESS_PARAMS6="rain temperature humidity wind abs_wind seaLevel_pressure"
PREPROCESS_PARAMS7="rain temperature humidity wind abs_wind station_pressure"


mlflow run --experiment-name $EXPERIMENT_NAME . --no-conda -P preprocess_params="$PREPROCESS_PARAMS1"
mlflow run --experiment-name $EXPERIMENT_NAME . --no-conda -P preprocess_params="$PREPROCESS_PARAMS2"
mlflow run --experiment-name $EXPERIMENT_NAME . --no-conda -P preprocess_params="$PREPROCESS_PARAMS3"
mlflow run --experiment-name $EXPERIMENT_NAME . --no-conda -P preprocess_params="$PREPROCESS_PARAMS4"
mlflow run --experiment-name $EXPERIMENT_NAME . --no-conda -P preprocess_params="$PREPROCESS_PARAMS5"
mlflow run --experiment-name $EXPERIMENT_NAME . --no-conda -P preprocess_params="$PREPROCESS_PARAMS6"
mlflow run --experiment-name $EXPERIMENT_NAME . --no-conda -P preprocess_params="$PREPROCESS_PARAMS7"