#!/bin/bash
EXPERIMENT_NAME=important_feats_standard

PREPROCESS_PARAMS1="rain"
PREPROCESS_PARAMS2="rain temperature"
PREPROCESS_PARAMS3="rain temperature humidity"

PREPROCESS_PARAMS4="rain temperature wind"
PREPROCESS_PARAMS5="rain temperature abs_wind"

PREPROCESS_PARAMS6="rain temperature humidity wind"
PREPROCESS_PARAMS7="rain temperature humidity wind abs_wind"
PREPROCESS_PARAMS8="rain temperature humidity abs_wind"
PREPROCESS_PARAMS9="rain temperature wind abs_wind"


PREPROCESS_PARAMS10="rain humidity"
PREPROCESS_PARAMS11="rain humidity wind"
PREPROCESS_PARAMS12="rain humidity wind abs_wind"
PREPROCESS_PARAMS13="rain humidity abs_wind"
PREPROCESS_PARAMS14="rain wind"
PREPROCESS_PARAMS15="rain wind abs_wind"
PREPROCESS_PARAMS16="rain abs_wind"


mlflow run --experiment-name $EXPERIMENT_NAME . --no-conda -P preprocess_params="$PREPROCESS_PARAMS1"
mlflow run --experiment-name $EXPERIMENT_NAME . --no-conda -P preprocess_params="$PREPROCESS_PARAMS2"
mlflow run --experiment-name $EXPERIMENT_NAME . --no-conda -P preprocess_params="$PREPROCESS_PARAMS3"
mlflow run --experiment-name $EXPERIMENT_NAME . --no-conda -P preprocess_params="$PREPROCESS_PARAMS4"
mlflow run --experiment-name $EXPERIMENT_NAME . --no-conda -P preprocess_params="$PREPROCESS_PARAMS5"
mlflow run --experiment-name $EXPERIMENT_NAME . --no-conda -P preprocess_params="$PREPROCESS_PARAMS6"
mlflow run --experiment-name $EXPERIMENT_NAME . --no-conda -P preprocess_params="$PREPROCESS_PARAMS7"
mlflow run --experiment-name $EXPERIMENT_NAME . --no-conda -P preprocess_params="$PREPROCESS_PARAMS8"
mlflow run --experiment-name $EXPERIMENT_NAME . --no-conda -P preprocess_params="$PREPROCESS_PARAMS9"
mlflow run --experiment-name $EXPERIMENT_NAME . --no-conda -P preprocess_params="$PREPROCESS_PARAMS10"
mlflow run --experiment-name $EXPERIMENT_NAME . --no-conda -P preprocess_params="$PREPROCESS_PARAMS11"
mlflow run --experiment-name $EXPERIMENT_NAME . --no-conda -P preprocess_params="$PREPROCESS_PARAMS12"
mlflow run --experiment-name $EXPERIMENT_NAME . --no-conda -P preprocess_params="$PREPROCESS_PARAMS13"
mlflow run --experiment-name $EXPERIMENT_NAME . --no-conda -P preprocess_params="$PREPROCESS_PARAMS14"
