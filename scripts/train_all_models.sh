while getopts "e:" opt
do
  case "$opt" in
    e ) EXPERIMENT_NAME="$OPTARG"
  esac
done

for modelName in Seq2Seq SASeq2Seq SAMSeq2Seq
do
  mlflow run --experiment-name $EXPERIMENT_NAME . --env-manager=local \
    -P model_name=$modelName \
    -P scaling_method=min_max \
    -P weights_initializer=he \
    -P is_obpoint_labeldata=false \
    -P multi_parameter_model_return_sequences=true \
    -P 'input_parameters=rain/temperature/humidity' \
    -P train_is_max_datasize_limit=false \
    -P train_epochs=500 \
    -P train_separately=false
    
  mlflow run --experiment-name $EXPERIMENT_NAME . --env-manager=local \
    -P model_name=$modelName \
    -P scaling_method=min_max \
    -P weights_initializer=he \
    -P is_obpoint_labeldata=false \
    -P multi_parameter_model_return_sequences=false \
    -P 'input_parameters=rain/temperature/humidity' \
    -P train_is_max_datasize_limit=false \
    -P train_epochs=500 \
    -P train_separately=true
done