while getopts "e:" opt
do
  case "$opt" in
    e ) EXPERIMENT_NAME="$OPTARG"
  esac
done

for modelName in SASeq2Seq SAMSeq2Seq
do
  if [[ $modelName == "Seq2Seq" ]] 
  then
    SAVE_ATTENTION_MAPS=false
  else
    SAVE_ATTENTION_MAPS=true
  fi

  mlflow run --experiment-name $EXPERIMENT_NAME . --env-manager=local \
    -P model_name=$modelName \
    -P scaling_method=min_max \
    -P weights_initializer=he \
    -P is_obpoint_labeldata=false \
    -P multi_parameter_model_return_sequences=true \
    -P 'input_parameters=rain/humidity' \
    -P train_is_max_datasize_limit=true \
    -P train_epochs=5 \
    -P train_separately=false \
    -P evaluate_save_attention_maps=$SAVE_ATTENTION_MAPS

  mlflow run --experiment-name $EXPERIMENT_NAME . --env-manager=local \
    -P model_name=$modelName \
    -P scaling_method=min_max \
    -P weights_initializer=he \
    -P is_obpoint_labeldata=false \
    -P multi_parameter_model_return_sequences=false \
    -P 'input_parameters=rain/humidity' \
    -P train_is_max_datasize_limit=true \
    -P train_epochs=5 \
    -P train_separately=true \
    -P evaluate_save_attention_maps=$SAVE_ATTENTION_MAPS
done
