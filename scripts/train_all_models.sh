while getopts "e:" opt
do
  case "$opt" in
    e ) EXPERIMENT_NAME="$OPTARG"
  esac
done
# Seq2Seq SASeq2Seq SAMSeq2Seq
for modelName in Seq2Seq SAMSeq2Seq 
do
  # Train only one model (multi parameter) and mult parameter model return sequences so only normal evaluation run.
  # mlflow run --experiment-name $EXPERIMENT_NAME . --env-manager=local \
  #   -P model_name=$modelName \
  #   -P scaling_method=min_max \
  #   -P weights_initializer=he \
  #   -P is_obpoint_labeldata=false \
  #   -P multi_parameter_model_return_sequences=true \
  #   -P 'input_parameters=rain/temperature/humidity' \
  #   -P train_is_max_datasize_limit=false \
  #   -P train_epochs=500 \
  #   -P train_separately=false
    
  if [[ $modelName == "Seq2Seq" ]] 
  then
    SAVE_ATTENTION_MAPS=false
  else
    SAVE_ATTENTION_MAPS=true
  fi
    
  # Train sepalately and multi parameter model does not return sequences but only return 1 step.
  # So, normal evaluation, sequential evaluation and combine models evalaution run.
  mlflow run --experiment-name $EXPERIMENT_NAME . --env-manager=local \
    -P model_name=$modelName \
    -P scaling_method=min_max \
    -P weights_initializer=he \
    -P is_obpoint_labeldata=false \
    -P multi_parameter_model_return_sequences=false \
    -P 'input_parameters=rain/temperature/humidity' \
    -P train_is_max_datasize_limit=false \
    -P train_epochs=500 \
    -P train_separately=false \
    -P evaluate_save_attention_maps=$SAVE_ATTENTION_MAPS

done