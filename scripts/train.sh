while getopts "e:m:" opt
do
  case "$opt" in
    e ) EXPERIMENT_NAME="$OPTARG";;
    m ) MODEL_NAME="$OPTARG";;
  esac
done


if [[ $modelName == "Seq2Seq" ]] 
then
  SAVE_ATTENTION_MAPS=false
else
  SAVE_ATTENTION_MAPS=true
fi

mlflow run --experiment-name $EXPERIMENT_NAME . --env-manager=local \
  -P model_name=$MODEL_NAME \
  -P scaling_method=min_max \
  -P weights_initializer=he \
  -P is_obpoint_labeldata=false \
  -P multi_parameter_model_return_sequences=false \
  -P 'input_parameters=rain' \
  -P train_is_max_datasize_limit=false \
  -P train_epochs=500 \
  -P train_separately=false \
  -P evaluate_save_attention_maps=$SAVE_ATTENTION_MAPS
