while getopts "e:m:" opt
do
  case "$opt" in
    e ) EXPERIMENT_NAME="$OPTARG";;
    m ) MODEL_NAME="$OPTARG";;
  esac
done

if [[ $MODEL_NAME == "Seq2Seq" ]] 
then
  SAVE_ATTENTION_MAPS=false
else
  SAVE_ATTENTION_MAPS=true
fi

declare -a arr=("rain" "rain/temperature/humidity" "rain/v_wind/u_wind" "rain/temperature/humidity/v_wind/u_wind")
for i in "${arr[@]}"
do
  mlflow run --experiment-name $EXPERIMENT_NAME . --env-manager=local \
    -P model_name=$MODEL_NAME \
    -P scaling_method=min_max \
    -P weights_initializer=he \
    -P is_obpoint_labeldata=false \
    -P multi_parameter_model_return_sequences=false \
    -P "input_parameters=$i" \
    -P train_is_max_datasize_limit=false \
    -P train_epochs=500 \
    -P train_separately=false \
    -P evaluate_save_attention_maps=$SAVE_ATTENTION_MAPS
done
