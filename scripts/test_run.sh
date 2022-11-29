while getopts "e:m:" opt
do
  case "$opt" in
    e ) EXPERIMENT_NAME="$OPTARG";;
    m ) MODEL_NAME="$OPTARG";;
  esac
done

mlflow run --experiment-name $EXPERIMENT_NAME . --env-manager=local \
  -P model_name=$MODEL_NAME \
  -P scaling_method=min_max \
  -P weights_initializer=he \
  -P 'input_parameters=rain' \
  -P train_is_max_datasize_limit=true \
  -P train_epochs=5 \
  -P train_separately=false
   
