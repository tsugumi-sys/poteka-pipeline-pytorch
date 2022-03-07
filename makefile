# Conda command
# Need to specify bash in order for conda activate to work.
SHELL=/bin/bash
# Note that the extra activate is needed to ensure that the activate floats env to the front of PATH
CONDA_ACTIVATE=source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate

EXPERIMENT_NAME = convlstm

CONDA_ENV_NAME = poteka-pipeline-pytorch

.PHONY: train
train:
	$(CONDA_ACTIVATE) $(CONDA_ENV_NAME) && mlflow run --experiment-name $(EXPERIMENT_NAME) . --no-conda

.PHONY: ui
ui:
	$(CONDA_ACTIVATE) $(CONDA_ENV_NAME) &&  mlflow ui -p 2345

.PHONY: test
test:
	$(CONDA_ACTIVATE) $(CONDA_ENV_NAME) &&  python -m unittest -v