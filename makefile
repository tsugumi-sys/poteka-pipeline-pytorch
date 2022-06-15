# Conda command
# Need to specify bash in order for conda activate to work.
SHELL=/bin/bash
# Note that the extra activate is needed to ensure that the activate floats env to the front of PATH
CONDA_ACTIVATE=source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate

EXPERIMENT_NAME = test_run

CONDA_ENV_NAME = poteka-pipeline-pytorch

.PHONY: multi_train
multi_train:
	$(CONDA_ACTIVATE) $(CONDA_ENV_NAME) && chmod +x ./scripts/run2.sh && ./scripts/run2.sh

.PHONY: train
train:
	$(CONDA_ACTIVATE) $(CONDA_ENV_NAME) && mlflow run --experiment-name ${EXPERIMENT_NAME} . --env-manager=local \
		-P override_hydra_conf='input_parameters=rain/temperature' -P use_dummy_data=true -P use_test_model=true

.PHONY: ui
ui:
	$(CONDA_ACTIVATE) $(CONDA_ENV_NAME) &&  mlflow ui -p 2345

.PHONY: test
test:
	$(CONDA_ACTIVATE) $(CONDA_ENV_NAME) &&  python -m unittest -v


# DEV in Windows commands
.PHONY: poetry_train
poetry_train:
	poetry run mlflow run --experiment-name ${EXPERIMENT_NAME} . --env-manager=local \
		-P override_hydra_conf='input_parameters=rain/temperature' -P use_dummy_data=true -P use_test_model=true

.PHONY: poetry_ui
poetry_ui:
	poetry run mlflow ui

.PHONY: poetry_test
poetry_test:
	poetry run python -m unittest -v
