# Conda command
# Need to specify bash in order for conda activate to work.
SHELL=/bin/bash
# Note that the extra activate is needed to ensure that the activate floats env to the front of PATH
CONDA_ACTIVATE=source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate

CONDA_ENV_NAME = poteka-pipeline-pytorch

EXPERIMENT_NAME = Conv-vs-SA
MODEL_NAME = SAMSeq2Seq

###
# Scripts for train
###
.PHONY: train
train:
	$(CONDA_ACTIVATE) $(CONDA_ENV_NAME) && chmod +x scripts/train.sh && scripts/train.sh -e $(EXPERIMENT_NAME) -m $(MODEL_NAME)
	
.PHONY: train_all_models
train_all_models:
	$(CONDA_ACTIVATE) $(CONDA_ENV_NAME) && chmod +x scripts/train_all_models.sh && scripts/train_all_models.sh -e $(EXPERIMENT_NAME)

.PHONY: train_different_inputs
train_different_inputs:
	$(CONDA_ACTIVATE) $(CONDA_ENV_NAME) && chmod +x scripts/train_different_inputparams.sh \
		&& scripts/train_different_inputparams.sh -e $(EXPERIMENT_NAME) -m $(MODEL_NAME)

###
# scripts for test
###
.PHONY: test-train
test-train:
	$(CONDA_ACTIVATE) $(CONDA_ENV_NAME) && chmod +x scripts/test_run.sh  && scripts/test_run.sh -e test-run -m $(MODEL_NAME)	
	
.PHONY: test-all-models
test-all-models:
	$(CONDA_ACTIVATE) $(CONDA_ENV_NAME) && chmod +x scripts/test_all_models.sh && scripts/test_all_models.sh -e test-run

.PHONY: ui
ui:
	$(CONDA_ACTIVATE) $(CONDA_ENV_NAME) &&  mlflow ui -p 2345

.PHONY: test
test:
	$(CONDA_ACTIVATE) $(CONDA_ENV_NAME) &&  python -m unittest -v

.PHONY: format
format:
	$(CONDA_ACTIVATE) $(CONDA_ENV_NAME) && black .

# DEV in poetry commands
.PHONY: poetry_train
poetry_train:
	poetry run mlflow run --experiment-name ${EXPERIMENT_NAME} --env-manager local \
		-P 'input_parameters=rain/temperature' -P use_dummy_data=true -P use_test_model=false .

.PHONY: poetry_ui
poetry_ui:
	poetry run mlflow ui

.PHONY: poetry_test
poetry_test:
	poetry run python -m unittest -v $(TARGET_MODULE)
