# Conda command
# Need to specify bash in order for conda activate to work.
SHELL=/bin/bash
# Note that the extra activate is needed to ensure that the activate floats env to the front of PATH
CONDA_ACTIVATE=source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate

EXPERIMENT_NAME = convlstm

.PHONY: train
train:
	$(CONDA_ACTIVATE) ml-dev && mlflow run --experiment-name $(EXPERIMENT_NAME) . --no-conda

.PHONY: ui
ui:
	$(CONDA_ACTIVATE) ml-dev && mlflow ui -p 2345

.PHONY: test
test:
	$(CONDA_ACTIVATE) ml-dev && python -m unittest -v

preprocess: preprocess/src/extract_data.py
	poetry run python preprocess/src/extract_data.py

# Poetry command
.PHONY: poetry_test
poetry_test:
	poetry run python -m unittest -v