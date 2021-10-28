.PHONY: train
train:
	poetry run mlflow run . --no-conda

.PHONY: ui
ui:
	poetry run mlflow ui -p 2345