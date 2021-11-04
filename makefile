.PHONY: train
train:
	poetry run mlflow run . --no-conda

.PHONY: ui
ui:
	poetry run mlflow ui -p 2345

test: test.py
	poetry run python test.py