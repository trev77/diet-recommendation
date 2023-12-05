.PHONY: all preprocess train evaluate

all: preprocess train evaluate

preprocess:
	@echo "Running data preprocessing..."
	python src/preprocessing/data_preprocessing.py

train:
	@echo "Training models..."
	python src/training/train_portion_model.py
	python src/training/train_presence_model.py
	python src/training/train_embedding_model.py

evaluate:
	@echo "Evaluating models..."
	python src/evaluation/evaluate_embeddings.py
