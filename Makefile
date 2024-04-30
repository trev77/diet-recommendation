.PHONY: all preprocess train evaluate

all: preprocess train evaluate recommend

preprocess:
	@echo "Running data preprocessing..."
	python3 src/preprocessing/data_preprocessing.py

train:
	@echo "Training models..."
	python3 src/training/train_presence_model.py breakfast
	python3 src/training/train_presence_model.py lunch
	python3 src/training/train_presence_model.py dinner
	python3 src/training/train_embedding_model.py

evaluate:
	@echo "Evaluating models..."
	python3 src/evaluation/evaluate_embeddings.py

recommend:
	@echo "Recommending meals"
	python3 src/recommend/recommend_meals.py
	python3 src/recommend/evaluate_recommendations.py
