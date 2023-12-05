# Diet Recommendation Project v1

This repository contains the codebase for the Diet Recommendation project (v1).

## Project Structure

- `data/`: Directory for all data-related items
  - `raw/`: Raw data, do not alter, do not upload large files to version control
  - `processed/`: Processed data, ready for training
  - `mappings/`: Mappings needed in later processing and training
  - `embeddings/`: Meal, food, and ingredient embedding and graph data
- `models/`: Trained model files
- `src/`: Source files for the project
  - `preprocessing/`: Scripts for data preprocessing
  - `training/`: Scripts for model training
  - `evaluation/`: Scripts for model evaluation
  - `utils/`: Utility scripts and helper functions
- `logs/`: Logs from model training and evaluation
- `results/`: Results from model evaluations

## Setup

1. Clone the repository and navigate to the project directory.
2. Ensure you are using Python 3.9.
3. Run `pip install -r requirements.txt` to install dependencies.
4. Execute `bash setup_repo.sh` to create the directory structure and initial setup files.

## Running the Pipeline

### Locally Using Makefile

You can run the entire pipeline or individual stages using the `Makefile`:

- Run the entire pipeline: `make all`
- Only preprocess data: `make preprocess`
- Only train models: `make train`
- Only evaluate models: `make evaluate`

### Using GitHub Actions

The project is configured with GitHub Actions for Continuous Integration (CI) and Continuous Deployment (CD):

- **CI**: On every push and pull request to the `main` branch, the CI pipeline tests the code.
- **CD**: The CD pipeline can be triggered manually via GitHub Actions to run the full pipeline, or it's scheduled to run automatically at midnight daily.

## Data Version Control (DVC)

We use DVC to track data and model versions:

1. Add new data or models: `dvc add your_file_or_directory`
2. Commit the changes: `git add . && git commit -m "Add changes"`
3. Push the changes: `git push && dvc push`

Refer to the [DVC documentation](https://dvc.org/doc) for more details.

## Experiment Tracking with Weights & Biases (W&B)

This project uses W&B for experiment tracking. Ensure you are logged in to W&B and have it integrated into the training and evaluation scripts.

Refer to the [W&B documentation](https://docs.wandb.ai/) for usage instructions.

## Contribution Guidelines

To contribute to this project:

1. Fork the repository and create a new branch (`git checkout -b feature/AmazingFeature`).
2. Make changes and test them.
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a pull request.

## License

This project is licensed under the Apache-2.0 License. Please see the [LICENSE](./LICENSE) file for details.
