# Example repository for MLOps

Just a simple repository to demonstrate MLOps. Stack includes:
- Python
- Docker
- GitHub Actions
- Weights & Biases
- Google Cloud Platform
- FastAPI

# Steps

Make sure you have `Make` installed on your system

1. Clone the repository
    ```bash
    git clone https://github.com/SkafteNicki/example_mlops
    ```

2. Create environment

    ```bash
    make create_environment
    ```

    this will create a conda environment called `example_mlops`

3. Install all dependencies

    ```bash
    make requirements
    ```

    or

    ```bash
    make dev_requirements
    ```

    which will install all project requirements and development requirements

4. Fetch the data

    ```bash
    dvc pull
    ```

    the data is stored in a GCP bucket and managed by DVC (Data Version Control). You need to have DVC installed and configured with your GCP credentials.

4. Installing the requirements and the project also install 3 project scripts as easy to use commands:
    - `train`: to train a model
    - `evaluate`: to evaluate a model
    - `model_management`: to manage models

    Try running `train` in the terminal and see the model training process. Afterwards, you can run
    `evaluate` with path to the model you just trained to see the evaluation results.
