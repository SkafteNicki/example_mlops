## This defines all targets as phony targets, i.e. targets that are always out of date
## This is done to ensure that the commands are always executed, even if a file with the same name exists
## See https://www.gnu.org/software/make/manual/html_node/Phony-Targets.html
## Remove this if you want to use this Makefile for real targets
.PHONY: *

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = example_mlops
PYTHON_VERSION = 3.11
PYTHON_INTERPRETER = python
CURRENT_DIR = ${shell pwd}
GCP_PROJECT_NAME = $(shell gcloud config get-value project)

pwd:
	@echo $(CURRENT_DIR)

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Set up python interpreter environment
create_environment:
	conda create --name $(PROJECT_NAME) python=$(PYTHON_VERSION) --no-default-packages -y

## Install Python Dependencies
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -e .

## Install Developer Python Dependencies
dev_requirements: requirements
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -e .["dev"]

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

container_build:
	docker build . -t base:latest -f dockerfiles/base.dockerfile
	docker build . -t app:latest -f dockerfiles/app.dockerfile

container_run:  # run using "make container_run command=<your-command>"
	docker run \
		--entrypoint $(command) \
		--env-file .env \
		-v $(CURRENT_DIR)/configs:/app/configs/ \
		-v $(CURRENT_DIR)/data:/app/data/ \
		-v $(CURRENT_DIR)/outputs:/app/outputs/ \
		--rm \
		base:latest

container_serve:
	docker run \
		-p 8000:8000 \
		--env-file .env \
		-e "PORT=8000" \
		-e "MODEL_CHECKPOINT=best_mlops_team/example_mlops_project/mnist_model:v19" \
		--rm \
		app:latest

check_code:
	ruff check . --fix --exit-zero
	ruff format .
	mypy src

tests:
	coverage run -m pytest tests/unittests --disable-warnings
	coverage report -m

start_sweep:
	wandb sweep --project "example_mlops_project" --entity "best_mlops_team" configs/sweep.yaml

start_agent:  # Start agent for the latest sweep
	export SWEEP_ID=$$(python -c "import wandb; \
        print(wandb.Api().project('example_mlops_project', \
        entity='best_mlops_team').sweeps()[0].id)") && \
        wandb agent best_mlops_team/example_mlops_project/$$SWEEP_ID

# run using "make -i data_service_account" to ignore if the service account already exists
service_account:
	gcloud iam service-accounts create bucket-service-account \
		--description="Service account for data" --display-name="bucket-service-account"
	gcloud projects add-iam-policy-binding $(GCP_PROJECT_NAME) \
		--member="serviceAccount:bucket-service-account@$(GCP_PROJECT_NAME).iam.gserviceaccount.com" \
		--role="roles/storage.objectUser"
	gcloud projects add-iam-policy-binding $(GCP_PROJECT_NAME) \
		--member="serviceAccount:bucket-service-account@$(GCP_PROJECT_NAME).iam.gserviceaccount.com" \
		--role="roles/serviceusage.serviceUsageConsumer"
	gcloud projects add-iam-policy-binding $(GCP_PROJECT_NAME) \
		--member="serviceAccount:bucket-service-account@$(GCP_PROJECT_NAME).iam.gserviceaccount.com" \
		--role="roles/cloudbuild.builds.builder"
	gcloud projects add-iam-policy-binding $(GCP_PROJECT_NAME) \
		--member="serviceAccount:bucket-service-account@$(GCP_PROJECT_NAME).iam.gserviceaccount.com" \
		--role="roles/secretmanager.secretAccessor"
	gcloud projects add-iam-policy-binding $(GCP_PROJECT_NAME) \
		--member="serviceAccount:bucket-service-account@$(GCP_PROJECT_NAME).iam.gserviceaccount.com" \
		--role="roles/run.developer"
	gcloud projects add-iam-policy-binding $(GCP_PROJECT_NAME) \
		--member="serviceAccount:bucket-service-account@$(GCP_PROJECT_NAME).iam.gserviceaccount.com" \
		--role="roles/iam.serviceAccountUser"
	gcloud projects add-iam-policy-binding $(GCP_PROJECT_NAME) \
		--member="serviceAccount:bucket-service-account@$(GCP_PROJECT_NAME).iam.gserviceaccount.com" \
		--role="roles/iam.securityAdmin"
	gcloud projects add-iam-policy-binding $(GCP_PROJECT_NAME) \
		--member="serviceAccount:bucket-service-account@$(GCP_PROJECT_NAME).iam.gserviceaccount.com" \
		--role="roles/aiplatform.user"
	gcloud iam service-accounts keys create service_account_key.json \
		--iam-account=bucket-service-account@$(GCP_PROJECT_NAME).iam.gserviceaccount.com
	echo "service_account_key.json" >> .gitignore

serve_app:
	uvicorn src.example_mlops.app:app

loadtest:
	locust -f tests/loadtests/locustfile.py --headless -u 100 -r 10 --run-time 1m -H http://localhost:8000

deploy_model:
	gcloud builds submit \
		--config cloudbuild_deploy_model.yaml \
		--substitutions=_MODEL_NAME=testing-model,_MODEL_CHECKPOINT=best_mlops_team/example_mlops_project/mnist_model:v19

#################################################################################
# Documentation RULES                                                           #
#################################################################################

## Build documentation
build_documentation: dev_requirements
	mkdocs build --config-file docs/mkdocs.yaml --site-dir build

## Serve documentation
serve_documentation: dev_requirements
	mkdocs serve --config-file docs/mkdocs.yaml
