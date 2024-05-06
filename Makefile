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

container_run:  # run using "make container_run command=<your-command>"
	docker run \
		--entrypoint $(command) \
		--env-file .env \
		-v $(CURRENT_DIR)/configs:/app/configs/ \
		-v $(CURRENT_DIR)/data:/app/data/ \
		-v $(CURRENT_DIR)/outputs:/app/outputs/ \
		--rm \
		base:latest

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

#################################################################################
# Documentation RULES                                                           #
#################################################################################

## Build documentation
build_documentation: dev_requirements
	mkdocs build --config-file docs/mkdocs.yaml --site-dir build

## Serve documentation
serve_documentation: dev_requirements
	mkdocs serve --config-file docs/mkdocs.yaml
