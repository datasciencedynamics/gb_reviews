# Makefile
# ------------------------------------------------------------------------------
# GLOBALS
# ------------------------------------------------------------------------------
PROJECT_NAME = gb_reviews
PYTHON_VERSION = 3.12
PYTHON_INTERPRETER = python
VENV_DIR = gb_venv
CONDA_ENV_NAME = conda_gb
MAKEFILE_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
PROJECT_DIRECTORY := $(abspath $(MAKEFILE_DIR))
export PYTHONPATH := $(PROJECT_DIRECTORY)

RAW_META       ?= $(PROJECT_DIRECTORY)/data/raw/gb_meta.jsonl
RAW_REVIEWS    ?= $(PROJECT_DIRECTORY)/data/raw/gb_reviews.jsonl
PROCESSED_DATA ?= $(PROJECT_DIRECTORY)/data/processed/reviews.parquet
LABELS         ?= $(PROJECT_DIRECTORY)/data/interim/labels_primary.jsonl
LABELS_CHECK   ?= $(PROJECT_DIRECTORY)/data/interim/labels_check.jsonl
APP_DATA_DIR   ?= $(PROJECT_DIRECTORY)/data/app
LABELS_DRYRUN       = $(PROJECT_DIRECTORY)/data/interim/labels_dryrun.jsonl
LABELS_CHECK_DRYRUN = $(PROJECT_DIRECTORY)/data/interim/labels_check_dryrun.jsonl
## LIMIT > 0 is a dry run: it writes to a scratch file, wiped first, so it always runs
DRYRUN           = $(if $(filter 0,$(strip $(LIMIT))),,1)
LABELS_OUT       = $(if $(DRYRUN),$(LABELS_DRYRUN),$(LABELS))
LABELS_CHECK_OUT = $(if $(DRYRUN),$(LABELS_CHECK_DRYRUN),$(LABELS_CHECK))

############################## Labeling Globals ################################

PRIMARY_MODEL ?= claude-haiku-5-5   # labels every review
CHECK_MODEL   ?= claude-sonnet-5-5  # relabels a sample for agreement
CHECK_FRAC    ?= 0.10               # share of reviews relabeled
CONCURRENCY   ?= 8                  # parallel API requests
LIMIT         ?= 0                  # 0 = all reviews; >0 for a dry run

############################ Change-Point Globals ##############################

CP_SAUCE      ?= super_hot
CP_BOOTSTRAPS ?= 1000
APP_PORT      ?= 8050
FORCE         ?= 0   # 1 = redo a step even if its output exists

# ------------------------------------------------------------------------------
# COMMANDS
# ------------------------------------------------------------------------------

## Set up python interpreter environment
.PHONY: create_conda_env
create_conda_env:
	@echo "Run 'conda create -n $(CONDA_ENV_NAME) python=$(PYTHON_VERSION)' to create conda environment"

## Create a virtual environment
.PHONY: create_venv
create_venv:
	$(PYTHON_INTERPRETER) -m venv $(VENV_DIR)
	@echo "Virtual environment created with $(PYTHON_INTERPRETER)$(PYTHON_VERSION)"

## Print the activation commands
.PHONY: activate_venv
activate_venv:
	@echo "Run 'conda deactivate' to deactivate the $(CONDA_ENV_NAME) conda environment"
	@echo "Run 'source $(VENV_DIR)/bin/activate' to activate the virtual environment"

## Remove the virtual environment
.PHONY: clean_venv
clean_venv:
	rm -rf $(VENV_DIR)
	@echo "Virtual environment removed"

## Install Python Dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt

## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

setup_dir_venv: create_folders create_conda_env create_venv activate_venv

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

################################################################################
################################ Folder Creation  ##############################
################################################################################

## Create the data and report folders
.PHONY: create_folders
create_folders:
	mkdir -p data/raw data/interim data/processed data/app
	mkdir -p reports/eval
	touch data/raw/.gitkeep data/interim/.gitkeep data/processed/.gitkeep
	touch data/app/.gitkeep reports/eval/.gitkeep

## Delete all generated data (keeps code)
.PHONY: clean_dir
clean_dir:
	@echo "Cleaning directory..."
	rm -rf data/

################################################################################
############################## Data Pipeline ###################################
################################################################################

## Stream Amazon Reviews'23 and keep Gringo Bandito rows (slow; FORCE=1 to re-pull)
.PHONY: data_gen
data_gen:
	@if [ -f $(RAW_REVIEWS) ] && [ "$(FORCE)" != "1" ]; then \
		echo "Raw data already at $(RAW_REVIEWS); skipping. Use 'make data_gen FORCE=1' to re-pull."; \
	else \
		mkdir -p data/raw; \
		$(PYTHON_INTERPRETER) $(PROJECT_DIRECTORY)/preprocessing/data_gen.py \
			--meta-out $(RAW_META) \
			--reviews-out $(RAW_REVIEWS) \
		2>&1 | tee data/raw/data_gen.txt; \
	fi
	
## Clean reviews, map each listing to a sauce, write Parquet
.PHONY: preprocess
preprocess:
	@mkdir -p data/processed
	$(PYTHON_INTERPRETER) $(PROJECT_DIRECTORY)/preprocessing/preprocessing.py \
		--meta-file $(RAW_META) \
		--reviews-file $(RAW_REVIEWS) \
		--output-data-file $(PROCESSED_DATA) \
	2>&1 | tee data/processed/preprocessing.txt

preproc_pipeline: data_gen preprocess

################################################################################
################################# Labeling #####################################
################################################################################

## Label every review with the primary model (resumable)
.PHONY: label_primary
label_primary:
	@mkdir -p data/interim
	@rm -f $(LABELS_DRYRUN)
	@if [ "$(FORCE)" = "1" ] && [ -f $(LABELS) ]; then \
		mv $(LABELS) $(LABELS).$$(date +%Y%m%d_%H%M%S).bak; \
		echo "Moved existing labels to a .bak file; relabeling from scratch."; \
	fi
	$(PYTHON_INTERPRETER) $(PROJECT_DIRECTORY)/labeling/label_reviews.py \
		--input-data-file $(PROCESSED_DATA) \
		--output-file $(LABELS_OUT) \
		--model $(PRIMARY_MODEL) \
		--sample-frac 1.0 \
		--concurrency $(CONCURRENCY) \
		--limit $(LIMIT) \
	2>&1 | tee data/interim/label_primary.txt

## Relabel a random sample with the check model
.PHONY: label_check
label_check:
	@mkdir -p data/interim
	@rm -f $(LABELS_CHECK_DRYRUN)
	@if [ "$(FORCE)" = "1" ] && [ -f $(LABELS_CHECK) ]; then \
		mv $(LABELS_CHECK) $(LABELS_CHECK).$$(date +%Y%m%d_%H%M%S).bak; \
		echo "Moved existing labels to a .bak file; relabeling from scratch."; \
	fi
	$(PYTHON_INTERPRETER) $(PROJECT_DIRECTORY)/labeling/label_reviews.py \
		--input-data-file $(PROCESSED_DATA) \
		--output-file $(LABELS_CHECK_OUT) \
		--model $(CHECK_MODEL) \
		--sample-frac $(CHECK_FRAC) \
		--concurrency $(CONCURRENCY) \
		--limit $(LIMIT) \
	2>&1 | tee data/interim/label_check.txt

label_pipeline: label_primary label_check

################################################################################
############################ Validation + Modeling #############################
################################################################################

## Agreement between models and with star ratings
.PHONY: validate
validate:
	@mkdir -p reports/eval
	$(PYTHON_INTERPRETER) $(PROJECT_DIRECTORY)/modeling/validate.py \
		--reviews-file $(PROCESSED_DATA) \
		--primary-labels $(LABELS) \
		--check-labels $(LABELS_CHECK) \
		--output-file reports/eval/validation.json \
	2>&1 | tee reports/eval/validation.txt

## Change-point model for heat complaints on one sauce
.PHONY: changepoint
changepoint:
	@mkdir -p reports/eval
	$(PYTHON_INTERPRETER) $(PROJECT_DIRECTORY)/modeling/changepoint.py \
		--reviews-file $(PROCESSED_DATA) \
		--labels-file $(LABELS) \
		--sauce $(CP_SAUCE) \
		--n-bootstraps $(CP_BOOTSTRAPS) \
		--output-file reports/eval/changepoint_$(CP_SAUCE).json \
	2>&1 | tee reports/eval/changepoint_$(CP_SAUCE).txt

## Aggregate everything the app reads into data/app
.PHONY: export_app
export_app:
	@mkdir -p $(APP_DATA_DIR)
	$(PYTHON_INTERPRETER) $(PROJECT_DIRECTORY)/modeling/aggregate.py \
		--reviews-file $(PROCESSED_DATA) \
		--labels-file $(LABELS) \
		--validation-file reports/eval/validation.json \
		--changepoint-file reports/eval/changepoint_$(CP_SAUCE).json \
		--output-dir $(APP_DATA_DIR)

model_pipeline: validate changepoint export_app

## Full run: pull, preprocess, label, validate, model, export
.PHONY: full_pipeline
full_pipeline: preproc_pipeline label_pipeline model_pipeline

################################################################################
################################## App #########################################
################################################################################

## Run the Dash app locally
.PHONY: app
app:
	GB_APP_DATA=$(APP_DATA_DIR) $(PYTHON_INTERPRETER) -m app.app --port $(APP_PORT)

## Copy the app and its data artifacts into a flask_apps checkout (FLASK_APPS=path)
.PHONY: export_flask_apps
export_flask_apps:
	@test -n "$(FLASK_APPS)" || (echo "Set FLASK_APPS=/path/to/flask_apps" && exit 1)
	mkdir -p $(FLASK_APPS)/gb_reviews/data
	cp app/app.py $(FLASK_APPS)/gb_reviews/
	cp -r app/assets $(FLASK_APPS)/gb_reviews/
	cp $(APP_DATA_DIR)/*.parquet $(APP_DATA_DIR)/*.json $(FLASK_APPS)/gb_reviews/data/
	@echo "Copied to $(FLASK_APPS)/gb_reviews"

################################################################################
################################# Tests ########################################
################################################################################

## Run unit tests (synthetic fixtures only, no API calls)
.PHONY: test
test:
	$(PYTHON_INTERPRETER) -m pytest -q tests

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
