SHELL := /bin/sh

PROJECT_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
RESULTS_DIR ?= $(PROJECT_ROOT)/results

# File types produced by the training, evaluation, and plotting scripts.
RESULT_FILE_EXPR := \( \
	-name '*.csv' -o \
	-name '*.txt' -o \
	-name '*.json' -o \
	-name '*.log' -o \
	-name '*.png' -o \
	-name '*.gif' -o \
	-name '*.svg' -o \
	-name '*.pdf' -o \
	-name '*.npy' -o \
	-name '*.npz' -o \
	-name '*.pkl' -o \
	-name '*.pickle' -o \
	-name '*.parquet' \
\)

define assert_safe_results_dir
	@results_path="$$(realpath -m -- "$(RESULTS_DIR)")"; \
	case "$$results_path" in \
		"$(PROJECT_ROOT)/results"|"$(PROJECT_ROOT)/results/"*) ;; \
		*) echo "Refusing unsafe RESULTS_DIR: $(RESULTS_DIR)"; \
		   echo "It must be the repository's results directory or one of its children."; \
		   exit 1 ;; \
	esac
endef

.DEFAULT_GOAL := help

.PHONY: help clean clean-results clean-results-preview clean-results-dry-run clean-results-all

help:
	@printf '%s\n' \
		'make clean-results-preview          List generated result files that normal cleanup removes' \
		'make clean-results                  Remove generated result files and empty subdirectories' \
		'make clean                          Alias for clean-results' \
		'make clean-results-all CONFIRM=1    Remove everything below results/' \
		'make clean-results RESULTS_DIR=...  Clean only a directory below results/'

clean-results-preview clean-results-dry-run:
	$(assert_safe_results_dir)
	@if [ -d "$(RESULTS_DIR)" ]; then \
		find "$(RESULTS_DIR)" -type f $(RESULT_FILE_EXPR) -print | sort; \
	else \
		echo "No results directory found: $(RESULTS_DIR)"; \
	fi

clean: clean-results

clean-results:
	$(assert_safe_results_dir)
	@if [ -d "$(RESULTS_DIR)" ]; then \
		find "$(RESULTS_DIR)" -type f $(RESULT_FILE_EXPR) -delete; \
		find "$(RESULTS_DIR)" -depth -mindepth 1 -type d -empty -delete; \
	fi
	@mkdir -p "$(RESULTS_DIR)"
	@echo "Cleaned generated result files from $(RESULTS_DIR)"

clean-results-all:
	$(assert_safe_results_dir)
	@if [ "$(CONFIRM)" != '1' ]; then \
		echo 'Full cleanup not run. Re-run with CONFIRM=1.'; \
		exit 1; \
	fi
	@if [ -d "$(RESULTS_DIR)" ]; then \
		find "$(RESULTS_DIR)" -mindepth 1 -delete; \
	fi
	@mkdir -p "$(RESULTS_DIR)"
	@echo "Removed everything below $(RESULTS_DIR)"
