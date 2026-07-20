SHELL := /bin/sh

PROJECT_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
RESULTS_DIR ?= $(PROJECT_ROOT)/results
LOGS_DIR ?= $(PROJECT_ROOT)/logs

# File types produced by the training, evaluation, and plotting scripts.
GENERATED_FILE_EXPR := \( \
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
	-name '*.parquet' -o \
	-name '*.pt' -o \
	-name '*.pth' -o \
	-name '*.ckpt' \
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

define assert_safe_logs_dir
	@logs_path="$$(realpath -m -- "$(LOGS_DIR)")"; \
	case "$$logs_path" in \
		"$(PROJECT_ROOT)/logs"|"$(PROJECT_ROOT)/logs/"*) ;; \
		*) echo "Refusing unsafe LOGS_DIR: $(LOGS_DIR)"; \
		   echo "It must be the repository's logs directory or one of its children."; \
		   exit 1 ;; \
	esac
endef

.DEFAULT_GOAL := help

.PHONY: help clean clean-preview clean-results clean-results-preview \
	clean-results-dry-run clean-results-all clean-logs clean-logs-preview \
	clean-logs-dry-run clean-logs-all clean-all

help:
	@printf '%s\n' \
		'make clean                          Clean generated files from logs/ and results/' \
		'make clean-preview                  List generated log and result files' \
		'make clean-results-preview          List generated result files that normal cleanup removes' \
		'make clean-results                  Remove generated result files and empty subdirectories' \
		'make clean-logs-preview             List generated log/checkpoint files' \
		'make clean-logs                     Remove generated log/checkpoint files and empty subdirectories' \
		'make clean-all CONFIRM=1            Remove everything below logs/ and results/' \
		'make clean-results-all CONFIRM=1    Remove everything below results/' \
		'make clean-logs-all CONFIRM=1       Remove everything below logs/' \
		'make clean-results RESULTS_DIR=...  Clean only a directory below results/' \
		'make clean-logs LOGS_DIR=...        Clean only a directory below logs/'

clean-preview: clean-results-preview clean-logs-preview

clean-results-preview clean-results-dry-run:
	$(assert_safe_results_dir)
	@if [ -d "$(RESULTS_DIR)" ]; then \
		find "$(RESULTS_DIR)" -type f $(GENERATED_FILE_EXPR) -print | sort; \
	else \
		echo "No results directory found: $(RESULTS_DIR)"; \
	fi

clean-logs-preview clean-logs-dry-run:
	$(assert_safe_logs_dir)
	@if [ -d "$(LOGS_DIR)" ]; then \
		find "$(LOGS_DIR)" -type f $(GENERATED_FILE_EXPR) -print | sort; \
	else \
		echo "No logs directory found: $(LOGS_DIR)"; \
	fi

clean: clean-results clean-logs

clean-results:
	$(assert_safe_results_dir)
	@if [ -d "$(RESULTS_DIR)" ]; then \
		find "$(RESULTS_DIR)" -type f $(GENERATED_FILE_EXPR) -delete; \
		find "$(RESULTS_DIR)" -depth -mindepth 1 -type d -empty -delete; \
	fi
	@mkdir -p "$(RESULTS_DIR)"
	@echo "Cleaned generated result files from $(RESULTS_DIR)"

clean-logs:
	$(assert_safe_logs_dir)
	@if [ -d "$(LOGS_DIR)" ]; then \
		find "$(LOGS_DIR)" -type f $(GENERATED_FILE_EXPR) -delete; \
		find "$(LOGS_DIR)" -depth -mindepth 1 -type d -empty -delete; \
	fi
	@mkdir -p "$(LOGS_DIR)"
	@echo "Cleaned generated log and checkpoint files from $(LOGS_DIR)"

clean-all: clean-results-all clean-logs-all

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

clean-logs-all:
	$(assert_safe_logs_dir)
	@if [ "$(CONFIRM)" != '1' ]; then \
		echo 'Full cleanup not run. Re-run with CONFIRM=1.'; \
		exit 1; \
	fi
	@if [ -d "$(LOGS_DIR)" ]; then \
		find "$(LOGS_DIR)" -mindepth 1 -delete; \
	fi
	@mkdir -p "$(LOGS_DIR)"
	@echo "Removed everything below $(LOGS_DIR)"
