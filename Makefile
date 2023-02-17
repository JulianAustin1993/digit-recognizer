.PHONY: data
#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROJECT_NAME = digit-recognizer
PYTHON_INTERPRETER = python3

#################################################################################
# COMMANDS                                                                      #
#################################################################################

data: 
	kaggle competitions download -c digit-recognizer -p data
	unzip data/digit-recognizer.zip -d data/
	rm data/digit-recognizer.zip