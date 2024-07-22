# Concept Extraction (ConExion)

## Overview

This repository contains code and resources for extracting concepts using unsupervised methods and large language models (LLMs). It includes setup instructions, scripts for running the models, and a brief guide on how to get started.

## Setup the Environment

To set up the environment, follow these steps:

1. Create and activate the conda environment:
    ```sh
    conda env create -f environment.yml
    conda activate conexion
    ```
2. Install the required Python packages:
    ```sh
    pip install -r requirements.txt
    ```

## Running the Scripts

To run the provided scripts, use the following command:

    ```sh
    nohup ./run_scripts.sh > logs/master_log.log 2>&1 &
    ```

## Keyword Extraction Methods
### Unsupervised Methods

Unsupervised keyword extraction methods rely on statistical and linguistic features of the text. These methods do not require labeled data. Common techniques include:
  TF-IDF (Term Frequency-Inverse Document Frequency): Weighs the importance of a term by comparing its frequency in a document to its frequency across all documents.
  TextRank: An algorithm inspired by PageRank, where words are nodes, and edges represent co-occurrence within a fixed window. Key phrases are identified by their importance in the network.
      LDA (Latent Dirichlet Allocation): A generative statistical model that identifies topics in a set of documents, which can then be used to extract relevant keywords.

### Large Language Models (LLMs)

Large language models can understand and generate human-like text. LLMs can be fine-tuned for concept extraction on specific datasets or used in zero-shot settings.

## File Structure

    batch_submit.sh: Script for submitting batch jobs.
    batch_test.sh: Script for testing batch jobs.
    download_models.sh: Script for downloading models.
    main.py: Main script for running the concept extraction.
    requirements.txt: Python package dependencies.
    run_fs_fixed_LLM-batch.sh: Script for running fixed LLMs in batch mode.
    run_fs_fixed_LLM-job.sh: Script for running fixed LLM jobs.
    run_fs_fixed_LLM-scripts.sh: Script for running fixed LLM scripts.
    run_fs_fixed_LLM_all_datasets.sh: Script for running fixed LLMs on all datasets.
    run_scripts.sh: Main script for running all other scripts.
    run_zs_LLM_scripts.sh: Script for running zero-shot LLM scripts.
    run_zs_fixedLLMscripts.sh: Script for running zero-shot fixed LLM scripts.


License

This project is licensed under the MIT License - see the LICENSE file for details.
