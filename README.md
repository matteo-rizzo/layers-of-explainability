# Layers of Explainability

## Introduction

This repository contains the implementation for the paper "Crossing the Divide: Designing Layers of Explainability." It
presents a novel approach to text classification that emphasizes explainability without significantly compromising
performance. The paper has been accepted at "The 23rd International Conference on Artificial Intelligence and Soft Computing" (ICAISC 2024).

**Model dumps** and **SUPPLEMENTAL MATERIAL** are
shared in [GDrive folder](https://drive.google.com/drive/folders/163E_XKLPwEDqZwnKbGgMlhpfGxPVmPiM?usp=sharing).

## Abstract

In the era of deep learning, the opaque nature of sophisticated models often stands at odds with the growing demand for
transparency and explainability in Artificial Intelligence. This paper introduces a novel approach to text
classification that emphasizes explainability without significantly compromising performance. We propose a modular
framework to distill and aggregate information in a manner conducive to human interpretation. At the core of our
methodology is the premise that features extracted at the finest granularity are inherently explainable and reliable;
compared with methods whose explanation is on word-level importance, this layered aggregation of low-level features
allows us to trace a clearer decision trail of the model's decision-making process. Our results demonstrate this
approach yields effective explanations with a marginal reduction in accuracy, presenting a compelling trade-off for
applications where understandability is paramount.

## Repository Structure

- `dataset/`: Contains the datasets used in the study.
- `dumps/`: Contains dumps of models
- `src/`: Source code for the proposed framework.
    - `src/explanations/`: Scripts for generating explanations.
    - `src/text_classification/base_models/evaluate_accuracy.py`: Scripts to reproduce the accuracy results reported in
      the paper for the **base models**.
    - `src/text_classification/deep_learning/evaluate_accuracy.py`: Scripts to reproduce the accuracy results reported
      in the paper for the **deep learning models**.

## Installation Instructions

```bash
# Create a Virtual Environment
python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install Requirements
pip install -r requirements.txt
```

## Dataset Download Instructions

### CMSB Dataset ("Call me sexist but") for Sexism Detection

1. **Download the Dataset**: Access and download the CMSB dataset from
   the [official source academic database](https://search.gesis.org/research_data/SDN-10.7802-2251).

2. **Place the Dataset in the `dataset/` Directory**: Move the dataset into this repository's `dataset/` directory.

### IMDb Dataset for Sentiment Analysis

1. **Download the Dataset**: Access and download the IMDb dataset
   from [Kaggle](https://www.kaggle.com/columbine/imdb-dataset-sentiment-analysis-in-csv-format) or
   the [official IMDb website](https://www.imdb.com/interfaces/).

2. **Place the Dataset in the `dataset/` Directory**: Move the dataset into this repository's `dataset/` directory.

## Download dumps

1. Download pretrained models
   from [GDrive folder](https://drive.google.com/drive/folders/163E_XKLPwEDqZwnKbGgMlhpfGxPVmPiM?usp=sharing).

2. Place them in `dumps/` folder

## Usage

To use the repository:

1. **Generate Explanations**: Navigate to `src/explanations/` and run the scripts to generate explanations.
    - in the script, set the model path in `local_explanation`/`local_explanation_deep`
    - set the dataset to be used

2. **Evaluate Accuracy**: Go to `src/text_classification/` and execute the scripts to reproduce the accuracy results
    - use `base_models/evaluate_accuracy.py` or `deep_learning/evaluate_accuracy.py`
    - set the model and `DATASET` variables in both scripts (deep model name must be set
      in `src/text_classification/deep_learning/config.yml` in `testing` -> `model name`)

### Train new models

You can use script `src/text_classification/deep_learning/finetune.py`
and `src/text_classification/base_models/train_classifier.py` to finetune a LM from HuggingFace or train a XGBoost
classifier respectively.
Features can be extracted running `src/features_analysis/aggregate_features.py`, and setting the correct dataset to be
used in the scripts.

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
