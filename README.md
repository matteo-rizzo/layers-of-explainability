# Layers of Explainability

## Introduction

This repository contains the implementation for the paper "Crossing the Divide: Designing Layers of Explainability". It
presents a novel approach to text classification that emphasizes explainability without significantly compromising
performance. **The paper is currently under review for a conference submission.**

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

2. **Place the Dataset in the `dataset/` Directory**: Move the dataset into the `dataset/` directory of this repository.

### IMDb Dataset for Sentiment Analysis

1. **Download the Dataset**: Access and download the IMDb dataset
   from [Kaggle](https://www.kaggle.com/columbine/imdb-dataset-sentiment-analysis-in-csv-format) or
   the [official IMDb website](https://www.imdb.com/interfaces/).

2. **Place the Dataset in the `dataset/` Directory**: Move the dataset into the `dataset/` directory of this repository.

## Usage

To use the repository:

1. **Generate Explanations**: Navigate to `src/explanations/` and run the scripts to generate explanations.

2. **Evaluate Accuracy**: Go to `src/text_classification/` and execute the scripts to reproduce the accuracy results.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.