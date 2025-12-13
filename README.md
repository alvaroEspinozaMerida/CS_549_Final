# CS 549 Final Project

This project focuses on detecting malicious URLs using lexical feature extraction and multiple machine learning models. We implemented and compared four models: Random Forest, SVM, XGBoost, and a simple Artificial Neural Network. The goal of the project is to evaluate how well different models perform when trained on the same cleaned dataset and feature set.

## Requirements

This project was developed using Python 3.10 to 3.13.

All required libraries are listed in `requirements.txt`.

Main packages include:

pandas
numpy
scikit learn
imbalanced learn
xgboost
matplotlib
jupyter

Install everything from `requirements.txt` to avoid missing dependencies.

## Setup

From the repo root, create and activate an environment, then install dependencies using the provided requirements file.

### Using venv

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data Files

This project uses three public URL datasets. Download the CSV files and place them inside the `data/` directory:

- `Phishing URLs.csv`
- `urldata.csv`
- `malicious_phish.csv`

These files may contain extra columns depending on the source. The preprocessing script automatically normalizes and removes unnecessary columns

## Dataset Generation

Before running any of the notebooks, the dataset must be cleaned and feature engineered.

From the repository root, run:

```bash
python src/util.py
```

This script performs the following steps:

- Merges the three raw datasets
- Cleans and normalizes class labels
- Removes duplicates and invalid URLs
- Extracts lexical features
- Saves the results to:
  - `data/all_urls.csv`
  - `data/final_dataset.csv`

## Run notebooks

 Each notebook follows the same general pipeline so results across models are directly comparable.

Make sure `python src/util.py` has been run successfully before opening any notebook.
