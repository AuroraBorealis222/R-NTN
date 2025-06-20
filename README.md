# Phishing Detection via Feature Extraction and XGBoost

This repository provides a lightweight pipeline for detecting phishing addresses on Ethereum using structured features and XGBoost classification. You may either use our preprocessed dataset or generate features from your own data.

---

## 📦 Project Structure

├── dataset  #Dataset
├── Feature_acquisition.py # Feature extraction entry
├── Feature_computation.py   #Feature extraction helper function
├── skip_gram/
│ └── skip_gram.py # Skip-gram embedding for account features
├── computation.py # Feature extraction helper function
├── XGB.py # XGBoost model training and evaluation
├── requirements.txt #Dependencies
---

## 🚀 Quick Start

### 🔹 Option 1: Use our provided dataset

If you would like to quickly try out the model using our preprocessed data, simply run:

```bash
python XGB.py
This will train and evaluate the XGBoost model on our dataset.

🔹 Option 2: Use your own dataset
If you wish to apply the pipeline to your own data, follow these steps:

Run feature extraction:
```bash
python Feature_acquisition.py

Run skip-gram encoding (optional, for behavioral/semantic embeddings):
```bash
python skip_gram/skip_gram.py

These scripts will generate a .csv file containing extracted features.

Train the model:
```bash
python XGB.py
Make sure your generated CSV is correctly referenced in XGB.py.


🛠 Requirements
Python 3.8+
pandas
numpy
scikit-learn
xgboost
tqdm

You can install them with:
pip install -r requirements.txt
