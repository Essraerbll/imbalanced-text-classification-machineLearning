# Imbalanced Text Classification: SMS Spam Detection Using Text Mining and Machine Learning

---

## 📚 Project Overview

This project focuses on solving the **imbalanced binary classification** problem in the context of **SMS spam detection**.  
Using a combination of **text mining**, **synthetic oversampling (SMOTE)**, and **Random Forest classification**, the project demonstrates how to effectively classify short text messages as either **spam** or **ham** (non-spam).

The primary objective is to improve classification performance on imbalanced datasets through effective text preprocessing and data balancing techniques.

---

## 🛠 Methods and Tools

- **Data Source:**  
  - SMS Spam Collection dataset (UCI Machine Learning Repository).

- **Text Preprocessing:**  
  - Lowercasing
  - Punctuation removal
  - Stopword removal (using NLTK)
  - Lemmatization (WordNetLemmatizer)

- **Feature Extraction:**  
  - **TF-IDF Vectorization** using unigrams, bigrams, and trigrams (`TfidfVectorizer`).

- **Data Balancing:**  
  - **SMOTE (Synthetic Minority Over-sampling Technique)** to balance the minority (spam) class.

- **Classification Model:**  
  - **Random Forest Classifier** with hyperparameters optimized for robustness and controlled model complexity.

- **Evaluation Metrics:**  
  - Accuracy
  - Confusion Matrix (True Positive, True Negative, False Positive, False Negative)

---

## 📦 File Structure

| File Name            | Purpose                                             |
|----------------------|-----------------------------------------------------|
| `Task5.py`            | Main script for preprocessing, training, and evaluation |
| `SMSSpamCollection`   | Text dataset (spam/ham messages)                    |

*Note: The dataset will be automatically downloaded if not available locally.*

---

## 🚀 How to Run

1. Install required libraries:
```bash
pip install numpy pandas scikit-learn imbalanced-learn nltk
```

2. Run the script:
```bash
python Task5.py
```

The script will:
- Preprocess text data
- Perform TF-IDF feature extraction
- Apply SMOTE for class balancing
- Train a Random Forest model
- Output evaluation metrics (accuracy, confusion matrix)

---

## 📈 Key Results

- **Balancing the dataset with SMOTE** significantly improves the classification of minority (spam) messages.
- **TF-IDF with n-gram features** enhances the model’s ability to capture contextual information.
- **Random Forest Classifier** achieves high accuracy with controlled overfitting, thanks to hyperparameter tuning.

---

## ✨ Motivation

In real-world text classification tasks, datasets are often highly imbalanced (e.g., spam detection, fraud detection, rare event prediction).  
Without proper handling, standard classifiers tend to be biased toward the majority class.

This project illustrates:
- The critical importance of **data balancing techniques** like SMOTE.
- The effectiveness of **feature engineering** through **text mining**.
- How simple, interpretable models like Random Forests can be highly effective when trained on properly preprocessed and balanced datasets.

---

## 🧠 Future Work

- Experiment with alternative classifiers (e.g., XGBoost, LightGBM).
- Compare SMOTE with other resampling methods (e.g., ADASYN, Tomek Links).
- Incorporate word embeddings (e.g., Word2Vec, GloVe) for richer feature representation.

---

## 📢 Acknowledgements

This project was inspired by challenges in **imbalanced learning**, **natural language processing (NLP)**, and **applied machine learning** in text classification.

---

# 🔥 Academic Keywords

> Text Mining, TF-IDF, SMOTE, Random Forest, Imbalanced Learning, SMS Spam Detection, Natural Language Processing (NLP), Binary Classification, Data Preprocessing

---
