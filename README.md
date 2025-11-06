# 🚀 Balancing the Imbalance: SMOTE vs ADASYN in IoT Machine Learning Applications

### 👨‍💻 Author: [Tiyyagura Naveen Reddy](https://github.com/Naveenreddy2711999)
### 📅 Year: 2025  
### 🏫 Final Year Project — B.Tech CSE (Data Science)

---

## 🧠 Overview
This project investigates two advanced **oversampling techniques — SMOTE and ADASYN** — to handle **class imbalance** in IoT network attack detection using the **BoT-IoT dataset**.

The dataset was originally **highly imbalanced**, with over 3.6 million attack records and only a few hundred normal samples.  
We implemented, balanced, and evaluated multiple machine learning models to identify the **best-performing combination** for intrusion detection.

---

## 🎯 Objectives
- Analyze and handle class imbalance in the BoT-IoT dataset.  
- Apply **SMOTE** and **ADASYN** to balance classes.  
- Train and compare multiple **machine learning models**.  
- Identify the most effective model and technique for IoT security.

---

## 🧩 Methodology

### 🔹 Step 1: Dataset Preparation
- Used **BoT-IoT dataset** (75 CSV files from Kaggle).
- Merged and cleaned data using Python (Pandas).
- Selected 18 relevant features.

### 🔹 Step 2: Handling Imbalance
- **SMOTE (Synthetic Minority Oversampling Technique)**  
- **ADASYN (Adaptive Synthetic Sampling)**

### 🔹 Step 3: Model Training
Trained and compared four supervised ML models:
1. Logistic Regression  
2. Decision Tree  
3. Random Forest  
4. Support Vector Machine (SVM)

### 🔹 Step 4: Evaluation Metrics
- Accuracy  
- Precision  
- Recall  
- F1-Score  
- Confusion Matrix  

---

## 📊 Class Distribution Summary

| Stage | Attack (1) | Normal (0) | Total | Ratio |
|--------|-------------|-------------|--------|--------|
| Original | 3,668,025 | 497 | 3,668,522 | 1 : 7378 |
| Downsampled | 50,000 | 497 | 50,497 | 1 : 100 |
| After SMOTE | 50,000 | 50,000 | 100,000 | 1 : 1 |
| After ADASYN | 50,000 | 50,018 | 100,018 | 1 : 1 |

---

## ⚙️ Implementation Environment
- **Platform:** Google Colab / VS Code  
- **Language:** Python 3  
- **Libraries:**  
  - pandas, numpy, matplotlib  
  - scikit-learn  
  - imbalanced-learn  
  - tqdm  

---

## 🧾 Results Summary

| Model | SMOTE Accuracy | ADASYN Accuracy | Best |
|------|----------------|----------------|------|
| Logistic Regression | 0.97 | 0.94 | SMOTE |
| Decision Tree | 1.00 | 0.992 | SMOTE |
| Random Forest | 1.00 | 0.992 | SMOTE |
| SVM | 0.98 | 0.90 | SMOTE |

✅ **SMOTE outperformed ADASYN** across all models, with **Random Forest + SMOTE** achieving nearly **100% accuracy**.

---

## 📈 Visual Comparison

The figure below shows the accuracy comparison between SMOTE and ADASYN for all models.

*(You can generate it using matplotlib in the notebook)*

python
import matplotlib.pyplot as plt
models = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'SVM']
smote_acc = [0.97, 1.00, 1.00, 0.98]
adasyn_acc = [0.94, 0.992, 0.992, 0.90]

plt.figure(figsize=(10,6))
plt.bar(models, smote_acc, alpha=0.7, label='SMOTE')
plt.bar(models, adasyn_acc, alpha=0.7, label='ADASYN')
plt.title("Model Accuracy Comparison (SMOTE vs ADASYN)")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
