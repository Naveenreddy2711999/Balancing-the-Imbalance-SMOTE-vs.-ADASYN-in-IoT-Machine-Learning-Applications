# 🚀 Balancing the Imbalance: SMOTE vs ADASYN in IoT Machine Learning Applications

### 👨‍💻 Author: [Tiyyagura Naveen Reddy](https://github.com/Naveenreddy2711999)

### 📅 Year: 2025

---

## 🧠 Overview

This project evaluates **class imbalance handling techniques**—**SMOTE**, **ADASYN**, and a **Hybrid (SMOTE + ADASYN)**—for **IoT network intrusion detection** using the **BoT-IoT dataset**.

The original dataset is **extremely imbalanced** (millions of attack samples vs a few hundred normal samples). We systematically **downsample**, **oversample**, and **benchmark multiple ML models** to identify the **most reliable and accurate combination** for real-world IoT security.

---

## 🎯 Objectives

* Analyze severe class imbalance in the BoT-IoT dataset.
* Apply and compare **SMOTE**, **ADASYN**, and **SMOTE + ADASYN**.
* Train and evaluate multiple supervised ML models.
* Select the **best-performing sampling + model** combination using robust metrics.

---

## 🧩 Methodology

### 🔹 Step 1: Dataset Preparation

* **Dataset:** BoT-IoT (75 CSV files from Kaggle)
* Merged, cleaned, and preprocessed using **Python (Pandas)**
* Selected **18 relevant features**

### 🔹 Step 2: Handling Class Imbalance

* **Downsampling** of the majority (Attack) class
* **Oversampling techniques:**

  * **SMOTE (Synthetic Minority Oversampling Technique)**
  * **ADASYN (Adaptive Synthetic Sampling)**
  * **Hybrid: SMOTE + ADASYN**

### 🔹 Step 3: Model Training

Four supervised ML models were trained and compared:

1. Logistic Regression
2. Support Vector Machine (SVM)
3. Random Forest
4. XGBoost

### 🔹 Step 4: Evaluation Metrics

* Accuracy
* Precision (Macro Avg)
* Recall (Macro Avg)
* F1-Score (Macro Avg)
* Confusion Matrix

---

## 📊 Class Distribution Summary

| Sampling Stage   | Attack (1) | Normal (0) | Total Samples | Class Ratio |
| ---------------- | ---------- | ---------- | ------------- | ----------- |
| Original Dataset | 3,668,048  | 474        | 3,668,522     | 1 : 7739    |
| Downsampled      | 75,000     | 474        | 75,474        | 1 : 158     |
| SMOTE            | 75,000     | 75,000     | 150,000       | 1 : 1       |
| ADASYN           | 75,000     | 74,995     | 149,995       | ≈ 1 : 1     |
| SMOTE + ADASYN   | 75,000     | 75,000     | 150,000       | 1 : 1       |

---

## ⚙️ Implementation Environment

* **Platform:** Google Colab / VS Code
* **Language:** Python 3
* **Libraries:**

  * pandas, numpy, matplotlib
  * scikit-learn
  * imbalanced-learn
  * xgboost
  * tqdm

---

## 🧾 Detailed Results (All Metrics)

| Sampling Technique | Model               | Accuracy          | Precision (Avg) | Recall (Avg) | F1-score (Avg) | Best Case |
| ------------------ | ------------------- | ----------------- | --------------- | ------------ | -------------- | --------- |
| SMOTE              | Logistic Regression | 0.97              | 0.97            | 0.97         | 0.97           | ❌         |
| SMOTE              | SVM                 | 0.98              | 0.98            | 0.98         | 0.98           | ❌         |
| SMOTE              | Random Forest       | **1.00**          | **1.00**        | **1.00**     | **1.00**       | ✔️        |
| SMOTE              | XGBoost             | **1.00**          | **1.00**        | **1.00**     | **1.00**       | ✔️        |
| ADASYN             | Logistic Regression | 0.95              | 0.95            | 0.95         | 0.95           | ❌         |
| ADASYN             | SVM                 | 0.95              | 0.95            | 0.95         | 0.95           | ❌         |
| ADASYN             | Random Forest       | 0.99              | 0.99            | 0.99         | 0.99           | ❌         |
| ADASYN             | XGBoost             | 0.99              | 0.99            | 0.99         | 0.99           | ❌         |
| SMOTE + ADASYN     | Logistic Regression | 0.97              | 0.97            | 0.97         | 0.97           | ❌         |
| SMOTE + ADASYN     | SVM                 | 0.97              | 0.97            | 0.97         | 0.97           | ❌         |
| SMOTE + ADASYN     | Random Forest       | **0.9995 ≈ 1.00** | **1.00**        | **1.00**     | **1.00**       | ✔️        |
| SMOTE + ADASYN     | XGBoost             | **0.9992 ≈ 1.00** | **1.00**        | **1.00**     | **1.00**       | ✔️        |

---

## 📈 Accuracy Comparison (SMOTE vs ADASYN)

```python
import matplotlib.pyplot as plt
models = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'SVM']
smote_acc = [0.97, 1.00, 1.00, 0.98]
adasyn_acc = [0.94, 0.992, 0.992, 0.90]

plt.figure(figsize=(10,6))
plt.bar(models, smote_acc, alpha=0.7, label='SMOTE')
plt.bar(models, adasyn_acc, alpha=0.7, label='ADASYN')
plt.title('Model Accuracy Comparison (SMOTE vs ADASYN)')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
```

---

## 🧩 Key Insights

* **SMOTE** generated clean and uniformly distributed synthetic samples, leading to **stable decision boundaries**.
* **ADASYN** emphasized hard-to-learn regions, which introduced **minor noise** and slightly reduced accuracy.
* **Hybrid (SMOTE + ADASYN)** increased complexity without delivering significant gains.
* **Tree-based models** (Random Forest, XGBoost) consistently outperformed linear models.

---

## 📊 Comparative Analysis Across Sampling Techniques

### 📌 Model Accuracy Comparison

| Model               | SMOTE    | ADASYN | SMOTE + ADASYN |
| ------------------- | -------- | ------ | -------------- |
| Logistic Regression | **0.97** | 0.95   | 0.97           |
| Random Forest       | **1.00** | 0.99   | **1.00**       |
| SVM                 | **0.98** | 0.95   | 0.97           |
| XGBoost             | **1.00** | 0.99   | **1.00**       |

**Observations:**

* SMOTE consistently delivers the highest accuracy across all models.
* ADASYN introduces minor noise, leading to slightly reduced performance.
* SMOTE + ADASYN does not significantly outperform SMOTE alone.

---

## 🏁 Final Results & Conclusion

### 🔹 Key Findings

* The original dataset was **extremely imbalanced**, making direct model training ineffective.
* Downsampling reduced majority dominance but retained imbalance.
* **SMOTE** achieved the best trade-off between accuracy, stability, and simplicity.
* **ADASYN** marginally reduced accuracy due to noise amplification.
* **Hybrid SMOTE + ADASYN** increased complexity without significant gains.

### 🏆 Final Recommendation
SMOTE combined with Random Forest or XGBoost provides the most reliable and accurate intrusion detection performance for highly imbalanced IoT datasets

---
