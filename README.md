# Sampling Techniques for Imbalanced Credit Card Dataset

**Name:** Ishaan Sharma  
**Roll Number:** 102303442

---

## 📋 Assignment Overview

This project focuses on understanding the importance of sampling techniques in handling imbalanced datasets and analyzing how different sampling strategies affect the performance of various machine learning models.

### Problem Statement

We are given a highly imbalanced credit card fraud detection dataset. The task is to:
1. Balance the dataset using different sampling techniques
2. Apply five different machine learning models
3. Evaluate and compare the accuracy of each model-sampling combination
4. Determine which sampling technique gives higher accuracy on which model

---

## 🎯 Objectives

- Understand the impact of class imbalance on machine learning models
- Implement and compare five different sampling techniques
- Evaluate five different ML models on balanced datasets
- Analyze which sampling technique works best for each model
- Generate comprehensive results and visualizations

---

## 📊 Dataset

**Dataset:** Credit Card Fraud Detection Dataset  
**Source:** `Creditcard_data.csv`

### Dataset Characteristics
- **Features:** 31 features (V1-V28, Time, Amount)
- **Target:** Class (0 = Legitimate, 1 = Fraud)
- **Class Distribution:** Highly imbalanced (majority class: Legitimate transactions)

---

## 🔬 Sampling Techniques Implemented

| Code | Technique | Description |
|------|-----------|-------------|
| **Sampling1** | Random Under-Sampling | Reduces majority class by randomly removing samples |
| **Sampling2** | Random Over-Sampling | Increases minority class by randomly duplicating samples |
| **Sampling3** | SMOTE | Synthetic Minority Over-sampling Technique - creates synthetic samples |
| **Sampling4** | ADASYN | Adaptive Synthetic Sampling - focuses on difficult examples |
| **Sampling5** | Tomek Links | Removes borderline majority samples to clean decision boundary |

---

## 🤖 Machine Learning Models

| Code | Model | Algorithm |
|------|-------|-----------|
| **M1** | Logistic Regression | Linear classifier for binary classification |
| **M2** | Decision Tree | Tree-based classifier using feature splits |
| **M3** | Random Forest | Ensemble of decision trees |
| **M4** | SVM | Support Vector Machine with RBF kernel |
| **M5** | KNN | K-Nearest Neighbors classifier |

---

## 📈 Results

### Accuracy Results Table (%)

| Model | Sampling1 | Sampling2 | Sampling3 | Sampling4 | Sampling5 |
|-------|-----------|-----------|-----------|-----------|-----------|
| **M1** (Logistic Regression) | 69.40 | 92.24 | 93.97 | 93.97 | 98.71 |
| **M2** (Decision Tree) | 38.79 | 96.98 | 98.28 | 98.28 | 98.28 |
| **M3** (Random Forest) | 66.81 | 99.14 | 98.28 | 99.14 | 98.71 |
| **M4** (SVM) | 83.19 | 96.55 | 96.55 | 96.55 | 98.71 |
| **M5** (KNN) | 90.09 | 96.98 | 94.83 | 94.83 | 98.71 |

---

## 🏆 Key Findings

### Best Sampling Technique for Each Model

- **M1 (Logistic Regression):** Sampling5 (Tomek Links) - **98.71%**
- **M2 (Decision Tree):** Sampling2, Sampling3, Sampling4 - **98.28%**
- **M3 (Random Forest):** Sampling2, Sampling4 - **99.14%**
- **M4 (SVM):** Sampling5 (Tomek Links) - **98.71%**
- **M5 (KNN):** Sampling5 (Tomek Links) - **98.71%**

### Best Model for Each Sampling Technique

- **Sampling1 (Under-Sampling):** M5 (KNN) - **90.09%**
- **Sampling2 (Over-Sampling):** M3 (Random Forest) - **99.14%**
- **Sampling3 (SMOTE):** M2 (Decision Tree) - **98.28%**
- **Sampling4 (ADASYN):** M3 (Random Forest) - **99.14%**
- **Sampling5 (Tomek Links):** M1, M4, M5 - **98.71%**

### Overall Best Combination

**Model:** M3 (Random Forest)  
**Sampling:** Sampling2 (Random Over-Sampling) / Sampling4 (ADASYN)  
**Accuracy:** **99.14%**

---

## 💡 Discussion and Conclusions

### 1. Impact of Sampling Techniques

- **Under-Sampling (Sampling1):** Generally performs poorly across all models, losing important information from the majority class.
- **Over-Sampling (Sampling2):** Shows excellent performance, especially with Random Forest (99.14%).
- **SMOTE (Sampling3):** Provides good balance, achieving 98.28% with Decision Tree.
- **ADASYN (Sampling4):** Performs similarly to SMOTE, achieving 99.14% with Random Forest.
- **Tomek Links (Sampling5):** Excellent for Logistic Regression, SVM, and KNN (98.71% each).

### 2. Model Performance Insights

- **Random Forest (M3)** consistently performs well across multiple sampling techniques, achieving the highest accuracy of 99.14%.
- **Decision Tree (M2)** benefits significantly from oversampling techniques (SMOTE, ADASYN).
- **Logistic Regression (M1)** performs best with Tomek Links, showing improvement from 69.40% to 98.71%.
- **SVM (M4)** and **KNN (M5)** both achieve their best results with Tomek Links (98.71%).

### 3. Recommendations

1. **For Credit Card Fraud Detection:**
   - Use **Random Forest** with **Random Over-Sampling** or **ADASYN** for best accuracy (99.14%).
   - Consider **Tomek Links** for Logistic Regression, SVM, and KNN models.

2. **Sampling Technique Selection:**
   - **SMOTE** and **ADASYN** are recommended for preserving data information while creating synthetic samples.
   - **Tomek Links** is effective for cleaning decision boundaries without losing much data.
   - **Random Under-Sampling** should be avoided as it loses valuable information.

3. **Model Selection:**
   - **Random Forest** is the most robust model for this imbalanced dataset.
   - **Tree-based models** (Decision Tree, Random Forest) handle imbalanced data better than linear models.

---

## 🛠️ Implementation Details

### Technologies Used

- **Python 3.x**
- **Libraries:**
  - `pandas` - Data manipulation
  - `numpy` - Numerical computations
  - `scikit-learn` - Machine learning models
  - `imbalanced-learn` - Sampling techniques
  - `matplotlib` & `seaborn` - Data visualization

### Project Structure

```
ass_anjulamam/
├── Creditcard_data.csv          # Original dataset
├── Sampling_Assignment.ipynb   # Main Jupyter notebook
├── sampling_results.csv         # Results in CSV format
└── README.md                    # This file
```

### How to Run

1. **Install Required Libraries:**
   ```bash
   pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn
   ```

2. **Open Jupyter Notebook:**
   ```bash
   jupyter notebook Sampling_Assignment.ipynb
   ```

3. **Run All Cells:**
   - Execute all cells sequentially
   - Results will be displayed in the notebook
   - Results will be saved to `sampling_results.csv`

---

## 📝 Assignment Tasks Completed

✅ **Task 1:** Dataset downloaded and loaded  
✅ **Task 2:** Dataset converted to balanced class dataset using 5 sampling techniques  
✅ **Task 3:** Five samples created using different sampling methods  
✅ **Task 4:** Five sampling techniques applied on five ML models (25 combinations)  
✅ **Task 5:** Determined which sampling technique gives higher accuracy on which model  
✅ **Task 6:** Results table generated matching assignment format  

---

## 📊 Visualizations Included

The notebook includes:
- Class distribution visualization (before and after sampling)
- Heatmap of accuracy results
- Grouped bar chart comparing all models
- Line chart showing performance trends
- Summary tables and statistics

---

## 🔗 GitHub Repository

[Add your GitHub repository link here]

---

## 📚 References

- Chawla, N. V., et al. (2002). "SMOTE: Synthetic Minority Over-sampling Technique." Journal of Artificial Intelligence Research.
- He, H., et al. (2008). "ADASYN: Adaptive synthetic sampling approach for imbalanced learning."
- Tomek, I. (1976). "Two modifications of CNN." IEEE Transactions on Systems, Man, and Cybernetics.

---

## 👤 Author

**Ishaan Sharma**  
**Roll Number:** 102303442

---

## 📄 License

This project is for educational purposes only.

---

**Date:** [Current Date]  
**Course:** Machine Learning / Data Science  
**Institution:** [Your Institution Name]
