# 🚀 Employee Attrition & Performance

A machine learning project that analyzes IBM's fictional HR dataset to understand and predict employee attrition, and to identify the factors that most influence whether an employee leaves the company.

## 📚 About the Dataset

The dataset (`WA_Fn-UseC_-HR-Employee-Attrition.xlsx`) is a fictional Human Resources dataset created by IBM Data Scientists to study employee attrition and performance. It contains **1,470 employee records** across **35 features**, including demographics, compensation, job role, satisfaction scores, and tenure information. It helps answer business questions such as:

- Which employees are most likely to leave?
- What factors (pay, satisfaction, overtime, tenure, etc.) drive attrition?
- How can HR intervene early to reduce turnover?

The target variable is **`Attrition`** (Yes/No) — a binary classification problem.

## 🗂️ Project Workflow

The notebook (`Employee_Attrition___Performance_.ipynb`) follows an end-to-end ML pipeline:

1. **Import Libraries & Resources** – numpy, pandas, matplotlib, seaborn, scikit-learn.
2. **Load Dataset** – reads the Excel file directly from a GitHub raw URL.
3. **Data Preprocessing**
   - Moves the `Attrition` target column to the end of the DataFrame.
   - Inspects structure, data types, and summary statistics.
   - Checks unique values for all categorical columns.
   - Drops duplicate rows and constant/uninformative columns (`Over18`, `EmployeeCount`, `StandardHours`, `EmployeeNumber`).
   - Encodes the target (`Yes`/`No` → `1`/`0`).
   - Checks for missing values (none found).
4. **Train-Test Split** – 80/20 split with `stratify=y` to preserve the class imbalance ratio (~84% No / ~16% Yes) in both sets.
5. **Encoding Categorical Data** – One-Hot Encoding (`drop='first'`) applied to nominal columns: `BusinessTravel`, `Department`, `EducationField`, `Gender`, `JobRole`, `MaritalStatus`, `OverTime`.
6. **Exploratory Data Analysis (EDA)**
   - Correlation matrix and heatmap of numeric features.
   - Box plots to detect outliers in continuous features (Age, Income, Tenure, etc.).
   - Outlier handling via IQR-based **capping** (winsorization) rather than row removal.
7. **Feature Scaling** – Standardization (`StandardScaler`) applied to all features.
8. **Dimensionality Reduction (PCA)**
   - Explained-variance analysis to determine components needed for 90% variance retention (28 components).
   - PCA applied to reduce the feature space before modeling.
9. **Model Training** – Logistic Regression with `class_weight='balanced'` to counteract class imbalance.
10. **Model Evaluation**
    - Accuracy, classification report, and confusion matrix on the held-out test set.
    - 5-fold **Stratified Cross-Validation** using **ROC-AUC** as the scoring metric (more reliable for imbalanced classes than accuracy).
11. **Feature Importance** – PCA coefficients are mapped back to the original feature space to identify and visualize the top 15 features driving attrition risk.

## 📊 Results

| Metric | Score |
|---|---|
| Test Accuracy | 0.77 |
| Precision (Attrition = Yes) | 0.38 |
| Recall (Attrition = Yes) | 0.68 |
| F1-score (Attrition = Yes) | 0.48 |
| Mean 5-fold CV ROC-AUC | 0.829 (± 0.027) |

Because the dataset is imbalanced (~84% stayed vs. ~16% left), `class_weight='balanced'` and ROC-AUC were used to give a more meaningful picture of performance than raw accuracy alone — the model favors catching more true attrition cases (higher recall) at the cost of some false positives.

## 🛠️ Tech Stack

- Python 3
- pandas, numpy
- matplotlib, seaborn (visualization)
- scikit-learn (`OneHotEncoder`, `StandardScaler`, `PCA`, `LogisticRegression`, cross-validation, evaluation metrics)

## 📦 Requirements

```bash
pip install numpy pandas matplotlib seaborn scikit-learn openpyxl
```

## ▶️ How to Run

1. Clone this repository.
2. Install the dependencies listed above.
3. Open `Employee_Attrition___Performance_.ipynb` in Jupyter Notebook / JupyterLab / Google Colab.
4. Run all cells sequentially. The dataset is loaded directly from a GitHub raw URL, so no manual download is required.

## 📁 Repository Structure

```
.
├── Employee_Attrition___Performance_.ipynb   # Main analysis & modeling notebook
└── README.md                                  # Project documentation
```

## 🔍 Key Takeaways

- Overtime, monthly income, job level, and years at company are among the most influential predictors of attrition (see the feature importance chart in the notebook).
- Class imbalance is a central challenge in this dataset; handling it (via `class_weight='balanced'` and stratified splitting/CV) is essential for meaningful model evaluation.
- PCA was used as a dimensionality-reduction step before modeling, with the explained-variance plot showing how many components are needed to retain most of the information.

## 📄 License

This project uses a fictional dataset published by IBM for educational and analytical purposes. Feel free to fork and adapt this project for learning or portfolio use.
