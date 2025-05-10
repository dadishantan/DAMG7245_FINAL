# 🧠 Insurance Risk Prediction – Machine Learning Project

This repository contains a complete machine learning pipeline for predicting **insurance risk levels** based on structured data. It was developed as part of a data science coursework to demonstrate practical applications of classification models for the insurance industry.

---

## 📌 What This Project Does

The project forecasts an individual's **insurance risk category** (e.g., low, medium, or high risk) using demographic and policy-related features. It uses various machine learning models to help insurers:

- Assess the likelihood of future claims
- Make informed decisions about policy approval or premium rates
- Identify potentially high-risk applicants

---

## 🔍 Key Features Used for Prediction

Some of the typical parameters (features) used by the models include:

- **Age**
- **Gender**
- **Region**
- **Employment status**
- **Claim history**
- **Policy type**
- **Vehicle or asset-related details**

The dataset includes a training file (`train.csv`) and a test file (`test 2.csv`) to evaluate model performance on unseen data.

---

## 🧠 Machine Learning Workflow

The pipeline includes:

1. **Data Cleaning**  
   - Handling missing values  
   - Feature engineering

2. **Exploratory Data Analysis (EDA)**  
   - Visualizations to understand feature relationships  
   - Insights into risk distribution

3. **Modeling**  
   - Algorithms used: Random Forest, XGBoost, Gradient Boosting, Logistic Regression  
   - Hyperparameter tuning via GridSearchCV

4. **Evaluation**  
   - Metrics: Accuracy, F1 Score, ROC-AUC, Precision, Recall  
   - Model comparison and selection

5. **Interpretability**  
   - Model explainability using SHAP and LIME for transparency

---

## 🛠️ How to Run

### ✅ Requirements
- Python 3.7+
- Jupyter Notebook
- Required libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`, `shap`, `lime`, `mlxtend`

### ▶️ Run Notebooks
1. Launch Jupyter:
   ```bash
   jupyter notebook
   ```
2. Open and execute the following notebooks in order:
   - `Loading.ipynb`  
   - `EDA.ipynb`  
   - `MacheLearning.ipynb`

---

## 📁 Project Structure

```
DAMG7245_FINAL-main/
├── dataset/
│   ├── train.csv
│   └── test 2.csv
├── JupyterNotebooks/
│   ├── Loading.ipynb
│   ├── EDA.ipynb
│   └── MacheLearning.ipynb
├── Final Project Proposal.pdf
└── README.md
```

---

## 📚 Technologies Used

- Python
- Jupyter Notebook
- pandas, NumPy
- scikit-learn, XGBoost, LIME, SHAP
- Matplotlib, Seaborn

---

## ✅ License

MIT — free to use, adapt, and share for academic and research purposes.

---

## 🙋‍♂️ Author

**Shantan Dadi**  
Course: DAMG7245  
Project: Final Submission – Insurance Risk Prediction
