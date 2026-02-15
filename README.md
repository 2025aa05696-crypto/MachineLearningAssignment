# 📘 Phishing Website Detection using Machine Learning

## Machine Learning – Assignment 2  
**M.Tech (AIML / DSE), BITS Pilani**

---

## a. Problem Statement

Phishing websites are fraudulent web pages designed to deceive users into revealing sensitive information such as login credentials, banking details, and personal data. With the rapid growth of online services, phishing attacks have become a major cybersecurity concern.

The objective of this project is to build and evaluate multiple machine learning classification models to automatically classify websites as legitimate or phishing using numerical features extracted from URLs. An interactive Streamlit web application is developed to demonstrate model performance and evaluation results.

---

## b. Dataset Description

- **Problem Type:** Binary Classification  
- **Total Instances:** ~11,000  
- **Number of Features:** 87 numerical features  
- **Target Variable:** `status`  
  - `legitimate` → 0  
  - `phishing` → 1  

The dataset consists of engineered numerical features derived from URLs and website properties such as URL length, presence of special characters, HTTPS usage, and query parameters. The raw URL column is removed during preprocessing, and only numerical features are used for model training. The dataset does not contain missing values.

---

## c. Machine Learning Models Used and Evaluation Metrics

All models were trained and evaluated on the same dataset using an **80–20 stratified train–test split** to ensure a fair comparison.

### Machine Learning Models Implemented

1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbors (KNN)  
4. Naive Bayes (Gaussian)  
5. Random Forest (Ensemble Model)  
6. XGBoost (Ensemble Model)

---

### Evaluation Metrics

The following evaluation metrics were calculated for each model:

- Accuracy  
- AUC (Area Under the ROC Curve)  
- Precision  
- Recall  
- F1 Score  
- Matthews Correlation Coefficient (MCC)

---

### Model Comparison Table

| ML Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---------|----------|-----|-----------|--------|----------|-----|
| Logistic Regression | ~0.93 | ~0.97 | ~0.94 | ~0.92 | ~0.93 | ~0.86 |
| Decision Tree | ~0.95 | ~0.95 | ~0.96 | ~0.94 | ~0.95 | ~0.90 |
| KNN | ~0.94 | ~0.96 | ~0.95 | ~0.93 | ~0.94 | ~0.88 |
| Naive Bayes | ~0.90 | ~0.94 | ~0.91 | ~0.89 | ~0.90 | ~0.80 |
| Random Forest | ~0.97 | ~0.99 | ~0.97 | ~0.97 | ~0.97 | ~0.94 |
| XGBoost | ~0.98 | ~0.99 | ~0.98 | ~0.98 | ~0.98 | ~0.96 |

---

### Model-wise Observations

| ML Model | Observation |
|---------|-------------|
| Logistic Regression | Conservative model with high precision but lower recall, resulting in missed phishing websites. |
| Decision Tree | Effectively captures non-linear patterns but may slightly overfit on training data. |
| KNN | Achieves good performance but is computationally expensive for large datasets. |
| Naive Bayes | Fast and efficient but limited by the assumption of feature independence. |
| Random Forest | Provides strong and balanced performance due to ensemble averaging. |
| XGBoost | Achieves the best overall performance by modeling complex feature interactions and reducing bias and variance. |

---

## Streamlit Web Application

An interactive Streamlit web application was developed to demonstrate the trained machine learning models. The application provides the following features:

- Upload external test datasets in CSV format  
- Select machine learning model from a dropdown menu  
- Display evaluation metrics including Accuracy, AUC, Precision, Recall, F1 Score, and MCC  
- Display confusion matrix  
- Display aligned classification report  

---

## Repository Structure

ML_Assignment_2/  
│── app.py  
│── requirements.txt  
│── README.md  
│  
└── model/  
    │── preprocessing.py  
    │── metrics.py  
    │── logistic_regression.py  
    │── decision_tree.py  
    │── knn.py  
    │── naive_bayes.py  
    │── random_forest.py  
    │── xgboost_model.py  

---

## How to Run the Project Locally

1. Create a virtual environment  
2. Install dependencies from `requirements.txt`  
3. Run the Streamlit application using `streamlit run app.py`  

---

## Deployment

The application is deployed using **Streamlit Community Cloud**. The live Streamlit application link is included in the final PDF submission as required by the assignment instructions.

---

## Conclusion

This project demonstrates a complete end-to-end machine learning workflow including data preprocessing, model training, evaluation, comparison, and deployment. Ensemble models such as Random Forest and XGBoost outperform simpler classifiers, highlighting their effectiveness for phishing website detection tasks.

---

## Author

**Name:** Prem Garg  
**BITS ID:** 2025aa05696  
**Email:** 2025aa05696@wilp.bits-pilani.ac.in  
**Program:** M.Tech (AIML / DSE)  
**Institute:** BITS Pilani (WILP)  
