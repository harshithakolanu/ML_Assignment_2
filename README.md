# Machine Learning Assignment 2: Dry Bean Classification

## a. Problem Statement
Build and compare five multiclass classifiers that predict one of seven dry-bean varieties from 16 image-derived numerical features, then evaluate saved pipelines through Streamlit without retraining.

## b. Dataset Description
UCI Dry Bean dataset, dataset ID 602: approximately 13,611 instances, 16 numerical predictors, target `Class`, and seven categories.

## c. Dataset Source and Attribution
Official UCI page: https://archive.ics.uci.edu/dataset/602/dry+bean+dataset  
DOI: https://doi.org/10.24432/C50S4B  
Citation: Koklu, M. and Ozkan, I. A. (2020), *Multiclass Classification of Dry Beans Using Computer Vision and Machine Learning Techniques*, Computers and Electronics in Agriculture, 174, 105507.  
Kaggle manual alternative: https://www.kaggle.com/datasets/zeynepazl/dry-beans/data

## d. GitHub Repository Link
https://github.com/harshithakolanu/ML_Assignment_2

## e. Data Preprocessing
Duplicates are removed after reporting. Features and target are separated, followed by one fixed 80:20 stratified split with `random_state=42`. Median imputation and scaling are fitted only inside pipelines on training data. The Decision Tree skips scaling because it is scale-invariant.

## f. Models Used
Logistic Regression, Decision Tree, KNN, Gaussian Naive Bayes, and RBF Support Vector Machine. Logistic Regression is a linear probabilistic baseline; Decision Tree captures nonlinear rules; KNN uses local similarity; GaussianNB is a fast distributional baseline; SVM learns nonlinear margins.

## g. Model Evaluation Metrics
Accuracy, macro One-vs-Rest AUC, macro Precision, macro Recall, macro F1, and MCC.

## h. Model Comparison Table
| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Support Vector Machine (SVM) | 0.923957 | 0.994431 | 0.935927 | 0.934556 | 0.935202 | 0.908019 |
| Logistic Regression | 0.919528 | 0.994719 | 0.931085 | 0.930320 | 0.930551 | 0.902760 |
| K-Nearest Neighbors (KNN) | 0.913621 | 0.985409 | 0.926850 | 0.923723 | 0.925042 | 0.895555 |
| Decision Tree Classifier | 0.902916 | 0.959648 | 0.913787 | 0.914780 | 0.914046 | 0.882647 |
| Gaussian Naive Bayes | 0.897010 | 0.991143 | 0.908075 | 0.908248 | 0.907220 | 0.876171 |

## i. Model Performance Observations
- SVM achieved the highest Accuracy, Precision, Recall, F1 Score, and MCC.
- Logistic Regression achieved the highest AUC score of 0.994719.
- SVM provided the best overall balance between Precision and Recall.
- Logistic Regression also demonstrated strong generalization performance.
- KNN achieved competitive performance with an Accuracy of 91.36%.
- Decision Tree achieved lower overall performance than SVM, Logistic Regression, and KNN.
- Gaussian Naive Bayes produced the lowest Accuracy, F1 Score, and MCC among the five evaluated models.
- Considering multiple evaluation metrics rather than Accuracy alone, SVM was selected as the overall winning model.

## j. Overall Winning Model
The overall winning model is **Support Vector Machine (SVM)**.

SVM achieved an Accuracy of **92.40%**, Precision of **93.59%**, Recall of **93.46%**, F1 Score of **93.52%**, and MCC of **0.9080**.

Although Logistic Regression achieved the highest AUC score of **0.994719**, SVM was selected as the overall winner because it provided the strongest combined performance across Accuracy, Precision, Recall, F1 Score, and MCC.

## k. Streamlit Application Description
The app validates an uploaded labeled test CSV, loads saved Joblib pipelines, calculates six metrics, displays a confusion matrix and classification report, compares all models, and downloads predictions. It never calls `fit()`.

## l. Deployment Link
`TO BE ADDED AFTER LIVE DEPLOYMENT AND TESTING`

## m. Project Folder Structure
The repository contains configuration, modular `src/` code, notebook, data folders, model outputs, reports, figures, deployment guides, screenshot instructions, and a submission PDF template.

## n. Installation Instructions
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

## o. Execution Instructions
```bash
python run_training.py
jupyter notebook notebooks/ML_Assignment_2.ipynb
streamlit run app.py
```
For Kaggle, download the XLSX/CSV manually, place it in `data/raw/`, and switch the clearly marked notebook source block.

## p. Conclusion
This project implemented a complete multiclass Machine Learning classification workflow using the UCI Dry Bean dataset. The original dataset contained 13,611 instances, 16 numerical input features, seven target classes, no missing values, and 68 duplicate records identified during data validation.

All five mandatory classification algorithms were trained using the same stratified 80:20 train-test split and evaluated using Accuracy, macro One-vs-Rest AUC, macro Precision, macro Recall, macro F1 Score, and Matthews Correlation Coefficient.

Among the five evaluated models, Support Vector Machine (SVM) achieved the strongest overall performance, with an Accuracy of 92.40%, F1 Score of 93.52%, and MCC of 0.9080. Logistic Regression achieved the highest AUC score of 0.994719.

Therefore, based on multiple evaluation measures rather than Accuracy alone, SVM was selected as the overall winning model. The trained models were serialized using Joblib and integrated into a Streamlit application that performs prediction and evaluation on uploaded test data without retraining the models.
