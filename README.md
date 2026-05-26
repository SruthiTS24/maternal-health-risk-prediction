# Maternal Health Risk Prediction Using Machine Learning

A machine learning project to predict maternal health risk levels (High/Low) 
using clinical and obstetric data with a complete preprocessing, EDA, 
and model evaluation pipeline.

## Objective
To develop and evaluate machine learning classification models that accurately 
predict maternal health risk levels based on clinical and obstetric parameters, 
enabling early identification of high-risk pregnancies and supporting timely 
medical intervention.

---
## Dataset
- **Name:** Maternal Health Risk Dataset
- **Source:** Mendeley Data
- **Link:** https://data.mendeley.com/datasets/p5w98dvbbk/1
- **Records:** 1205
- **Features:** 12 (11 input + 1 target)

| Feature | Description | Unit |
|---|---|---|
| Age | Age of pregnant woman | Years |
| SystolicBP | Upper blood pressure | mmHg |
| DiastolicBP | Lower blood pressure | mmHg |
| BloodSugar | Blood glucose level | mmol/L |
| BodyTemp | Body temperature | Fahrenheit |
| BMI | Body Mass Index | kg/m² |
| PreviousComplications | History of obstetric complications | 0/1 |
| PreexistingDiabetes | Preexisting diabetes status | 0/1 |
| GestationalDiabetes | Gestational diabetes status | 0/1 |
| MentalHealth | Mental health status | 0/1 |
| HeartRate | Resting heart rate | bpm |
| RiskLevel | Target — Maternal health risk | High/Low |

---
## Project Structure
`````
maternal-health-risk-prediction/
├── Maternal_Health_Risk_Prediction.ipynb  ← complete final notebook
├── Maternal_Risk_Preprocessing_EDA.ipynb  ← progress submission 1
├── Maternal_Risk_Model_Building_Evaluation.ipynb  ← progress submission 2
├── maternal_health_risk_dataset.csv
├── maternal_health_risk_model.pkl
├── scaler.pkl
└── README.md
`````
------
## Workflow

**1. Data Preprocessing**
- Column renaming for consistency
- Duplicate check and removal
- Missing value treatment using median imputation
- Outlier detection and handling using clinical bounds

**2. Feature Engineering**
- Label encoding of target variable

**3. Exploratory Data Analysis**
- Class distribution analysis
- Feature vs Risk Level boxplots
- Correlation heatmap
- Pairplot of key clinical features

**4. Data Preparation**
- Train test split (80/20)
- SMOTE resampling on training data only
- StandardScaler normalization

**5. Model Building**
- Five classification models trained and compared

**6. Model Evaluation**
- Accuracy, Precision, Recall, F1 Score
- Confusion Matrix
- ROC Curve and AUC Score
- 
- **8. Cross Validation**
- K-Fold cross validation (k=5) on best model (Logistic Regression) to assess generalization performance

**8. Hyperparameter Tuning**
- Optimized Logistic Regression using GridSearchCV with 5-fold cross-validation
- Selected best parameters based on validation accuracy

**9. Model Saving**
- Saved using joblib for future inference

**10. Unseen Data Testing**
- Model tested on a sample unseen patient record

-----

## Models Used

| Model | Reason for Selection |
|---|---|
| Logistic Regression | Linear baseline model for binary classification |
| Decision Tree | Interpretable rule-based model mimicking clinical decision making |
| Random Forest | Ensemble method robust to overfitting and class imbalance |
| XGBoost | Advanced gradient boosting for improved predictive performance |
| SVM | Kernel based classifier for comparison across algorithmic approaches |

-----

## Results

| Model | Accuracy | F1 Score | AUC |
|---|---|---|---|
| Logistic Regression | 97.86% | 0.98 | 0.9787 |
| Random Forest | 97.44% | 0.97 | 0.9784 |
| XGBoost | 97.44% | 0.97 | 0.9768 |
| SVM | 97.01% | 0.97 | 0.9698 |
| Decision Tree | 94.44% | 0.94 | 0.9432 |


**Best Model:** Logistic Regression — Accuracy 97.86%, AUC 0.9787

**Cross Validation:** Mean accuracy of 93.75% (k=5) confirming model generalization.

**Key Predictors:** BloodSugar, PreexistingDiabetes and GestationalDiabetes 
consistently identified as strongest predictors across Random Forest and 
XGBoost feature importance analysis

**Unseen Data Test:** Model predicted High risk with 100% confidence on 
a sample patient record with high risk clinical indicators.

-----

## Requirements

->pandas<br>
->numpy<br>
->matplotlib<br>
->seaborn<br>
->scikit-learn<br>
->xgboost<br>
->imbalanced-learn<br>
->joblib

------

## Conclusion
Logistic Regression achieved the best performance with 97.86% accuracy and 
AUC of 0.9787. All models exceeded 94% accuracy confirming that basic clinical 
parameters are highly predictive of maternal health risk levels. The final saved model was further validated using an unseen patient record containing high-risk clinical indicators and successfully classified the case as High Risk with 100% predicted probability, highlighting its potential usefulness for early risk identification and clinical decision support.

-----

## Author
**Name:** Sruthi T S<br>
**Organization:** Entri Elevate<br>
**Course:** Advanced Program in Data Science with Machine Learning 


