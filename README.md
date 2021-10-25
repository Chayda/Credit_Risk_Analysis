# Credit_Risk_Analysis
Apply Machine Learning to solve for credit card risk using scikit-learn and imbalanced-learn Python Packages.

## Overview: explain purpose of analysis
Using a credit card dataset from LendingClub, a peer-to-peer lending services company, using multiple Machine Learning (ML) models to predict credit risk. The target is loan_status, which is categorized as ‘low_risk’ or ‘high_risk’. The model with the best performance will have a good balance of precision and recall, as indicated by a high F1 Score.

### Six ML models are created: 
1.	**RandomOverSampler** – algorithm to oversample data where instances of minority class are randomly selected and added to the training set.
2.	**SMOTE** – algorithm to oversample data where new instances are interpolated to create synthetic values that increase the size of the minority group.
3.	**ClusterCentroids** – algorithm to undersample data that identifies clusters of the majority class, then generates synthetic data points.
4.	**SMOTEENN** – algorithm that combines aspects of both over- and under-sampling. 
5.	**BalancedRandomForestClassifier** – ensemble method to sample the data and build several smaller, simpler decision trees to make predictions.
6.	**EasyEnsembleClassifier** – ensemble method that uses boosting technique to combine weak learners into a strong learner, where weak learners are used sequentially as one model learns from the mistakes of the previous model.


## Results: 

### RandomOverSampler 
<img width="800" alt="RandomOverSampler_results" src="https://user-images.githubusercontent.com/74624855/138618626-2ab150b2-5ef5-4020-b4ba-775d1708b97a.png">

- **Balanced Accuracy Score:** 0.6474
- **Precision:** 
  - High risk: 0.01
  - Low risk: 1.00
- **Recall:**
  - High risk: 0.69
  - Low risk: 0.60
- **F1 Score:**
  - High risk: 0.02
  - Low risk: 0.75

### SMOTE 
<img width="800" alt="SMOTE_results" src="https://user-images.githubusercontent.com/74624855/138618631-18071875-9767-4d80-9328-9cbc4335a1ae.png">

- **Balanced Accuracy Score:** 0.6622
- **Precision:**
  - High risk: 0.01
  - Low risk: 1.00
- **Recall:**
  - High risk: 0.63
  - Low risk: 0.69
- **F1 Score:**
  - High risk: 0.02
  - Low risk: 0.82

### ClusterCentroids
<img width="800" alt="ClusterCentroids_results" src="https://user-images.githubusercontent.com/74624855/138618639-d9e7247b-4cd7-4fda-b8ef-4a15793403e4.png">

- **Balanced Accuracy Score:** 0.5447
- **Precision:** 
  - High risk: 0.01
  - Low risk: 1.00
- **Recall:**
  - High risk: 0.69
  - Low risk: 0.40
- **F1 Score:**
  - High risk: 0.01
  - Low risk: 0.57

### SMOTEENN
<img width="800" alt="SMOTEENN_results" src="https://user-images.githubusercontent.com/74624855/138618641-d12456b6-235d-4cae-b5c5-5919c5afe87c.png">

- **Balanced Accuracy Score:** 0.6775
- **Precision:**
  - High risk: 0.01
  - Low risk: 1.00
- **Recall:**
  - High risk: 0.78
  - Low risk: 0.57
- **F1 Score:**
  - High risk: 0.02
  - Low risk: 0.73

### BalancedRandomForestClassifier 
<img width="800" alt="BalancedRandomForests_results" src="https://user-images.githubusercontent.com/74624855/138618827-207995c1-a5f6-4665-8a79-11662498d5c1.png">

- **Balanced Accuracy Score:** 0.7812
- **Precision:**
  - High risk: 0.03
  - Low risk: 1.00
- **Recall:**
  - High risk: 0.70
  - Low risk: 0.86
- **F1 Score:**
  - High risk: 0.06
  - Low risk: 0.92


### EasyEnsembleClassifier
<img width="800" alt="EasyEnsembleClassifier_results" src="https://user-images.githubusercontent.com/74624855/138618645-3a71a3ed-0171-4862-8148-a23760780912.png">

- **Balanced Accuracy Score:** 0.9316
- **Precision:**
  - High risk: 0.09
  - Low risk: 1.00
- **Recall:**
  - High risk: 0.92
  - Low risk: 0.94
- **F1 Score:**
  - High risk: 0.16
  - Low risk: 0.97


## Summary: 

Based on the results, the model that resulted in the best performance was the EasyEnsembleClassifier. Of the six models, this was the most accurate and robust, with an accuracy score of 93%, and an F1 score for the low-risk group of 0.97. The results for the over-sampled data vs the under-sampled data were similar, but the ClusterCentroids model performed the most poorly, with an accuracy score of only 54%. None of the models had a very high precision for the high-risk group.

Although the best model of the group was the EasyEnsembleClassifier, the sensitivity of this model (recall of 16%) wasn’t great at detecting high-risk. It wouldn’t be advisable to recommend this model to the bank to use to predict credit risk. It would be advisable to look at the BalancedRandomForestClassifier model and remove some of the features that aren’t relevant to predicting results for the target. Looking at the list generated in descending order in the ensemble code, the model could be improved by dropping the lower-ranked features and re-running the model. This model had a higher recall for the high-risk group (70%) than the EasyEnsembleClassifier. The bank should be advised to continue training for a model that has better results for predicting credit risk.
