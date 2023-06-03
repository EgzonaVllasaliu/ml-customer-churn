[object Object]

# Telco Customer Churn Prediction

This project aims to analyze the Telco Customer Churn dataset to understand customer churn patterns, identify important features, and predict customer churn using machine learning algorithms. The dataset contains customer demographics, usage patterns, and transaction history.

## Dataset

The dataset is obtained from Kaggle's Telco Customer Churn dataset: https://www.kaggle.com/blastchar/telco-customer-churn.

### Features

The dataset contains the following features:

1. CustomerID: Unique identifier for each customer
2. Gender: Customer's gender (Male, Female)
3. SeniorCitizen: Whether the customer is a senior citizen or not (1, 0)
4. Partner: Whether the customer has a partner or not (Yes, No)
5. Dependents: Whether the customer has dependents or not (Yes, No)
6. Tenure: Number of months the customer has stayed with the company
7. PhoneService: Whether the customer has a phone service or not (Yes, No)
8. MultipleLines: Whether the customer has multiple lines or not (Yes, No, No phone service)
9. InternetService: Customer’s internet service provider (DSL, Fiber optic, No)
10. OnlineSecurity: Whether the customer has online security or not (Yes, No, No internet service)
11. OnlineBackup: Whether the customer has online backup or not (Yes, No, No internet service)
12. DeviceProtection: Whether the customer has device protection or not (Yes, No, No internet service)
13. TechSupport: Whether the customer has tech support or not (Yes, No, No internet service)
14. StreamingTV: Whether the customer has streaming TV or not (Yes, No, No internet service)
15. StreamingMovies: Whether the customer has streaming movies or not (Yes, No, No internet service)
16. Contract: The contract term of the customer (Month-to-month, One year, Two year)
17. PaperlessBilling: Whether the customer has paperless billing or not (Yes, No)
18. PaymentMethod: The customer’s payment method (Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic))
19. MonthlyCharges: The amount charged to the customer monthly
20. TotalCharges: The total amount charged to the customer
21. Churn: Whether the customer churned or not (Yes, No)

## Data Analysis

The data analysis process includes data exploration, visualization, and preprocessing to understand the dataset and identify important features.

## Machine Learning Algorithms

Several algorithms can be used to predict customer churn. Some possible algorithms to consider:

1. Logistic Regression
2. Support Vector Machine
3. Decision Trees
4. Random Forest
5. Gradient Boosting Machines (GBM)
6. XGBoost
7. Neural Networks

## Model Evaluation

To evaluate the performance of the different algorithms, we can use metrics such as:

1. Accuracy
2. Precision
3. Recall
4. F1-score
5. Area Under the Receiver Operating Characteristic curve (AUROC)

## Getting Started

Based on the dataset, we can identify some features that might not be very useful for the prediction of customer churn:
CustomerID: This is a unique identifier for each customer and doesn't carry any meaningful information for churn prediction.
As for the other features, they all seem to have some potential in helping to predict customer churn. However, during the data analysis and preprocessing phase, we should further explore the relationships between these features and customer churn to identify any multicollinearity or irrelevant features.

Regarding the best algorithm for this example, it's difficult to say without actually testing them on the dataset. However, some insights into why certain algorithms might be better suited for this problem:

Logistic Regression: It's a simple algorithm that can work well with binary classification problems like customer churn. It can also help identify the importance of each feature.
Random Forest: This ensemble method often performs well in various classification tasks. It can handle a mix of categorical and continuous features, which is the case in this dataset.
Gradient Boosting Machines (GBM) and XGBoost: These algorithms are known for their high performance in classification tasks. They can handle a mix of categorical and continuous features and automatically deal with missing data.
Neural Networks: With proper feature scaling and architecture, neural networks can be effective in handling complex relationships between features and the target variable.

### Prerequisites

Install python 3.x, Jupyter notebook

# This project aimed to analyze the factors affecting customer churn in a telecommunications company and to build a machine learning model to predict customer churn

The data was preprocessed, and the relevant features were identified. The following steps were taken during the analysis:

1. Visualized the correlation matrix to identify the relationships between the features.
2. Analyzed the distribution of numerical features and visualized their impact on churn.
3. Analyzed the distribution of categorical features and visualized their impact on churn.
4. Split the data into training and test sets, and applied SMOTE to balance the classes in the training set.
5. Trained a Random Forest Classifier using cross-validation and hyperparameter tuning.
6. Identified the important features using the feature importances of the best model.
7. Evaluated the model's performance using accuracy, precision, recall, F1 score, and AUC-ROC score.
8. Analyzed feature importances using permutation importance.

## Findings

* The correlation matrix showed some relationships between the features, such as tenure and TotalCharges, as well as MonthlyCharges and TotalCharges.
* The distribution of numerical features (tenure, MonthlyCharges, and TotalCharges) showed different patterns, with some skewed distributions.
* The distribution of categorical features showed varying relationships with churn, with some features like Contract and InternetService having a strong association with churn.
* The best Random Forest Classifier had an accuracy of 79.3%, precision of 64.4%, recall of 52.6%, F1 score of 57.8%, and an AUC-ROC score of 84.1%.
* The top three features identified by the built-in feature importances method were tenure, MonthlyCharges, and TotalCharges.
* The permutation importance method showed that tenure, Contract, and InternetService were the most important features, with TotalCharges having a relatively low importance.

The analysis revealed that features like tenure, Contract, and InternetService are the most important factors affecting customer churn. The model's performance could be improved by considering adding more relevant features to the dataset. The insights gained from this analysis can be used by the telecommunications company to improve their customer retention strategies and better understand the factors affecting customer churn.

## Acknowledgements

Kaggle, Telco
