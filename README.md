# KNN-Diabetes-Classifier
A simple K-Nearest Neighbour (KNN) classifier model that predicts diabetes diagnosis based on patient data found on kaggle.
The model predicts diabetes diagnosis and displays 1 if yes 0 if no

The goal of this model is to use the KNN algorithm to classify patients as diabetic or non-diabetic using features such as:
- Age
- BMI
- Physical activity per week
- Family history of diabetes
- Fasting glucose levels

Libraries used:
Pandas
NumPy 
Matplotlib 
scikit-learn

Features of Project: 

Exploratory Data Analysis (EDA):
- Checked for missing values
- Examined distributions of age and target variable

Feature Selection:
- Started with age, then expanded to include bmi, physical_activity_minutes_per_week, family_history_diabetes, and glucose_fasting.

Data Scaling:
- Used StandardScaler to standardize features for KNN.

Train-Test Split:
- 80% training, 20% testing; stratified by target to preserve label distribution.

Model Training:
- Tested different K values (1–10) to find optimal number of neighbors.

Evaluation:
- Accuracy
- Precision, Recall, F1-score

Visualization:
- Plotted accuracy vs. K to visualize model performance.
- Age distribution histogram

Kaggle link = https://www.kaggle.com/datasets/mohankrishnathalla/diabetes-health-indicators-dataset
By Elfa Alata
