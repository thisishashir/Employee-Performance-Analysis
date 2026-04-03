# EMPLOYEE PERFORMANCE + SALARY PREDICTION + OUTLIER HANDLING

## 1. IMPORT LIBRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

## 2. LOAD DATA

# IMPORTANT: change path for deployment
df = pd.read_csv('HR_Analytics.csv')

print("Dataset Preview:")
print(df.head())

## 3. DATA CLEANING

df.drop([
    'EmpID', 'EmployeeCount', 'EmployeeNumber',
    'Over18', 'StandardHours'
], axis=1, inplace=True, errors='ignore')

df = df.dropna()

## 4. PERFORMANCE ANALYSIS

# FIXED ERROR HERE (removed `4`)
df['Performance_Score'] = (
    df['PerformanceRating'] * 0.4 +
    df['JobInvolvement'] * 0.2 +
    df['JobSatisfaction'] * 0.2 +
    df['WorkLifeBalance'] * 0.2
)

# Categorize performance
def categorize(score):
    if score >= 3.5:
        return "High"
    elif score >= 2.5:
        return "Average"
    else:
        return "Low"

df['Performance_Category'] = df['Performance_Score'].apply(categorize)

print("\nPerformance Categories Count:")
print(df['Performance_Category'].value_counts())

# Visualization
plt.figure()
plt.hist(df['Performance_Score'])
plt.title("Performance Score Distribution")
plt.xlabel("Score")
plt.ylabel("Count")
plt.show()

## 5. SALARY VARIABLE

df['Salary'] = df['MonthlyIncome']

## 6. OUTLIER DETECTION

Q1 = df['Salary'].quantile(0.25)
Q3 = df['Salary'].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df['Outlier'] = df['Salary'].apply(
    lambda x: True if (x < lower_bound or x > upper_bound) else False
)

print("\nOutliers Detected:")
print(df['Outlier'].value_counts())

## 7. HANDLE OUTLIERS

plt.figure(figsize=(10, 6))
sns.boxplot(x=df['Salary'])
plt.title('Salary Distribution Before Outlier Handling')
plt.xlabel('Salary')
plt.show()

# Remove outliers
df = df[df['Outlier'] == False]

print("\nDataset shape after removing outliers:", df.shape)

plt.figure(figsize=(10, 6))
sns.boxplot(x=df['Salary'])
plt.title('Salary Distribution After Outlier Handling')
plt.xlabel('Salary')
plt.show()

## 8. DATA PREPARATION

le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

## 9. FEATURES & TARGET

feature_columns = [
    'Age', 'AgeGroup', 'Attrition', 'BusinessTravel', 'DailyRate', 'Department',
    'DistanceFromHome', 'Education', 'EducationField', 'EnvironmentSatisfaction', 'Gender',
    'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobRole', 'JobSatisfaction',
    'MaritalStatus', 'MonthlyRate', 'NumCompaniesWorked', 'OverTime', 'PercentSalaryHike',
    'PerformanceRating', 'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears',
    'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole',
    'YearsSinceLastPromotion', 'YearsWithCurrManager', 'Performance_Score'
]

X = df[feature_columns]
y = df['Salary']

# Scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

## 10. MODEL

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

## 11. EVALUATION

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n===== MODEL PERFORMANCE =====")
print("Mean Squared Error:", mse)
print("R2 Score:", r2)

## 12. VISUALIZATION

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'r--', lw=2
)
plt.title('Actual vs Predicted Salary')
plt.xlabel('Actual Salary')
plt.ylabel('Predicted Salary')
plt.grid(True)
plt.show()

## 13. FINAL INSIGHT

print("\n===== FINAL INSIGHTS =====")
print("- Performance affects salary")
print("- Outliers detected using IQR method")
print("- Outliers removed to improve accuracy")
print("- Linear Regression used")
