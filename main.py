import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Employee Salary Prediction", layout="wide")

st.title("Employee Performance and Salary Prediction")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("HR_Analytics.csv")

df = load_data()

st.subheader("Dataset Preview")
st.dataframe(df.head())

# Data cleaning
df.drop([
    'EmpID', 'EmployeeCount', 'EmployeeNumber',
    'Over18', 'StandardHours'
], axis=1, inplace=True, errors='ignore')

df = df.dropna()

# Performance score
df['Performance_Score'] = (
    df['PerformanceRating'] * 0.4 +
    df['JobInvolvement'] * 0.2 +
    df['JobSatisfaction'] * 0.2 +
    df['WorkLifeBalance'] * 0.2
)

def categorize(score):
    if score >= 3.5:
        return "High"
    elif score >= 2.5:
        return "Average"
    else:
        return "Low"

df['Performance_Category'] = df['Performance_Score'].apply(categorize)

st.subheader("Performance Categories")
st.write(df['Performance_Category'].value_counts())

fig1, ax1 = plt.subplots()
ax1.hist(df['Performance_Score'])
ax1.set_title("Performance Score Distribution")
st.pyplot(fig1)

# Salary
df['Salary'] = df['MonthlyIncome']

# Outlier detection
Q1 = df['Salary'].quantile(0.25)
Q3 = df['Salary'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df['Outlier'] = df['Salary'].apply(
    lambda x: True if (x < lower_bound or x > upper_bound) else False
)

st.subheader("Outliers")
st.write(df['Outlier'].value_counts())

fig2, ax2 = plt.subplots()
sns.boxplot(x=df['Salary'], ax=ax2)
ax2.set_title("Before Outlier Removal")
st.pyplot(fig2)

# Remove outliers
df = df[df['Outlier'] == False]

st.write("Dataset shape after removing outliers:", df.shape)

fig3, ax3 = plt.subplots()
sns.boxplot(x=df['Salary'], ax=ax3)
ax3.set_title("After Outlier Removal")
st.pyplot(fig3)

# Encoding
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

# Features
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

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.subheader("Model Performance")
st.write(f"Mean Squared Error: {mse}")
st.write(f"R2 Score: {r2}")

fig4, ax4 = plt.subplots()
ax4.scatter(y_test, y_pred, alpha=0.6)
ax4.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'r--', lw=2
)
ax4.set_title("Actual vs Predicted Salary")
st.pyplot(fig4)

# Prediction input
st.subheader("Predict Salary")

input_data = {}
for col in feature_columns:
    input_data[col] = st.number_input(col, value=0.0)

input_df = pd.DataFrame([input_data])
input_scaled = scaler.transform(input_df)

if st.button("Predict"):
    prediction = model.predict(input_scaled)
    st.write(f"Predicted Salary: {prediction[0]:.2f}")

# Insights
st.subheader("Insights")
st.write(
    "Performance has a measurable impact on salary. "
    "Outliers were identified using the IQR method and removed to improve model reliability. "
    "A linear regression model was used for prediction."
)
