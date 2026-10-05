# -*- coding: utf-8 -*-
"""Healthcare Stroke EDA, Cleaning, and Feature Engineering

Full solution answering all assignment TODOs.
Dataset: healthcare-dataset-stroke-data.csv
"""

import matplotlib.pyplot as plt
import numpy as np
# ==========================================
# TODO 0: Import the main libraries
# ==========================================
import pandas as pd
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid')

# ==========================================
# TODO 1: Load and preview the dataset
# ==========================================
DATA_PATH = 'healthcare-dataset-stroke-data.csv'
df = pd.read_csv(DATA_PATH)

print('--- First 5 rows ---')
print(df.head())
print('\n--- Last 5 rows ---')
print(df.tail())

# One row represents: An individual patient's medical and demographic record.
# Target variable: 'stroke' (binary: 0 = No, 1 = Yes).

# ==========================================
# TODO 2: Understand the dataset structure
# ==========================================
print(f'\nDataset Shape: {df.shape[0]} rows, {df.shape[1]} columns')
print('\nDataset Info:')
df.info()

# Classification:
# - Identifier: id
# - Categorical: gender, ever_married, work_type, Residence_type, smoking_status
# - Binary: hypertension, heart_disease, stroke (target)
# - Numerical: age, avg_glucose_level, bmi

# ==========================================
# TODO 3: Generate descriptive statistics
# ==========================================
print('\n--- Numerical Statistics ---')
print(df.describe().T)

print('\n--- Categorical Statistics ---')
cat_cols = [
    'gender',
    'ever_married',
    'work_type',
    'Residence_type',
    'smoking_status',
]
print(df[cat_cols].describe())

# ==========================================
# TODO 4: Check duplicates and the ID column
# ==========================================
print(f"\nFully duplicated rows: {df.duplicated().sum()}")
print(f"Duplicated IDs: {df['id'].duplicated().sum()}")
# id has no predictive power and will be excluded during modeling.

# ==========================================
# TODO 5 & 6: Categorical values & Missing values
# ==========================================
print('\n--- Missing Values Summary ---')
missing = pd.DataFrame({
    'missing_count': df.isna().sum(),
    'missing_percentage': (df.isna().mean() * 100).round(2),
})
print(missing)

print('\n--- Categorical Distributions ---')
for col in cat_cols:
  print(f'\n{col}:')
  print(df[col].value_counts())

# ==========================================
# TODO 7: Handle missing BMI values
# ==========================================
# Strategy: Median imputation is robust against right-skewed BMI outliers.
bmi_median = df['bmi'].median()
df['bmi_imputed'] = df['bmi'].fillna(bmi_median)
print(f'\nBMI Median Imputed value: {bmi_median:.2f}')

# ==========================================
# TODO 8: Handle unusual categorical values
# ==========================================
# 'Other' in gender only has 1 record -> drop row
df_clean = df[df['gender'] != 'Other'].copy()
df_clean['bmi'] = df_clean['bmi'].fillna(bmi_median)

# 'Unknown' in smoking_status represents 1,544 rows (30.2%), so it must be retained as a distinct category.

# ==========================================
# TODO 9: Validate numerical ranges & Outliers
# ==========================================
for col in ['age', 'avg_glucose_level', 'bmi']:
  q1 = df_clean[col].quantile(0.25)
  q3 = df_clean[col].quantile(0.75)
  iqr = q3 - q1
  lower = q1 - 1.5 * iqr
  upper = q3 + 1.5 * iqr
  outliers = df_clean[(df_clean[col] < lower) | (df_clean[col] > upper)]
  print(f'{col}: {len(outliers)} statistical outliers detected (IQR method).')

# ==========================================
# TODO 10: Final cleaned analytical dataset
# ==========================================
df_clean = df_clean.drop(columns=['bmi_imputed'], errors='ignore')
print(f'\nCleaned Dataset Shape: {df_clean.shape}')

# ==========================================
# TODO 11 to 15: Target & Group Distributions
# ==========================================
stroke_counts = df_clean['stroke'].value_counts()
print(f'\nStroke Target Distribution:\n{stroke_counts}')
print(
    f'Stroke Incident Rate: {(df_clean["stroke"].mean() * 100):.2f}% (Severe'
    ' Class Imbalance)'
)

# Stroke rate across categorical groups
for col in ['hypertension', 'heart_disease', 'smoking_status', 'work_type']:
  print(f'\nStroke Rate by {col}:')
  print((df_clean.groupby(col)['stroke'].mean() * 100).round(2))

# ==========================================
# TODO 16: Feature Engineering + Business Question 1
# ==========================================
# 1. Age Group
df_clean['age_group'] = pd.cut(
    df_clean['age'],
    bins=[0, 18, 40, 60, np.inf],
    labels=[
        'Child (0-17)',
        'Young Adult (18-39)',
        'Middle-aged (40-59)',
        'Senior (60+)',
    ],
    right=False,
)

# 2. BMI Category
df_clean['bmi_category'] = pd.cut(
    df_clean['bmi'],
    bins=[0, 18.5, 25, 30, np.inf],
    labels=['Underweight', 'Normal', 'Overweight', 'Obese'],
    right=False,
)

# 3. Glucose Category
df_clean['glucose_category'] = pd.cut(
    df_clean['avg_glucose_level'],
    bins=[0, 100, 140, np.inf],
    labels=['Normal (<100)', 'Elevated (100-140)', 'High (>140)'],
    right=False,
)

# Business Question 1: Highest observed stroke rate combination
combo = (
    df_clean.groupby(
        ['age_group', 'bmi_category', 'glucose_category'], observed=False
    )['stroke']
    .agg(
        total='count',
        stroke_cases='sum',
        stroke_rate=lambda x: round(x.mean() * 100, 2),
    )
    .reset_index()
    .sort_values(by='stroke_rate', ascending=False)
)
print('\n--- Top Combinations with Highest Stroke Rate (Min 10 patients) ---')
print(combo[combo['total'] >= 10].head())

# ==========================================
# TODO 17: Combined Risk Profile + Business Questions 2 & 3
# ==========================================
df_clean['risk_factor_count'] = (
    (df_clean['hypertension'] == 1).astype(int)
    + (df_clean['heart_disease'] == 1).astype(int)
    + (df_clean['avg_glucose_level'] >= 140).astype(int)
    + (df_clean['bmi'] >= 30).astype(int)
    + (df_clean['smoking_status'].isin(['smokes', 'formerly smoked'])).astype(
        int
    )
)

print('\n--- Stroke Rate by Risk Factor Count (BQ 2) ---')
bq2 = df_clean.groupby('risk_factor_count')['stroke'].agg(
    total='count', stroke_cases='sum', stroke_rate=lambda x: round(x.mean() * 100, 2)
)
print(bq2)

# High Risk Segment (3 or more risk factors)
high_risk = df_clean[df_clean['risk_factor_count'] >= 3]
print(f'\nTotal High-Risk individuals: {len(high_risk)}')
print('\nHigh Risk Distribution by Age Group (BQ 3):')
print(high_risk['age_group'].value_counts())

# Export processed dataset
df_clean.to_csv('healthcare_stroke_cleaned_engineered.csv', index=False)
print(
    "\nCleaned and engineered dataset saved to"
    " 'healthcare_stroke_cleaned_engineered.csv'"
)
