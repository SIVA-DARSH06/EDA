# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2: Load the dataset
df = pd.read_csv('train.csv')

# Step 3: Basic Information
print("Dataset Info:")
print(df.info())

print("\nFirst 5 Rows:")
print(df.head())

print("\nStatistical Summary:")
print(df.describe())

# Step 4: Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Step 5: Value counts for categorical columns (if any)
print("\nValue Counts:")
for col in df.columns:
    if df[col].dtype == 'object':
        print(f"\nColumn: {col}")
        print(df[col].value_counts())

# Step 6: Univariate Analysis
# Histograms
df.hist(bins=30, figsize=(15, 10), color='skyblue')
plt.suptitle("Histograms for all features")
plt.show()

# Boxplots
for col in df.select_dtypes(include=np.number).columns:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df[col], color='lightgreen')
    plt.title(f"Boxplot of {col}")
    plt.show()

# Step 7: Bivariate Analysis
# Pairplot
sns.pairplot(df)
plt.suptitle("Pairplot of features", y=1.02)
plt.show()

# Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")

