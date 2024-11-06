import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns   

# Load Data
df1 = pd.read_csv('../../data/raw/churn-bigml-20.csv')
df2 = pd.read_csv('../../data/raw/churn-bigml-80.csv')

# Concatenate Data
df = pd.concat([df1, df2])

# ----------------------------- # 
# Data Preprocessing
# ----------------------------- #

# Check for null values
df.isnull().sum()  # No missing values

# Check for duplicates
df.duplicated().sum()  # No duplicates

# Check for column types 
df.dtypes

# Save Dataset
df.to_pickle('../../data/processed/churn.pkl')