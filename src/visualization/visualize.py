import seaborn as sns 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

# import data 
df = pd.read_pickle('../../data/processed/churn.pkl')

# Check correlation
df.corr(numeric_only=True)['Churn'].sort_values(ascending=False)





