import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 


# Load Data
df = pd.read_pickle('../../data/processed/churn.pkl')

# ----------------------------- #
# Data Preprocessing 
# ----------------------------- #

# Drop extra columns 
df1 = df.drop(['State','Area code'],axis=1)

# Convert categorical variables to numerical variables
df_dummies = pd.get_dummies(df1, columns=['International plan','Voice mail plan'], drop_first=True)

#-----------------------------#
# Logistic Regression 
#-----------------------------#

# Train Test Split 
from sklearn.model_selection import train_test_split

X = df_dummies.drop('Churn',axis=1)
y = df_dummies['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2,
                                                    random_state=101)

# Scale and fit Data 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)

# Fit Model 
from sklearn.linear_model import LogisticRegression

log_model = LogisticRegression()
log_model.fit(scaled_X_train, y_train)

# Predictions 
y_pred = log_model.predict(scaled_X_test)

# Extract coefficients 
coef = log_model.coef_ 
columns = X.columns
coef_df = pd.DataFrame(coef, columns=columns).T
coef_df.columns = ['Log Coefficient']
coef_df_sorted = coef_df.sort_values('Log Coefficient', ascending=False)

# Model Evaluation 
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
y_probas = accuracy_score(y_test, y_pred) # 0.86 

# Confusion Matrix
import scikitplot as skplt
conf_matrix = confusion_matrix(y_test, y_pred)
skplt.metrics.plot_confusion_matrix(y_test, y_pred)

# Classification Report
print(classification_report(y_test, y_pred))

