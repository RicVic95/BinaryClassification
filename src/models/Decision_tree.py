import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 

# Load Data 
df = pd.read_pickle('../../data/processed/churn.pkl')

# Drop extra columns 
df1 = df.drop(['State','Area code'],axis=1)

# Get dummy variables 
df_dummies = pd.get_dummies(df1, columns=['International plan','Voice mail plan'], 
                            drop_first=True)

# ----------------------------- #
# Decision Tree Classification 
# ----------------------------- #

# Split Data 
X = df_dummies.drop('Churn',axis=1)
y = df_dummies['Churn']

# Train-Test Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=101,
                                                    stratify=y)

# Create model with base parameters
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

base_pred = model.predict(X_test)

# Model Evaluation
from sklearn.metrics import classification_report, confusion_matrix
from scikitplot.metrics import plot_confusion_matrix

print(classification_report(y_test, base_pred)) # Accuracy: 0.91
plot_confusion_matrix(y_test, base_pred)

# Feature Importance
feat_imp_df = pd.DataFrame(data=model.feature_importances_, index=X.columns,
                           columns=['Feature Importance'])\
                               .sort_values('Feature Importance', ascending=False)

# Create Function to report model evaluation
def report_model(model):
    model_pred = model.predict(X_test)
    print(classification_report(y_test, model_pred))
    print('\n')
    plot_confusion_matrix(y_test, model_pred)
    
# Hyperparameter Tuning
from sklearn.model_selection import GridSearchCV

param_grid = {'criterion': ['gini', 'entropy'],
              'max_depth': [3,4,5,6,7,8],
              'max_leaf_nodes':[5,10,15,20]}  


grid_model = GridSearchCV(DecisionTreeClassifier(), param_grid)

grid_model.fit(X_train, y_train)

# Find best params
grid_model.best_params_

# Report Model 
report_model(grid_model.best_estimator_) # Accuracy: 0.94

# Feature Importance
feat_imp_final = pd.DataFrame(data=grid_model.best_estimator_.feature_importances_, 
                              index=X.columns,
                              columns=['Feature Importance'])\
                                  .sort_values('Feature Importance', ascending=False)