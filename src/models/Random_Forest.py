import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 


# Load Data
df = pd.read_pickle('../../data/processed/churn.pkl')

# Drop extra columns
df = df.drop(['State','Area code'],axis=1)

# Split Features from Label 
df_dummies = pd.get_dummies(df, columns=['International plan','Voice mail plan'],
                            drop_first=True)
X = df_dummies.drop('Churn',axis=1)
y = df_dummies['Churn']

# Train-Test Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=101)

# ----------------------------- #
# Random Forest Classification
# ----------------------------- #

# Create model with base parameters
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state=101)
model.fit(X_train, y_train)
preds = model.predict(X_test)

# Model Evaluation 
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from scikitplot.metrics import plot_confusion_matrix

plot_confusion_matrix(y_test, preds)
print(classification_report(y_test, preds)) # Accuracy: 0.95 

# Feature Importance
feat_imp_df = pd.DataFrame(data=model.feature_importances_, index=X.columns,
                           columns=['Feature Importance'])\
                               .sort_values('Feature Importance', ascending=False)
                               
# ----------------------------- #
# Hyperparameter Tuning
# ----------------------------- #

from sklearn.model_selection import GridSearchCV

param_grid = {'n_estimators': [64, 100, 128, 200],
              'max_features':[3,5,10,'auto'],
              'bootstrap': [True, False],
              'oob_score': [True, False]}

grid_model = GridSearchCV(model,param_grid)
grid_model.fit(X_train, y_train)

# Best Parameters
grid_model.best_params_

# Re-fit model with best parameters
final_model = RandomForestClassifier(**grid_model.best_params_)
final_model.fit(X_train, y_train)

# Model Evaluation 
final_preds = final_model.predict(X_test)
plot_confusion_matrix(y_test, final_preds)

print(classification_report(y_test, final_preds)) # Accuracy: 0.95

# OOB Score 
final_model.oob_score_ # 0.95

# Feature Importance
feat_imp_final = pd.DataFrame(data=final_model.feature_importances_, 
                              index=X.columns,
                              columns=['Feature Importance'])\
                                  .sort_values('Feature Importance', ascending=False)
                                  

# Exploring best n_estimators 
errors = []
misclassifications = []

for n in range(1, 128):
    model = RandomForestClassifier(n_estimators=n, 
                                   max_features=5,
                                   bootstrap=True,
                                   oob_score=True,
                                   random_state=101)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    err = 1 - accuracy_score(y_test, preds)
    n_missed = np.sum(preds != y_test)
    
    errors.append(err)
    misclassifications.append(n_missed)
    
plt.plot(range(1,128), errors)
plt.plot(range(1,128), misclassifications)


    