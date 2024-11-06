import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns   

# Load Data 
df = pd.read_pickle('../../data/processed/churn.pkl')

# Drop extra columns 
df1 = df.drop(['State','Area code'],axis=1)

# Get dummy variables
df_dummies = pd.get_dummies(df1, columns=['International plan','Voice mail plan'], 
                            drop_first=True, dtype=int)

# Split the data
X = df_dummies.drop('Churn',axis=1)
y = df_dummies['Churn']

# Train Test Split 
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=101, stratify=y)

# Scale the data 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)

# Instantiate models 
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

log_model = LogisticRegression()
tree = DecisionTreeClassifier()
rf = RandomForestClassifier()
knn = KNeighborsClassifier()

# Create tuples for the models
classifiers = [('Logistic Regression', log_model),
               ('Decision Tree', tree),
               ('Random Forest', rf),
               ('KNN', knn)]

# Iterate over the list of tuples containing the classifiers 
from sklearn.metrics import accuracy_score

for name, model in classifiers: 
    model.fit(scaled_X_train, y_train)
    y_pred = model.predict(scaled_X_test)
    print('{:s}:{:.3f}'.format(name, accuracy_score(y_test,y_pred)))
    
# ----------------------------- #
# Voting Classifier
# ----------------------------- #

from sklearn.ensemble import VotingClassifier

# Instantiate the Voting Classifier
voting = VotingClassifier(estimators=classifiers)
voting.fit(scaled_X_train, y_train)
voting_pred = voting.predict(scaled_X_test)
print('Voting Classifier:', accuracy_score(y_test, voting_pred))
    
# Evaluate Voting Classifier on confusion matrix 
from scikitplot.metrics import plot_confusion_matrix

plot_confusion_matrix(y_test, voting_pred)
plt.title('Voting Classifier')
plt.savefig('../../reports/figures/Voting_Classifier_Confusion_Matrix.png')
plt.show()



    