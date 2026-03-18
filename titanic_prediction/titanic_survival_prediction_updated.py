# Importing libraries

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import linear_model
from sklearn.linear_model import LogisticRegression

from sklearn import svm
from sklearn.svm import SVC, LinearSVC

from sklearn import ensemble
from sklearn.ensemble import RandomForestClassifier

from sklearn import neighbors
from sklearn.neighbors import KNeighborsClassifier

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import SGDClassifier

from sklearn import naive_bayes
from sklearn.naive_bayes import GaussianNB

from sklearn import metrics
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

from sklearn import model_selection
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, cross_val_predict

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

# Data selection

project_url = 'https://raw.githubusercontent.com/gimseng/99-ML-Learning-Projects/'
data_path = 'master/001/data/'
train=pd.read_csv(project_url+data_path+'train.csv')
test=pd.read_csv(project_url+data_path+'test.csv')

train.head()

test.head()

# Deleting columns not requiered for modelling

del train['PassengerId']
del train['Ticket']
del train['Fare']
del train['Cabin']
del train['Name']

train.head()

train.shape

train.describe()

train.isnull().sum()

del test['Ticket']
del test['Fare']
del test['Cabin']
del test['Name']

test.head()

test.isnull().sum()

test.describe()

def getNum(str):
    if str=='male':
        return '1'
    if str=='female':
        return '2'
train["Gender"]=train["Sex"].apply(getNum)
train.head()
test["Gender"]=test["Sex"].apply(getNum)
test.head()

del train['Sex']
del test['Sex']
train.head()
test.head()

train[['Pclass', 'Survived']].groupby(['Pclass']).mean().sort_values(by='Survived', ascending=True)

train[['Gender', 'Survived']].groupby(["Gender"]).mean().sort_values(by='Survived', ascending=True)

train[['SibSp', 'Survived']].groupby(['SibSp']).mean().sort_values(by='Survived', ascending=True)

train[['Parch', 'Survived']].groupby(['Parch']).mean().sort_values(by='Survived', ascending=True)

# Visualising the data

sns.countplot(x='Survived', data=train)

age_hist = sns.FacetGrid(train, col='Survived')
age_hist.map(plt.hist, 'Age')
age_hist.set_ylabels('Number')

# Number of people who survived over age and passenger class

pclass_age_grid = sns.FacetGrid(train, col='Survived', row='Pclass', height=2.0, aspect=1.6)
pclass_age_grid.map(plt.hist, 'Age', alpha=0.5, bins=20)
pclass_age_grid.add_legend()
pclass_age_grid.set_ylabels('Number')

# Mean survived age
mean_sur= train[train.Survived==1]['Age'].mean()
mean_sur

# Mean non survived age
mean_nsur=train[train['Survived']==0]['Age'].mean()
mean_nsur

data = [train, test]

for dataset in data:
    mean = train["Age"].mean()
    std = test["Age"].std()
    is_null = dataset["Age"].isnull().sum()
    
    rand_age = np.random.randint(mean - std, mean + std, size = is_null)
    
    age_slice = dataset["Age"].copy()
    age_slice[np.isnan(age_slice)] = rand_age
    dataset["Age"] = age_slice
    dataset["Age"] = train["Age"].astype(int)
    
train["Age"].isnull().sum()

# Renaming columns

train.rename(columns={'Gender' : 'Sex'}, inplace=True)

test.rename(columns={'Gender' : 'Sex'}, inplace=True)

train.dropna(inplace=True)

train.head()

train.info()

train['Family_Members']=train['Parch']+train['SibSp'] + 1
test['Family_Members']=test['Parch']+test['SibSp'] + 1

del train['SibSp']
del train['Parch']

del test['SibSp']
del test['Parch']

train.head()

# Grouping the age data

data=[train,test]

for dataset in data:
    dataset.loc[ dataset['Age'] <= 10, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 10) & (dataset['Age'] <= 20), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 20) & (dataset['Age'] <= 25), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 25) & (dataset['Age'] <= 30), 'Age'] = 3
    dataset.loc[(dataset['Age'] > 30) & (dataset['Age'] <= 37), 'Age'] = 4
    dataset.loc[(dataset['Age'] > 37) & (dataset['Age'] <= 45), 'Age'] = 5
    dataset.loc[ dataset['Age'] > 45 , 'Age'] = 6

train['Age'].value_counts()

train.head()

test.head()

data=[train, test]

for dataset in data:
    dataset['Embarked']=dataset['Embarked'].map({'C':0, 'S':1, 'Q':2}).astype(int)
    
train.head()

train[['Age', 'Survived']].groupby(['Age']).mean().sort_values(by='Survived', ascending=True)

train[['Embarked', 'Survived']].groupby(['Embarked']).mean().sort_values(by='Survived', ascending=True)

train[['Pclass', 'Survived']].groupby(['Pclass']).mean().sort_values(by='Survived', ascending=True)

sns.heatmap(train.corr(), annot=True)

sns.heatmap(test.corr(), annot=True)

test

# Building models for prediction

X_train= train.drop(['Survived'], axis=1)
y_train= train['Survived']

X_test=test.drop('PassengerId', axis=1).copy()
X_test.shape

# Logistic Regression

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

y_pred = log_reg.predict(X_test)
print(y_pred)

round(np.mean(y_pred), 2)

log_rec_acc = round(log_reg.score(X_train, y_train)*100, 2)
print(log_rec_acc,'%')

#Find the feature correlation

df_coeff= pd.DataFrame(train.columns.delete(0))
df_coeff.columns = ['Feature']
df_coeff['Correlation'] = pd.Series(log_reg.coef_[0])

df_coeff.sort_values(by='Correlation', ascending=False)

# Cross Validation

kf = KFold(n_splits = 10)
scores= cross_val_score(log_reg, X_train, y_train, cv = kf, scoring='accuracy')

mean_acc_log = scores.mean()*100

print('Scores', scores*100,'%')
print('Mean', mean_acc_log,'%')
print('Standard Deviation: ', scores.std()*100, '%\n')

pred= cross_val_predict(log_reg, X_train, y_train, cv=kf)
print('Confusion Matrix: \n', confusion_matrix(y_train, pred), '\n')

print("Precision: ", round(precision_score(y_train, pred)*100, 2),'%')
print("Recall: ", round(recall_score(y_train, pred)*100, 2), '%')
print('F1 Score: ', round(f1_score(y_train, pred)*100, 2),'%')

# Support Vector Machine

svc = SVC()
svc.fit(X_train, y_train)

y_pred = svc.predict(X_test)
print(y_pred)
y_pred.shape

round(np.mean(y_pred), 2)

# Checking SVM accuracy

svc_acc = round(svc.score(X_train, y_train)*100, 2)
print(svc_acc, '%')

# Print CV accuracy estimate
scores= cross_val_score(SVC(), X_train, y_train, cv=kf, scoring='accuracy')

mean_acc_svc = scores.mean()*100

print('Scores: ', scores*100, '%')
print('Mean: ', mean_acc_svc, '%')
print('Standard Deviation: ', scores.std()*100, '%\n')

pred= cross_val_predict(svc, X_train, y_train, cv=kf)
print('Confusion Matrix: \n', confusion_matrix(y_train, pred), '\n')

print("Precision: ", round(precision_score(y_train, pred)*100, 2),'%')
print("Recall: ", round(recall_score(y_train, pred)*100, 2), '%')
print('F1 Score: ', round(f1_score(y_train, pred)*100, 2), '%')

# K-Nearest Neighbor (KNN)

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

y_pred= knn.predict(X_test)
print(y_pred)

round(np.mean(y_pred), 2)

# Checking KNN accuracy

knn_acc = round(knn.score(X_train, y_train)*100, 2)
print(knn_acc,'%')

# Print CV accuracy estimate

scores= cross_val_score(KNeighborsClassifier(), X_train, y_train, cv = kf, scoring='accuracy')

mean_acc_knn = scores.mean()*100

print('Scores: ', scores*100,'%')
print('Mean: ', mean_acc_knn,'%')
print('Standard Deviation: ', scores.std()*100,'%\n')

pred= cross_val_predict(knn, X_train, y_train, cv=kf)
print('Confusion Matrix: \n', confusion_matrix(y_train, pred), '\n')

print("Precision: ", round(precision_score(y_train, pred)*100, 2),'%')
print("Recall: ", round(recall_score(y_train, pred)*100, 2),'%')
print('F1 Score: ', round(f1_score(y_train, pred)*100, 2),'%')

# Decision Tree

dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)

y_pred = dtree.predict(X_test)
print(y_pred)

round(np.mean(y_pred), 2)

tree_acc = round(dtree.score(X_train, y_train)*100, 3)
print(tree_acc, '%')

scores= cross_val_score(DecisionTreeClassifier(), X_train, y_train, cv = kf, scoring='accuracy')

mean_acc_tree = scores.mean()*100

print('Scores: ', scores*100, '%')
print('Mean: ', mean_acc_tree, '%')
print('Standard Deviation: ', scores.std()*100,'%\n')

# Random Forest

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)

y_pred = random_forest.predict(X_test)
print(y_pred)

round(np.mean(y_pred),2)

# Checking accuracy

random_forest.score(X_train, y_train)
acc_random_forest = round(random_forest.score(X_train, y_train) * 100, 3)
print(acc_random_forest)

# Accuracy Estimates

scores= cross_val_score(RandomForestClassifier(), X_train, y_train, cv = kf, scoring='accuracy')

mean_acc_ranfor = scores.mean()*100

print('Scores: ', scores*100, '%')
print('Mean: ', mean_acc_ranfor, '%')
print('Standard Deviation: ', scores.std()*100, '%\n')

pred= cross_val_predict(random_forest, X_train, y_train, cv=kf)
print('Confusion Matrix: \n' ,confusion_matrix(y_train, pred), '\n')

print("Precision: ", round(precision_score(y_train, pred)*100, 2),'%')
print("Recall: ", round(recall_score(y_train, pred)*100, 2),'%')
print('F1 Score: ', round(f1_score(y_train, pred)*100, 2),'%')

# Stochastic Gradient Descent

sgd = SGDClassifier(max_iter=5, tol=None)
sgd.fit(X_train, y_train)

y_pred = sgd.predict(X_test)
print(y_pred)

round(np.mean(y_pred),2)

# Checking accuracy

sgd.score(X_train, y_train)
acc_sgd = round(sgd.score(X_train, y_train)*100, 2)
print(acc_sgd,'%')

# Accuracy Estimates

scores= cross_val_score(SGDClassifier(max_iter=5, tol=None), X_train, y_train, cv=kf, scoring='accuracy')

mean_acc_sgd = scores.mean()*100

print('Scores: ', scores*100, '%')
print('Mean: ', mean_acc_sgd, '%')
print('Standard Deviation: ', scores.std()*100, '%\n')

pred= cross_val_predict(sgd, X_train, y_train, cv=kf)
print('Confusion Matrix: \n',confusion_matrix(y_train, pred), '\n')

print("Precision: ", round(precision_score(y_train, pred)*100, 2),'%')
print("Recall: ", round(recall_score(y_train, pred)*100, 2),'%')
print('F1 Score: ', round(f1_score(y_train, pred)*100, 2),'%')


# Gaussian Naive Bayes

gaussian = GaussianNB()
gaussian.fit(X_train, y_train)

y_pred = gaussian.predict(X_test)
print(y_pred)

round(np.mean(y_pred),2)

# Checking accuracy

acc_gaussian = round(gaussian.score(X_train, y_train)*100, 2)
print(acc_gaussian)

# Accuracy estimates

scores= cross_val_score(GaussianNB() , X_train, y_train, cv=kf, scoring='accuracy')

mean_acc_gau = scores.mean()*100

print('Scores: ', scores*100, '%')
print('Mean: ', mean_acc_gau, '%')
print('Standard Deviation: ', scores.std()*100, '%\n')

pred= cross_val_predict(gaussian, X_train, y_train, cv=kf)
print('Confusion Matrix: \n', confusion_matrix(y_train, pred), '\n')

print("Precision: ", round(precision_score(y_train, pred)*100, 2),'%')
print("Recall: ", round(recall_score(y_train, pred)*100, 2),'%')
print('F1 Score: ', round(f1_score(y_train, pred)*100, 2),'%')


# Finding the best model

results = pd.DataFrame({
    'Model': ['Logistic Regression', 'Support Vector Machines', 'KNN', 'Decision Tree', 'Random Forest', 'Stochastix Gradient Decent', 'Gaussian Naive Bayes'],
    'Score': [log_rec_acc, svc_acc, knn_acc, tree_acc, acc_random_forest, acc_sgd, acc_gaussian],
    'Mean Score': [mean_acc_log, mean_acc_svc, mean_acc_knn, mean_acc_tree, mean_acc_ranfor, mean_acc_sgd, mean_acc_gau]})
df_result = results.sort_values(by='Mean Score', ascending=False)
df_result = df_result.set_index('Model')
df_result

importances=pd.DataFrame({"Features": X_train.columns, 'Importance':np.round(random_forest.feature_importances_, 3)})
importances = importances.sort_values('Importance', ascending=False).set_index('Features')
print(importances.head())
importances.plot.bar()

# Final chosen model
final_model = SVC()
final_model.fit(X_train, y_train)

final_y_pred = final_model.predict(X_test)

predicted_survivors = (final_y_pred == 1).sum()
predicted_non_survivors = (final_y_pred == 0).sum()
predicted_survivor_percentage = final_y_pred.mean() * 100

print("Final Model: Support Vector Machines")
print("Predicted survivors:", predicted_survivors)
print("Predicted non-survivors:", predicted_non_survivors)
print(f"Predicted survivor percentage:{predicted_survivor_percentage:.2f}%")

submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': final_y_pred
})

submission.to_csv('submission.csv', index=False)
print("Submission file created successfully")