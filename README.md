# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
  1.Load the employee dataset using pandas.
  2. Convert the categorical salary column into numerical values using Label Encoding.
  3.Select input features and target variable (left). 
  4.Split the dataset into training and testing sets. 
  5.Train the Decision Tree Classifier using the entropy criterion and predict test results. 
  6.Calculate model accuracy and predict whether a new employee will leave or not..

## Program:
```

#Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
#Developed by: SUVETHA K M S
#RegisterNumber:212225230278


import pandas as pd
data = pd.read_csv('Employee.csv')
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["salary"] = le.fit_transform(data["salary"])
data.head()
X = data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
X.head()
y=data["left"]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(X_train,y_train)
y_pred=dt.predict(X_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
print(accuracy)

dt.predict([[0.5,0.8,9,260,6,0,1,2]])

```

## Output:
<img width="267" height="180" alt="image" src="https://github.com/user-attachments/assets/1fbc6ce2-390a-49d6-b329-767f81b21664" />



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
