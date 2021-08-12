import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import pickle
warnings.filterwarnings("ignore")
import math

dataset = pd.read_csv('train_data.csv')
# dataset=dataset.values
dataset.drop(dataset.columns[[0]], axis = 1, inplace = True)
dataset['Loan_Status'].replace('N', 0,inplace=True)
dataset['Loan_Status'].replace('Y', 1,inplace=True)
dataset['Gender'].replace('Male',1,inplace=True)
dataset['Gender'].replace('Female',0,inplace=True)
dataset['Married'].replace('No',0,inplace=True)
dataset['Married'].replace('Yes',1,inplace=True)
dataset['Dependents'].replace('3+',3,inplace=True)
dataset['Education'].replace('Graduate',1,inplace=True)
dataset['Education'].replace('Not Graduate',0,inplace=True)
dataset['Property_Area'].replace('Urban',1,inplace=True)
dataset['Property_Area'].replace('Semiurban',2,inplace=True)
dataset['Property_Area'].replace('Rural',3,inplace=True)
dataset['Self_Employed'].replace('Yes',1,inplace=True)
dataset['Self_Employed'].replace('No',0,inplace=True)
dataset['Gender'].fillna(dataset['Gender'].mode()[0], inplace=True)
dataset['Married'].fillna(dataset['Married'].mode()[0], inplace=True)
dataset['Dependents'].fillna(dataset['Dependents'].mode()[0], inplace=True)
dataset['Self_Employed'].fillna(dataset['Self_Employed'].mode()[0], inplace=True)
dataset['LoanAmount'].fillna(dataset['LoanAmount'].median(), inplace=True)
dataset['Loan_Amount_Term'].fillna(dataset['Loan_Amount_Term'].median(), inplace=True)
dataset['Credit_History'].fillna(dataset['Credit_History'].mode()[0], inplace=True)
# dataset=np.array(dataset)

X= np.array(dataset.iloc[:, :-1])
Y= np.array(dataset.iloc[:, -1])

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.15, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, Y_train)
# Y_pred = classifier.predict(X_test)

pickle.dump(classifier,open('model.pkl','wb'))
# model = pickle.load(open('model.pkl','rb'))
