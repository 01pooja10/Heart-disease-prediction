import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

df=pd.read_csv('../input/heart-disease-uci/heart.csv')
df.head()

#LogisticRegression
clf1=LogisticRegression()
clf1.fit(xtrain,ytrain)
pred1=clf1.predict(xtest)
s1=accuracy_score(ytest,pred1)
scores.append(s1*100)
print(s1*100)

#RandomForestClassifier
clf2=RandomForestClassifier(max_depth=2,random_state=0)
clf2.fit(xtrain,ytrain)
pred2=clf2.predict(xtest)
s2=accuracy_score(ytest,pred2)
scores.append(s2*100)
print(s2*100)

#KNeighborsClassifier
clf3=KNeighborsClassifier()
clf3.fit(xtrain,ytrain)
pred3=clf3.predict(xtest)
s3=accuracy_score(ytest,pred3)
scores.append(s3*100)
print(s3*100)

#SVM
clf4=svm.SVC(kernel='rbf',C=1)
clf4.fit(xtrain,ytrain)
pred4=clf4.predict(xtest)
s4=accuracy_score(ytest,pred4)
scores.append(s4*100)
print(s4*100)

#DecisionTreeClassifier
clf5=DecisionTreeClassifier(max_depth=3,random_state=0)
clf5.fit(xtrain,ytrain)
pred5=clf5.predict(xtest)
s5=accuracy_score(ytest,pred5)
scores.append(s5*100)
print(s5*100)
