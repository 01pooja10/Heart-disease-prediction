import numpy as np
import pandas as pd

d1=pd.get_dummies(df['cp'],drop_first=True,prefix='cp')
d2=pd.get_dummies(df['thal'],drop_first=True,prefix='thal')
d3=pd.get_dummies(df['slope'],drop_first=True,prefix='slope')
df=pd.concat([df,d1,d2,d3],axis=1)
df.drop(['cp','thal','slope'],axis=1,inplace=True)
df.head()

df['age'].min()
df['age'].max()
df['seniors'] = df['age'].map(lambda s: 1 if s >= 60 else 0)

X=df.drop('target',axis=1)
y=df['target']
xtrain,xtest,ytrain,ytest=train_test_split(X,y,test_size=0.2,random_state=42)

scale=StandardScaler()
xtrain=scale.fit_transform(xtrain)
xtest=scale.transform(xtest)
scores=[]
