#PINTULAL 

#CS403 (Foundations of Machine Learning_) Assignment 1.

#-------------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import cross_validation


df = pd.read_csv('bikeDataTrainingUpload.csv',header=0)
df2 = pd.read_csv('TestX.csv',header=0)
#print("----------------------------------------------------------------------------------")
#df.info()

cols = ['casual','registered']
df = df.drop(cols,axis=1)
#print("\n")
#print("----------------------------------------------------------------------------------")
#df.info()


dummies = []
cols = ['season','yr','mnth','holiday','weekday','workingday','weathersit']

for col in cols:
	dummies.append(pd.get_dummies(df[col]))

titanic_dummies = pd.concat(dummies, axis=1)
 
df = pd.concat((df,titanic_dummies),axis=1)


df = df.drop(cols,axis=1)

#df.info()


#-----------------------------------------------------------------------------------

dummies2 = []


for col in cols:
	dummies2.append(pd.get_dummies(df2[col]))

titanic_dummies2 = pd.concat(dummies2, axis=1)
 
df2 = pd.concat((df2,titanic_dummies2),axis=1)


df2 = df2.drop(cols,axis=1)





#print("\n")
#print("----------------------------------------------------------------------------------")
#df.info()
#df2.info()




X = df.values
y = df['cnt'].values

X = np.delete(X,4,axis=1)

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.0,random_state=0)

clf = linear_model.LinearRegression() #Ridge(alpha=1.0) 

clf.fit(X_train,y_train)


#------------------------------------------------------------------------------------------
X2 = df2.values
pred = []
pred=clf.predict(X2)
#print("\n")
#print("----------------------------------------------------------------------------------")

pred = np.abs(pred)
raw_data = {'cnt':pred}
df3 = pd.DataFrame(raw_data, columns = ['cnt'])
df3.to_csv('output.csv',index_label='id')



