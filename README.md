# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

## Output:
![Screenshot 2024-02-20 111117](https://github.com/Jawaharraj27/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/139842416/27eb10c0-e155-4462-bd9d-a1b02d4c1f8f)
![Screenshot 2024-02-20 111154](https://github.com/Jawaharraj27/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/139842416/ce8074dd-5209-448b-a7ca-157d1cfb6ad1)
![Screenshot 2024-02-20 111117](https://github.com/Jawaharraj27/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/139842416/fc9591af-e0cd-4706-acc9-51d0f48dbb5e)
![Screenshot 2024-02-20 111117](https://github.com/Jawaharraj27/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/139842416/9846505b-6b3d-4274-8b3d-ce8549824a52)
![Screenshot 2024-02-20 111117](https://github.com/Jawaharraj27/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/139842416/deef35e6-a566-4d4a-b92b-d5fdd9e070d8)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
