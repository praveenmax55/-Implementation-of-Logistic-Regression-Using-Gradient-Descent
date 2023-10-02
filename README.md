# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the packages required.
2.Read the dataset.
3.Define X and Y array.
4.Define a function for costFunction,cost and gradient.
5.Define a function to plot the decision boundary and predict the Regression value.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Praveen D
RegisterNumber:212222240076  
*/
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data = np.loadtxt("ex2data1.txt",delimiter = ',')
X = data[:,[0,1]]
y = data[:,2]

X[:5]

y[:5]

plt.figure()
plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
plt.xlabel("Exam 1 Score")
plt.ylabel("Exam 2 Score")
plt.legend()
plt.show()

def sigmoid(z):
    return 1 / (1 + np.exp(-z))
    
plt.plot()
X_plot = np.linspace(-10,10,100)
plt.plot(X_plot,sigmoid(X_plot))
plt.show()

def costFunction(theta,X,y):
    h = sigmoid(np.dot(X,theta))
    J = -(np.dot(y, np.log(h)) + np.dot(1 - y,np.log(1-h))) / X.shape[0]
    grad = np.dot(X.T, h - y) / X.shape[0]
    return J,grad
    
X_train = np.hstack((np.ones((X.shape[0],1)), X))
theta = np.array([0,0,0])
J,grad = costFunction(theta,X_train,y)
print(J)
print(grad)

X_train = np.hstack((np.ones((X.shape[0],1)), X))
theta = np.array([-24,0.2,0.2])
J,grad = costFunction(theta,X_train,y)
print(J)
print(grad)

def cost(theta,X,y):
    h = sigmoid(np.dot(X,theta))
    J = -(np.dot(y, np.log(h)) + np.dot(1 - y, np.log(1 - h))) / X.shape[0]
    return J
def gradient(theta,X,y):
    h = sigmoid(np.dot(X,theta))
    grad = np.dot(X.T,h-y)/X.shape[0]
    return grad
X_train = np.hstack((np.ones((X.shape[0], 1)), X))
theta  = np.array([0,0,0])
res = optimize.minimize(fun=cost, x0=theta, args=(X_train, y),method='Newton-CG', jac=gradient)
print(res.fun)
print(res.x)

def plotDecisionBoundary(theta,X,y):
    x_min, x_max = X[:, 0].min() - 1,X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1,X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min,x_max, 0.1),np.arange(y_min,y_max, 0.1))
    X_plot = np.c_[xx.ravel(), yy.ravel()]
    X_plot = np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
    y_plot = np.dot(X_plot,theta).reshape(xx.shape)
    
    plt.figure()
    plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
    plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
    plt.contour(xx,yy,y_plot,levels=[0])
    plt.xlabel("Exam 1 Score")
    plt.ylabel("Exam 2 Score")
    plt.legend()
    plt.show()


plotDecisionBoundary(res.x,X,y)

prob = sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)

def predict(theta, X):
    X_train = np.hstack((np.ones((X.shape[0], 1)),X))
    prob = sigmoid(np.dot(X_train,theta))
    return (prob>=0.5).astype(int)
    
np.mean(predict(res.x,X) == y)
```

## Output:
![image](https://github.com/praveenmax55/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/113497509/6d1d20a6-0b19-44f0-8c65-f6ceb115f7fb)
![image](https://github.com/praveenmax55/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/113497509/9dfe49dd-09c0-4e43-816f-5d123ad9fa0d)
![image](https://github.com/praveenmax55/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/113497509/9d82b5f8-43f7-40f4-8deb-d6aae8763f0e)
![image](https://github.com/praveenmax55/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/113497509/731ab8b2-bcfa-4f95-a4e9-b0f3b49a93e6)
![image](https://github.com/praveenmax55/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/113497509/ecd780ef-6b98-4d61-99ca-148fa5349207)
![image](https://github.com/praveenmax55/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/113497509/f0bb4a94-1827-4266-98a6-eb6c185363f2)
![image](https://github.com/praveenmax55/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/113497509/009e4a2f-cda6-488c-aca6-20b8abbee72b)
![image](https://github.com/praveenmax55/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/113497509/a2a339a2-5f4a-4ef7-98ee-e50e10f8bb19)
![image](https://github.com/praveenmax55/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/113497509/e13ad14e-906d-4af1-b490-d7b2f5dd2b8f)
![image](https://github.com/praveenmax55/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/113497509/ef5a491a-f63b-4c07-82b8-8047ed9496e6)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

