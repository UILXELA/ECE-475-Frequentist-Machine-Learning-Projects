"""
Frequentist Machine Learning
Mini Project 2
Zheng Liu

"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import lasso_path
from sklearn import linear_model
from itertools import cycle
import sklearn
import matplotlib.pyplot as plt

df = pd.read_csv('airfoil_self_noise.dat', sep='\s+', header=0, skiprows=0)
#df = pd.read_dat("airfoil_self_noise.dat")
df = df.iloc[np.random.permutation(len(df))]
df = df.reset_index(drop=True)
#print(df)

row_train = (int)(df.shape[0]*0.8)
row_val = (int)(df.shape[0]*0.9)
row_test = (int)(df.shape[0]-row_val)
#print(row_train)


train_x = df.iloc[0:row_train,0:5]
train_x['constant'] = np.ones(row_train)
train_x = train_x.values

val_x = df.iloc[row_train:row_val,0:5]
val_x = val_x.apply(lambda x:x-np.mean(x), axis=0)
val_x['contant'] = np.ones(row_val-row_train)
val_x = np.array(val_x)

#print(val_x)

test_x = df.iloc[0:row_test,0:5]
test_x['constant'] = np.ones(row_test)
test_x = np.array(test_x)

train_y = df.iloc[0:row_train,5:].values
val_y = np.array(df.iloc[row_train:row_val,5:])
test_y = np.array(df.iloc[row_val:,5:])
#print(test_x)

#################################
#old linear regression
beta1 = np.linalg.inv(train_x.T @ train_x) @ train_x.T @ train_y
#print(beta1)

mse1 = np.mean((test_y - test_x @ beta1) ** 2)
mse1_train = np.mean((train_y - train_x @ beta1) ** 2)
print("MSE of regular linear regression on the test set is", mse1)
print("\nMSE of regular linear regression on the training set is", mse1_train)

#################################
#ridge regression
lmbdas = np.arange(0, 10, 0.01)
min_mse = 999999
best_lmbda = 0

for lmbda in lmbdas:
    beta2 = np.linalg.inv(train_x.T @ train_x + lmbda * np.eye(6)) @ train_x.T @ train_y
    mse_valid = np.mean((val_y - val_x @ beta2)**2)
    if mse_valid < min_mse:
        min_mse = mse_valid
        best_lmbda = lmbda

#print(min_mse)
#print(best_lmbda)

beta2 = beta2 = np.linalg.inv(train_x.T @ train_x + best_lmbda * np.eye(6)) @ train_x.T @ train_y
mse2 = np.mean((test_y - test_x @ beta2)**2)
mse2_train = np.mean((train_y - train_x @ beta2) ** 2)
#print(beta2)
print("\nMSE of ridge regression on the test set is", mse2)
print("\nMSE of ridge regression on the training set is", mse2_train)


#################################
#LASSO
alphas = np.arange(0.1, 10, 0.01)
min_mse2 = 999999
best_alpha = 0

for alpha in alphas:
    lasso = Lasso(alpha=alpha, fit_intercept=True, normalize=True)
    lasso.fit(train_x, train_y)
    pred_y_val = lasso.predict(val_x)
    mse_lasso = np.mean((val_y - pred_y_val) ** 2)
    if mse_valid < min_mse2:
        min_mse2 = mse_valid
        best_alpha = alpha

lasso = Lasso(alpha=best_alpha)
lasso.fit(train_x, train_y)
pred_y = lasso.predict(test_x)
pred_y_train = lasso.predict(train_x)
mse3 = np.mean((test_y - pred_y)**2)
mse3_train = np.mean((train_y - pred_y_train)**2)
#print(best_alpha)
print("\nMSE of Lasso on the test set is", mse3)
print("\nMSE of Lasso on the training set is", mse3_train)

##################################
#Plot
alphas, _, coefs = linear_model.lars_path(train_x, train_y.flatten(), method='lasso', verbose=True)

xx = np.sum(np.abs(coefs.T), axis=1)
xx /= xx[-1]

plt.plot(xx, coefs.T)
ymin, ymax = plt.ylim()
plt.vlines(xx, ymin, ymax, linestyle='dashed')
plt.xlabel('|coef| / max|coef|')
plt.ylabel('Coefficients')
plt.title('LASSO Path')
plt.axis('tight')
plt.show()

#Comments
print("Comparing all the MSEs, we can see that on the test set, Lasso < ridge < regular. It agrees with the general expectation (Lasso perform the best). The MSEs are very close because there are only five features, and no much regularization is done. THe MSEs for training set are significantly lower as expected except for Lasso. I don't know exactly why but I guess it is because of the regularization.")




