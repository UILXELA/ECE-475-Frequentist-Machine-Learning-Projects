#Zheng Liu
#ECE-475
#Mini Project 3

"""
The code may take a while to generate the graphs.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

def readFile(name):
    with open(name) as f:
        data = []
        next(f)
        for line in f:
            row = line.strip().split(',')
            row = [float(a) for a in row[2:]]
            data.append(row)
    np.random.shuffle(data)
    return np.array(data)

def data_proc(data):
    x = data[:,0:4]
    x = preprocessing.scale(x)
    x = np.concatenate([np.ones((len(x), 1)), x], axis = 1)
    y = data[:,5]
    return x,y

def SGD_reg(alpha,lamda,data):
    theta = np.zeros((data.shape[1]-1))
    oldtheta = -1
    j = 0
    max_it = 8000
    precision = 0.00001
    x,y =data_proc(data)
    
    L = []
    while np.linalg.norm(theta - oldtheta) > precision and j<max_it:
        i = np.random.randint(data.shape[0])
        oldtheta = theta
        y_est = 1/(1+np.exp(np.dot(x[i],(-theta)))) 
        y_vec = lambda theta: 1/(1+np.exp(np.dot(x,(-theta)))) 
        L.append(np.sum(np.log10(y_vec(theta))*y+np.log10(1 - y_vec(theta))*(1 - y)))
        theta = theta + alpha*(y[i]-y_est)*x[i] - lamda*theta
        j+=1
    return theta,L,j




def cross_val(data,theta):
    x,y = data_proc(data)
    y_vec = lambda theta: 1/(1+np.exp(np.dot(x,(-theta)))) 
    L = np.sum(np.log10(y_vec(theta))*y+np.log10(1 - y_vec(theta))*(1 - y))
    return L

def graphld(L,length):
    xx = np.linspace(1,length,num=length)
    L = np.array(L)
    plt.plot(xx,L)

def test(data,theta):
    x,y =data_proc(data)
    y_vec = lambda theta: np.rint(1/(1+np.exp(np.dot(x,(-theta)))))
    y_test = y_vec(theta)
    ratio = 1-np.sum(np.absolute(y_test-y))/y.shape
    return(float(ratio))

def main():
#Data    
    data = readFile('datatraining.txt')
    val_dat = readFile('dataval.txt')
    test_dat = readFile('datatest.txt')
    alpha = 0.01
#Unregularized
    theta_unreg, L_unreg, length_unreg = SGD_reg(alpha,0,data)
    

#Regularized
    for lamda in range(1,5):
        L_comp = -1000
        theta_reg = np.zeros((data.shape[1]-1))
        lamda /= 10000
        lamda_best = 0
        theta, L_,j = SGD_reg(alpha,lamda,data)
        L = cross_val(val_dat,theta)
        if L > L_comp:
            L_comp = L
            L_reg = L_
            theta_reg = theta
            lamda_best = lamda
            length_reg = j

#Test
    print("The regularized correct rate is ",test(test_dat, theta_reg)*100,"%\n")
    print("The unregularized correct rate is ",test(test_dat, theta_unreg)*100,"%\n")
    
#Comment
    print("\nThe unregularized one converges to a higher likelihood because the regularized one is penalized to get too high to prevent overfitting")

#Draw
    plt.subplot(211)        
    graphld(L_unreg,length_unreg)
    plt.title('Unregularized')
    plt.subplot(212)
    graphld(L_reg,length_reg)
    plt.title('Regularized')
    plt.show()


main()
