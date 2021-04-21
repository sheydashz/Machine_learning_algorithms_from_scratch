import numpy as np
####################### Get Data ###########################
from tools import getData
import numpy as np
x_train,y_train,x_test,y_test=getData()
x_train=x_train.astype(int)
y_train=y_train.astype(int)
x_test=x_test.astype(int)
y_test=y_test.astype(int)
extra_column=np.ones((len(x_train),1))
x_train_1=np.c_[extra_column, x_train].astype(int)/255
extra_column=np.ones((len(x_test),1))
x_test_1=np.c_[extra_column, x_test].astype(int)/255
n_feature=len(x_train[0])+1
y_train_new=np.zeros((10,60000))
for i in range(10):
    y_train_new[i]=list(map(int,np.array(y_train==i,dtype=bool)))
h=1

##################### Linear Regression ######################
def linear_regression(x_train, y_train, x_test, y_test):
    data = x_train.reshape(-1, 784)
    x = np.c_[np.ones(len(x_train)), data]
    j1 = np.linalg.pinv((x.T).dot(x)).dot(x.T)
    target= np.zeros((len(x_train), 10))
    w = np.zeros((784 + 1, 10))
    for i in range(10):
        for k in range(len(x_train)):
            if y_train[k] == i:
                target[k, i] = 1
        w[:,i] = j1.dot(target[:, i])
    x_test = x_test.reshape(-1, 784)
    x_test = np.c_[np.ones(len(x_test)), x_test]
    correct = 0
    for i in range(len(x_test)):
        pred_y = np.zeros((10))
        for k in range(10):
            pred_y[k] = x_test[i].dot(w[:, k])
        if np.argmax(pred_y) == y_test[i]:
            correct += 1
    accuracy = correct / len(x_test)
    print('Linear Regression Accuracy: ' ,accuracy)
linear_regression(x_train, y_train, x_test, y_test)
############# Logistic Regression Class #####################
#############################################################
class LR():
    def __init__(self,x):
        self.x=x #(60000,785)
        self.features=785
        self.N=len(x)
    def sigmoid(self,z):
        h=np.zeros_like(z)
        for i,v in enumerate(z):
            if v > -700:
                h[i] = 1 / (1 + np.exp(-v))
        return h
    def grads(self,theta_t):
        z = self.x @ theta_t
        h=self.sigmoid(z)
        # compute gradient
        j_0 = 1 / self.N * (self.x.transpose() @ (h - self.y))[0]
        j_1 = 1 / self.N * (self.x.transpose() @ (h - self.y))[1:]
        grad = np.vstack((j_0[:, np.newaxis], j_1))
        return grad
    def optimize(self,y):
        self.alpha=0.2
        self.y = np.array(y).reshape((60000, 1))
        theta_t = np.ones((785, 1))*1e-7
        counter=0
        while counter<150:
            grad = self.grads(theta_t)
            theta_t = theta_t - (self.alpha * grad)
            counter=counter+1
        return theta_t.reshape((785))
##############################################################################
###############################################################################
thetas=np.zeros((10,785))
obj=LR(x_train_1)
h=0
for i in range(10):
    print('learning classifier ',i,'...')
    thetas[i]=obj.optimize(y_train_new[i])
mtrx =thetas @ x_test_1.T
solution = np.argmax(mtrx, axis=0)
for i in range(len(x_test)):
    if solution[i]==y_test[i]:
        h+=1
accuracy=h/len(x_test)
print('Logistic regression accuracy is:',accuracy)
