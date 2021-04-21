# non-linear SVM - Sheyda Zarandi
import numpy as np
from tools import getData
from sklearn.metrics import accuracy_score
import pickle
class SVM():
    """
    functions:
    get_digits: to categories train set into 10 groups corresponding to the 10 digits we have
    get_Q: to calculate projection we need to calculate Q which for linear SVM is Q=[labels * label.T] * [kernel(x,y)]
    get_kernel: to calculate rbf kernel as exp(-gamma ||x-y||^2)
    optimization: using PGD to optimize variables alpha
    train: for each 45 classifiers we save values of alpha
    test: for each test sample we check the 45 classifiers each would result in a vote (if labels=1 is more than labels= -1 we set the vote of
          current classifier as digit i (corresponding to label 1) otherwise the vote would be set to digit j. eventually the digit that appears the most in votes would be the
          predicted label for the current test sample.
    """
    def __init__(self,x_train,y_train,learning_rate,gamma,iterations):
        self.x_train = x_train
        self.y_train = y_train
        self.digits = self.get_digits()
        self.lr=learning_rate
        self.iterations=iterations
        self.gamma = gamma

    ## categorize training set based on their labels
    def get_digits(self):
        digits=[]
        for i in range(10):
            digits.append(np.array([self.x_train[k] for (k, v) in enumerate(self.y_train) if int(v) == i]))
        return digits
    ## calculate Q = [yy.T] * [kernel]
    def get_Q(self,kernel,y):
        y_y = np.dot(y, y.T)
        Q = np.multiply(y_y, kernel)
        return Q
    ## calculate rbf kernel
    def get_kernel(self,x,y=None):
        norm_x = (x ** 2).sum(axis=1)
        y, norm_y = x, norm_x
        if y is not None : norm_y = (y ** 2).sum(axis=1)
        kernel = np.dot(x, y.T)  # slow already parallel
        kernel *= -2
        kernel += norm_x.reshape(-1, 1)
        kernel += norm_y
        kernel *= -self.gamma
        np.exp(kernel, kernel)  # exponentiates in-place
        return kernel
    def optimize(self,Q,y):
        e = np.ones((len(Q), 1))
        alpha = np.zeros_like(e)
        iter = 0
        lnr=self.lr
        while iter < self.iterations:
            grad = np.matmul(Q, alpha) - e
            y_bar = np.dot(y.T, grad) / (np.linalg.norm(y) ** 2)
            grad_bar = grad - y_bar * y
            alpha -= (lnr * grad_bar)
            for i in range(len(alpha)):
                if alpha[i] < 0: alpha[i] = 0
            if iter % 15 == 0:
                lnr *= 0.1
            iter += 1
        return alpha
    def get_alpha(self,x,y):
        kernel = self.get_kernel(x)
        Q = self.get_Q(kernel, y)
        alpha = self.optimize(Q, y)
        return alpha
    def train(self):
        alphas = []
        for i in range(9):
            for j in range(i+1):
                x = np.concatenate((self.digits[i], self.digits[j]))
                y = np.concatenate((np.ones((len(self.digits[i]), 1)), (-1 * np.ones((len(self.digits[j]), 1)))))
                alpha = self.get_alpha(x,y)
                alphas.append(alpha)
        with open('alphas.pkl','wb') as f:
            pickle.dump(alphas,f)
        return alphas

    def test(self,alphas,x_test,y_test):
        predictions=[]
        for s in range(len(x_test)):
            votes=np.zeros((10,10))
            counter=0
            for i in range(9):
                for j in range(i+1,10):
                    x = np.concatenate((self.digits[i], self.digits[j]))
                    y = np.concatenate((np.ones((len(self.digits[i]), 1)), (-1 * np.ones((len(self.digits[j]), 1)))))
                    kernel = self.get_kernel(x_test[s],x)
                    alpha_y = np.multiply(alphas[counter], y)
                    f = np.dot(alpha_y,kernel)
                    f = np.sign(f)
                    num1=np.count_nonzero(f==1)
                    num2=np.count_nonzero(f==-1)
                    if num1>num2:
                        votes[i,j] =1
                    elif num2>=num1:
                        votes[j,i] =1
                    counter+=1
            final = np.sum(votes, axis=1)
            winner=np.argmax(final)
            predictions.append(winner)
        return predictions

x_train, y_train, x_test, y_test = getData()
x_train = np.array(x_train) / 255
x_test = np.array(x_test) / 255
y_train=y_train.astype(int)
y_test=y_test.astype(int)

#### parameters
learning_rate= 1e-4
gamma=2
iterations=1000
obj = SVM(x_train,y_train,learning_rate,gamma,iterations)
alphas = obj.train()
predictions = obj.test(alphas,x_test,y_test)

accuracy = accuracy_score(y_test,predictions)
print('for learning rate',learning_rate,'gamma:',gamma,'iterations:',iterations,'accuracy is: ',accuracy)




