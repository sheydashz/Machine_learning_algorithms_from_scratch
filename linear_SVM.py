# sheyda zarandi - linear SVM
import numpy as np
from tools import getData
import pickle
from sklearn.metrics import accuracy_score
class SVM():
    """
    functions:
    get_digits: to categories train set into 10 groups corresponding to the 10 digits we have
    get_Q: to calculate projection we need to calculate Q which for linear SVM is Q=[labels * label.T] * [data * data.T]
    optimization: using PGD to optimize lagrangian multipliers alpha
    train: for each 45 classifiers we have, we caluclate the line that separates pair of classe --> output: linear model for each classifier
    test: for each test sample we check the 45 classifiers each would result in a vote (if labels=1 is more than labels= -1 we set the vote of
          current classifier as digit i (corresponding to label 1) otherwise the vote would be set to digit j. eventually the digit that appears the most in votes would be the
          predicted label for the current test sample.
    """
    def __init__(self,x_train,y_train,learning_rate,iterations):
        self.x_train = x_train
        self.y_train = y_train
        self.C=1
        self.get_digits()
        self.lr=learning_rate
        self.iterations=iterations
    def get_digits(self):
        digits = []
        for i in range(10):
            digits.append(np.array([self.x_train[k] for (k, v) in enumerate(y_train) if int(v) == i]))
        self.digits = digits
    def get_Q(self,x,y):
        x_x = np.matmul(x, x.T)
        y_y = np.matmul(y, y.T)
        Q = np.multiply(y_y, x_x)
        return Q
    def optimize(self,y,Q):
        e = np.ones((len(Q), 1))
        alpha = np.ones((len(Q), 1))*1e-5
        lnr = self.lr
        itr =1
        while itr < self.iterations:
            grad = np.dot(Q, alpha) - e
            y_bar = np.dot(y.T, grad) / (np.linalg.norm(y) ** 2)
            grad_bar = grad - y_bar * y
            alpha -= (lnr * grad_bar)
            alpha[alpha < 0] = 0
            alpha[alpha>self.C] =self.C
            lnr *= 2/np.sqrt(15*itr)
            itr += 1
        return alpha
    def train(self):
        W_tot = []
        ## Train 45 classifiers
        for i in range(9):
            for j in range(i + 1, 10):
                x = np.concatenate((self.digits[i], self.digits[j]))
                y = np.concatenate((np.ones((len(self.digits[i]), 1)), (-1 * np.ones((len(self.digits[j]), 1)))))
                Q=self.get_Q(x,y)
                alpha = self.optimize(y,Q)
                u = np.multiply(alpha, y)
                w = np.dot(u.T, x)
                W_tot.append(w.reshape(785))
        with open('W.pkl','wb') as f:
            pickle.dump(W_tot,f)
        return W_tot
    def test(self,x_test,W):
        prediction =[]
        for sample in range(len(x_test)):
            votes = np.zeros((10, 10))
            pair = 0
            for i in range(9):
                for j in range(i + 1, 10):
                    line = np.dot(x_test[sample], W[pair].T)
                    if np.sign(line) == 1:
                        votes[i, j] = 1
                    elif np.sign(line) == -1:
                        votes[j, i] = 1
                    pair += 1
            final = np.sum(votes, axis=1)
            result = np.argmax(final)
            prediction.append(result)
        return prediction

########################
x_train, y_train, x_test, y_test = getData()
#normalize data
x_train = np.array(x_train) / 255
x_test = np.array(x_test) / 255
# add one to x --> to represent b in linear model wx+b
x_train = np.concatenate((np.ones((len(x_train),1)), x_train), axis=1)
x_test = np.concatenate((np.ones((len(x_test),1)), x_test), axis=1)
y_train=y_train.astype(int)
y_test=y_test.astype(int)
########################
LR = 1e-5
iterations = 180
obj = SVM(x_train,y_train,LR,iterations)
W = obj.train()
prediction = obj.test(x_test,W)
accuracy = accuracy_score(y_test.astype(int),prediction)
print(accuracy)
