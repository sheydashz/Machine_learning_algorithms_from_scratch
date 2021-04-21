import numpy as np
from tools import nn_dataset

##################### Get data ############################
X_train, y_train, X_val, y_val, X_test, y_test = nn_dataset()
################ Activation and Dense Layers ################################
class ReLU():
    def forward(self, input): return np.maximum(0, input)
    def backward(self, input, grad_output):
        grad = input > 0
        return grad_output *grad
class Sigmoid():
    def forward(self,z): return 1 / (1 + np.exp(-z))
    def backward(self,z,grad_received):
        grad = 1 / (1 + np.exp(-z))*(1-1 / (1 + np.exp(-z)))*grad_received
        return grad
class Dense():
    def __init__(self, input, output, l_rate=0.1):
        self.learning_rate = l_rate
        self.weights = np.random.normal(loc=0.0, scale=np.sqrt(2 / (input + output)),size=(input, output))
        self.biases = np.zeros(output)
    def forward(self, z):
        return np.dot(z, self.weights) + self.biases
    def backward(self, input, grad_output):
        grad = np.dot(grad_output, self.weights.T)
        grad_weights = np.dot(input.T, grad_output)
        grad_biases = grad_output.mean(axis=0) * input.shape[0]
        self.weights = self.weights - self.learning_rate * grad_weights
        self.biases = self.biases - self.learning_rate * grad_biases
        return grad

########## Cost Function ##############################
def cost_func(y_bar, y):
    estim = y_bar[np.arange(len(y_bar)), y]
    cost = - estim + np.log(np.sum(np.exp(y_bar), axis=-1))
    return cost
def grad_cost_func(y_bar, y):
    #one-hot
    one_hot = np.zeros_like(y_bar)
    one_hot[np.arange(len(y_bar)), y] = 1
    softmax = np.exp(y_bar) / np.exp(y_bar).sum(axis=-1, keepdims=True)
    return (- one_hot + softmax) / y_bar.shape[0]

########## Define Network Structure #####################
network = []
network.append(Dense(X_train.shape[1], 100))
network.append(ReLU())
network.append(Dense(100, 400))
network.append(ReLU())
network.append(Dense(400, 10))


################## Network Training Path ##########
def forward(network, x):
    activs = []
    input = x
    for l in network:
        z= l.forward(input)
        activs.append(z)
        input = z
    return activs
def predict(network, x):
    estimation = forward(network, x)[-1]
    return estimation.argmax(axis=-1)
def train(network, x, y):
    activs = forward(network, x)
    layer_inputs = [x] +activs
    y_bar =activs[-1]
    cost = cost_func(y_bar, y)
    grad = grad_cost_func(y_bar, y)
    for l in range(len(network))[::-1]:
        layer = network[l]
        grad = layer.backward(layer_inputs[l], grad)
    return np.mean(cost)
def batches(x,y,size):
    indices = np.random.permutation(len(x))
    for indx in range(0, len(y) - size + 1, size):
        yield x[indices[indx:indx + size]], y[indices[indx:indx + size]]

for epoch in range(4):
    for x,y in batches(X_train, y_train,size=32):
        train(network, x, y)
val_log=np.mean(predict(network, X_val) == y_val)

print("Val accuracy:", val_log)

###### Relu
# for layers: 4 - nodes: 100,400,10,10 - epoch: 5 --> accuracy:98%
