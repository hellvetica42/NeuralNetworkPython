import numpy as np
import random as ran
from sigmoid import *

class Layer:

    weights = []
    weightGradient = []
    biasGradient = []
    biases = []

    activation = []
    weightedSum = []
    errors = []

    def __init__(self, numNeurons, numInputs, isTest=False, isOutput=False):
        #init necassary data
        if isTest:
            self.weights = np.array([[1,2,3], [3,4,5]])
            self.biases = np.array([1,2])
        else:
            self.weights = np.array(np.random.rand(numNeurons,numInputs))
            self.weightGradient = np.zeros(self.weights.shape)
            self.biases = np.array(np.random.rand(numNeurons))
            self.biasGradient = np.zeros(self.biases.shape)

    def resetGradients(self):
            self.weightGradient = np.zeros(self.weights.shape)
            self.biasGradient = np.zeros(self.biases.shape)

    def feedForward(self, lastLayerActivation):
        #WEIGHTS x INPUTS + BIAS
        z = np.add(np.matmul(self.weights, lastLayerActivation), self.biases)

        self.weightedSum = z

        #map activation to sigmoid
        f = lambda z: sigmoid(z)
        self.activation = f(z)
        return self.activation

    def getDataErrors(self, data):
        # (yi - ai)^2 square error
        # e = np.subtract(data, self.activation)
        # self.errors = np.multiply(e, e)

        f = lambda z: dsigmoid(z)
        dZ = f(self.weightedSum)

        self.errors = np.multiply(self.activation - data, dZ)

        return self.errors

    def getErrors(self, nextWeights, nextErrors):
        #WEIGHTS transposed x ERRORS of next layer
        we = np.matmul(np.transpose(nextWeights), nextErrors)

        #map z to dsigmoid
        f = lambda x: dsigmoid(x)
        dz = f(self.weightedSum)

        #WE * dz 
        self.errors = np.multiply(we, dz)
        return self.errors 

    def fit(self, lastActivation):

        #construct weight gradient matrix
        #dC/dWjk = ak * ej

        #this is actually self.errors x lastActivation.transposed but it outputs a scalar
        w_gradient = []

        #iterate over error nad over activation to output matrix of partial derivatives
        for e in self.errors:
            for c in lastActivation:
                w_gradient.append(e*c)
        
        w_gradient = np.array(w_gradient)
        w_gradient = w_gradient.reshape((self.errors.shape[0], lastActivation.shape[0]))

        self.weightGradient = self.weightGradient + w_gradient


        #construct bias gradient vector
        #bias gradient vector is the current error vector
        b_gradient = self.errors

        self.biasGradient = self.biasGradient + b_gradient

        return (self.weightGradient, self.biasGradient)

        # #fit weights
        # # W = W - learningRate * weightGradient
        # self.weights = self.weights - np.multiply(learningRate, weightGradient)

        # #fit biases
        # # B = B - learningRate * biasGradient
        # self.biases = np.subtract(self.biases, np.multiply(learningRate, biasGradient))




xor = [
    [[0, 0], 0],
    [[1, 0], 1],
    [[0, 1], 1],
    [[1, 1], 0],
]

inputLayer = Layer(2,0)
hiddenLayer = Layer(2,2)
outputLayer = Layer(1,2)

lr = 0.2
batchLen = 50

for i in range(1000):
    for miniBatch in range(batchLen):
        test = xor[ran.randint(0,3)]
        h = hiddenLayer.feedForward(np.array(test[0]))
        outputLayer.feedForward(h)

        outputLayer.getDataErrors(np.array(test[1]))
        hiddenLayer.getErrors(outputLayer.weights, outputLayer.errors)

        outputLayer.fit(hiddenLayer.activation)
        hiddenLayer.fit(np.array(test[0]))

    outputLayer.weights = outputLayer.weights - (lr / batchLen) * outputLayer.weightGradient
    outputLayer.biases = outputLayer.biases - (lr / batchLen) * outputLayer.biasGradient
    outputLayer.resetGradients()

    hiddenLayer.weights = hiddenLayer.weights - (lr / batchLen) * hiddenLayer.weightGradient
    hiddenLayer.biases = hiddenLayer.biases - (lr / batchLen) * hiddenLayer.biasGradient
    hiddenLayer.resetGradients()



for x in xor:
    print(x)
    h = hiddenLayer.feedForward(np.array(x[0]))
    print(outputLayer.feedForward(h))

# L = Layer(3, 4, True)

# nW = np.transpose(np.array([[5,4,3], [3,2,1]]))
# nE = np.array([3,2,1])
# nZ = np.array([2,2])

# print(L.feedForward(np.array([1, 2, 3])))
# print(L.getDataErrors(np.array([1, 2])))
# print(L.getErrors(nW, nE, nZ))

# lA = np.array([1, 10, 1000])

# print(L.fit(lA, 1))

