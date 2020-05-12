import numpy as np
from sigmoid import *
import random


class Network(object):

    def __init__(self, shapes):
        """shape is a list of sizes of layers
        So shape [2, 3, 2] would have
        3 layers
        2 inputs, 2 outputs ans 3 hidden neurons"""

        self.numLayers = len(shapes)
        self.shapes = shapes
        self.biases = [np.random.randn(x, 1) for x in shapes[1:]]
        self.weights = [np.random.randn(y, x) for x,y in zip(shapes[:-1], shapes[1:])]
        # self.biases = [np.full((x, 1), 0.5) for x in shapes[1:]]
        # self.weights = [np.full((y, x), 0.5) for x,y in zip(shapes[:-1], shapes[1:])]

    def feedForward(self, activation):
        #return output of network
        #iterate over all layers and feed activation forward

        activation = np.array(activation)
        for b, w in zip(self.biases, self.weights):
           activation = sigmoid(np.matmul(w, activation) + b[0])
        return activation


    def StochasticGradientDescent(self, trainingData, epochs, batchSize, learningRate):


        n = len(trainingData)

        for e in range(epochs):
            random.shuffle(trainingData) 
            mini_batches = [trainingData[m:m+batchSize] for m in range(0, n, batchSize)]
            epochCost = 0
            for batch in mini_batches:
                epochCost += self.runBatch(batch, learningRate)
            
            print("Epoch %i finished with Cost: " % e, epochCost/epochs)


    def runBatch(self, batch, learningRate):

        b_gradient = [np.zeros(b.shape) for b in self.biases]
        w_gradient = [np.zeros(w.shape) for w in self.weights]

        batchCost = 0

        for b in batch:
            w_d_gradient, b_d_gradient, totalCost = self.backPropagation(b[0], b[1])
            # w_d_gradient, b_d_gradient, totalCost = self.backPropagation(np.array([0]), np.array([0]))

            batchCost += totalCost

            #add up the errors over the whole batch to use as gradient
            w_gradient = [wg+wdg for wg, wdg in zip(w_gradient, w_d_gradient)]
            b_gradient = [bg+bdg for bg, bdg in zip(b_gradient, b_d_gradient)]

        #adjust weights and biases
        self.weights = [w - (learningRate/len(batch)) * w_g
                        for w, w_g in zip(self.weights, w_gradient)]

        self.biases = [b - (learningRate/len(batch)) * w_b
                        for b, w_b in zip(self.biases, b_gradient)]

        return batchCost/len(batch) 
        

    
    def backPropagation(self, x, y):
        #return touple of lists of arrays/vectors of cost function partial derivatives to weights/biases 


        y = np.array(y)[np.newaxis].T 


        b_gradient = [np.zeros(b.shape) for b in self.biases]
        w_gradient = [np.zeros(w.shape) for w in self.weights]

        activation = np.array(x)[np.newaxis].T #first layer activation and used for iteration
        # activation = np.array(x)
        activations = [activation] #activation layer by layer 
        weightedSums = [] #list of all weighted sums layer by layer

        for b, w in zip(self.biases, self.weights):
            zs = np.matmul(w, activation) + b[0]
            weightedSums.append(zs)
            activation = sigmoid(zs)
            activations.append(activation)
        
        #get errors of last layer with cost function

        error = self.d_cost(activations[-1], y) * dsigmoid(weightedSums[-1])
        error = error
        b_gradient[-1] = error

        totalCost = np.absolute(error)


        w_gradient[-1] = np.matmul(error, activations[-2].T)
        #the newaxis thing already transposes the vector, dont ask me why


        #propagate error backwards through network

        for l in range(2, self.numLayers):

            #find error in current layer
            error = np.matmul(self.weights[-l+1].transpose(), error) * dsigmoid(weightedSums[-l])
            #error = error[np.newaxis]

            b_gradient[-l] = error


            w_gradient[-l] = np.matmul(error, np.array(activations[-l-1].transpose()))
            #w_gradient[-l] = (w_gradient[-l])[np.newaxis]

        return (w_gradient, b_gradient, totalCost)




    def d_cost(self, outputActivation, data):
        return (outputActivation - data) 
        


