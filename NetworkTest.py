import numpy as np
import random
from Network import Network
from sigmoid import *

for i in range(1,10):
    for j in range(1,10):
        for k in range(1,10):
            NN = Network([i,j,k])

            print("Testing network [ %i, %i, %i ]" % (i, j, k))

            #check weight matrices
            assert len(NN.weights) == 2
            assert NN.weights[0].shape == (j, i)
            assert NN.weights[1].shape == (k, j)

            #check bias matrices
            assert len(NN.biases) == 2
            assert NN.biases[0].shape == (j, 1)
            assert NN.biases[1].shape == (k, 1)

            #check if feeding forward properly
            inputs = np.zeros((i ,1))
            outputs = NN.feedForward(inputs).shape
            print("Output: ", outputs)
            assert outputs == (k, 1)

            #check backprop
            inputs = np.zeros(i)
            outputs = np.zeros(k)
            w_g, b_g, c = NN.backPropagation(inputs, outputs)

            assert len(w_g) == 2
            # print("W0: ", w_g[0].shape)
            # print("W1: ", w_g[1].shape)
            assert w_g[0].shape == (j, i)
            assert w_g[1].shape == (k, j)

            assert len(b_g) == 2
            # print("B0: ",b_g[0].shape)
            # print("B1: ",b_g[1].shape)
            assert b_g[0].shape == (j, 1)
            assert b_g[1].shape == (k, 1)


            test = (np.zeros((i)), np.zeros((k)))
            # inputs = [0 for i in range()]
            # test = ([0], [0])
            # print("test ", test[0].shape, test[1].shape)
            batch = []

            for n in range(10):
                batch.append(test)
                
            NN.runBatch(batch, 1)
            #check weight matrices
            # print("W0: ", NN.weights[0].shape, " should be ", (j, i))
            # print("W1: ", NN.weights[1].shape)
            assert NN.weights[0].shape == (j, i)
            assert NN.weights[1].shape == (k, j)

            #check bias matrices
            assert NN.biases[0].shape == (j, 1)
            assert NN.biases[1].shape == (k, 1)