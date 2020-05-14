from Network import *
import numpy as np
import random
import pickle

xor = [
    (np.array([0, 0]), np.array([0])),
    (np.array([1, 0]), np.array([1])),
    (np.array([0, 1]), np.array([1])),
    (np.array([1, 1]), np.array([0])),
]

training_data = []

for i in range(100):
    training_data.append(xor[random.randint(0,3)])

# print(training_data[0], training_data[0][0].shape, training_data[0][0].shape)
learningRate = 0.1

N = Network([2,2,1])

acceptableChange = 0.0002
lastCost = 0

while True:
    cost = N.StochasticGradientDescent(training_data, 50, 1, 0.1, training_data[:15])
    if cost < 0.01:
        break

    if abs(lastCost - cost) < acceptableChange:
        print("Network stopped improving")
        break
    
    lastCost = cost

for x in xor:
    print(x)
    print(N.feedForward(x[0]))

outfile = open('XORNN', 'wb')
pickle.dump(N, outfile)
outfile.close()

# for i in range(50000):
#     x = xor[random.randint(0,3)] 

#     # b_gradient = [np.zeros(b.shape) for b in N.biases]
#     # w_gradient = [np.zeros(w.shape) for w in N.weights]

#     w_d_gradient , b_d_gradient, err = N.backPropagation(x[0], x[1])

#     print(err)

#     w_gradient = w_d_gradient
#     b_gradient = b_d_gradient

#     # w_gradient = [wg+wdg for wg, wdg in zip(w_gradient, w_d_gradient)]
#     # b_gradient = [bg+bdg for bg, bdg in zip(b_gradient, b_d_gradient)]

#     N.weights = [w - (learningRate) * w_g
#                     for w, w_g in zip(N.weights, w_gradient)]

#     N.biases = [b - (learningRate) * w_b
#                     for b, w_b in zip(N.biases, b_gradient)]
                    

# for x in xor:
#     print(x)
#     print(N.feedForward(x[0]))