from mnistdata.pmnist.mnist.loader import MNIST
import pickle
import random
import numpy as np
from Network import *

mndata = None
class mnist_loader:

    def start(self): 
        self.createNetwork()
        return self.get_training_data()

    def get_training_data(self):
        print("Loading dataset")
        mndata = MNIST('mnistdata/samples')

        images, labels = mndata.load_training()

        print("Loaded data. Converting...")

        training_data = []

        for im, l in zip(images, labels):
            output = [1 if i == l else 0 for i in range(0,10)]
            training_data.append(( np.array(im)/255, np.array(output) ))

        print("Converted")
        return training_data

    def get_test_data(self):
        print("Loading test dataset")
        mndata = MNIST('mnistdata/samples')

        images, labels = mndata.load_testing()

        print("Loaded test data. Converting...")

        test_data = []

        for im, l in zip(images, labels):
            output = [1 if i == l else 0 for i in range(0,10)]
            test_data.append(( np.array(im)/255, np.array(output) ))

        print("Converted")
        return test_data

    def createNetwork(self):
        self.N = Network([784, 100, 10])

    def runEpoch(self, training_data, epochs, learningRate):

        cost =self.N.StochasticGradientDescent(training_data, epochs, 30, learningRate, training_data[:1000])

        print("Ended with cost: ", cost)
        print("Accuracy: ")
        self.test(training_data[:100])
        outfile = open('mnistNN', 'wb')
        pickle.dump(self.N, outfile)
        outfile.close()

    def test(self, test_data):
        correct = 0
        num = len(test_data)
        for t in test_data:

            out = network.feedForward(t[0])
            max = np.max(out)
            out = np.array([1 if max == i else 0 for i in out])

            if np.array_equal(out, t[1]):
                correct+=1

        print("%i / %i" % (correct, num))
# index = random.randint(0, len(training_data))
# print(mndata.display(training_data[index][0]))
# print(training_data[index][1])

# N = Network([784, 30, 10])


# N.StochasticGradientDescent(training_data, 30, 10, 0.5, training_data[:10])


# outfile = open('mnistNN', 'wb')
# pickle.dump(N, outfile)
# outfile.close()