import numpy as np

# Funcion de activacion: f(x) = 1 / 1 + e^(-x))
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def feedforward(self, inputs):
        # Pesa los valores de entrada, agrega tendencia de desviacion y luego usa la funcion de activacion
        total = np.dot(self.weights, inputs) + self.bias

        return sigmoid(total)

weights = np.array([0, 1]) # w sub 1 = 0, w sub 2 = 1
bias = 4 # b = 4
n = Neuron(weights, bias)
x = np.array([2, 3]) # x sub 1 = 2, x sub 2 = 3

print(n.feedforward(x))

