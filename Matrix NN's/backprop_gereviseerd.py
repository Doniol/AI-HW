# Gereviseerde versie van backprop.py  (zodanig dat ik (MV) hem zelf begrijp en kan uitleggen)

# meer over numpy array slicing and manipulating:
# https://machinelearningmastery.com/index-slice-reshape-numpy-arrays-machine-learning-python/

import numpy as np
from math import e
import random

random.seed(1)

def sigmoid(x):
    """Standard sigmoid; since it relies on ** to do computation, it broadcasts on vectors and matrices"""
    return 1 / (1 + (e**(-x)))

def derivative_sigmoid(x):
    """Expects input x to be already sigmoid-ed"""
    return x * (1 - x)

def tanh(x):
    """Standard tanh; since it relies on ** and * to do computation, it broadcasts on vectors and matrices"""
    return (e ** (2*x) - 1) / (e ** (2*x) + 1)

def derived_tanh(x):
    """Expects input x to already be tanh-ed."""
    return 1 - x*x

# NB: veel sneller zou natuurlijk zijn om 1x forward propagation toe te passen, en vervolgens tijdens de backprop-stap
# de resultaten daarvan op te halen, ipv ze steeds opnieuw te berekenen tot een bepaalde layer..
# maar goed, doordat we dat cachen nu weglaten blijft de code wat eenvoudiger. Vooruit dus, voor nu..
def forward(inputs,weights,function=sigmoid,layer=-1):
    """Function needed to calculate activation on a particular layer 
    (WITHOUT PREPENDING 1 as output, but with prepending 1 in its input).
    The weights array consists of a weight matrix for each layer.
    The first weight of each row of such a weight matrix is the bias weight.
    layer=-1 calculates all layers, thus provides the output of the network
    layer=0 returns the inputs
    any layer in between, returns the output vector of that particular (hidden) layer"""
    if layer == 0:
        return inputs
    elif layer == -1:
        layer = len(weights) - 1
    
    function = np.vectorize(function)
    for weight in weights:
        inputs = function(np.dot(weight, np.insert(inputs, 0, 1)))
    return inputs#TODO FIX DAT IE NIET ALLES DOET

    
def backprop(inputs, outputs, weights, function=sigmoid, derivative=derivative_sigmoid, eta=0.01):
    """
    Function to calculate deltas matrix based on gradient descent / backpropagation algorithm.
    Deltas matrix represents the changes that are needed to be performed to the weights (per layer) to
    improve the performance of the neural net.
    :param inputs: (numpy) array representing the input vector of the trainingsample.
    :param outputs:  (numpy) array representing the desired output vector from the training sample.
    :param weights:  list of numpy arrays (matrices) that represent the weights per layer.
    :param function: activation function to be used, e.g. sigmoid or tanh
    :param derivative: derivative of activation function to be used.
    :param learnrate: rate of learning.
    :return: list of numpy arrays representing the delta per weight per layer.
    """
    inputs = np.array(inputs)
    outputs = np.array(outputs)
    updates = []
    
    layers = len(weights) # set current layer to output layer +1, because 0 means input layer, for which there are no weights.
    
    for layer in range(layers,0,-1):    # layer 0 behandelen we niet in de backpropagation, want dat zijn slechts inputs.
        a_now = forward(inputs, weights, function, layer)    # calculate activation of a layer
        a_prev = forward(inputs, weights, function, layer-1) # calculate activation of a layer
        if layer == layers:
            # it is the output layer.
            print("HERE")
            delta = np.array(derivative(a_now) * (outputs - a_now)).T  # calculate error on output
        else:
            delta = derivative(a_now) * weights[layer][:,1:].T.dot(delta) # calculate error on current layer (that rule needs to work with the weights excluding the biases only. therefore, discard the first column with bias-weights.
            
        thetaUpdate = eta * np.outer(delta,np.append(1, a_prev)) # calculate adjustments to weights
        print("NOW HERE")
        print(delta,np.append(1, a_prev))
        print(np.outer(delta,np.append(1, a_prev)))
        print(weights, layer)
        print(thetaUpdate)
        updates.insert(0, thetaUpdate) # store adjustments

    return updates
        
# updating weights:
# given an array w of numpy arrays per layer and the deltas calculated by backprop, do
# for index in range(len(w)):
#     w[index] = w[index] + updates[index]


def get_random_weights(neuron_count, input_count, weight_0_location=[]):
    # weight_0_location is list filled with coordinates: [input, neuron], sorted by input low to high
    weights = []
    for neuron in range(0, neuron_count):
        temp_weights = []
        for input in range(0, input_count):
            if weight_0_location:
                if input == weight_0_location[0][0] and neuron == weight_0_location[0][1]:
                    temp_weights.append(0)
                    weight_0_location.pop(0)
                else:
                    temp_weights.append(random.uniform(-1, 1))
            else:
                temp_weights.append(random.uniform(-1, 1))
        weights.append(temp_weights)
    return weights


def main():
    training_data = [[0, 0], [1, 0], [0, 1], [1, 1]]
    training_solutions = [0, 1, 1, 0]
    learning_rate = 0.1
    weights_1 = np.array(get_random_weights(2, len(training_data[0]) + 1))
    weights_2 = np.array(get_random_weights(1, len(training_data[0]) + 1))
    weights = [weights_1, weights_2]
    iterations = 400

    # for i in range(0, iterations):
    for training_index in range(0, len(training_data)):
        weight_change = backprop(training_data[training_index], training_solutions[training_index], weights, tanh, derived_tanh, learning_rate)
        print("1", weights)
        print("2", weight_change)
        for weight_index in range(0, len(weights)):
            print(3, weight_index)
            print(weights[weight_index])
            print(weight_change[weight_index])
            print(weights[weight_index] + weight_change[weight_index])



main()