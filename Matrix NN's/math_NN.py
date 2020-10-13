import numpy as np
import math
import mpmath
import random

random.seed(1)


def tanh(x: float) -> float:
    ''' This function calculates the tanh of the given input
    
    return: The float of the calculated tanh
    '''
    return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x)) 


def tanh_array(array):
    output = []
    for entry in array:
        output.append(tanh(entry))
    return output


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
            

def get_random_inputs(input_count):
    inputs = []
    for input in range(0, input_count):
        inputs.append(random.uniform(-1, 1))
    return inputs


def calc_a(OG_inputs, weights_all, function):
    inputs = OG_inputs
    for weights in weights_all:
        inputs = function(np.dot(weights, np.insert(inputs, 0, 1)))
    return inputs


def calc_C(results, training_solutions):
    return sum([x**2 for x in np.array(results) - np.array(training_solutions)])


def calc_parallel_a(OG_matrix_inputs, weights_all, function):
    matrix_inputs = OG_matrix_inputs
    for weights in weights_all:
        matrix_inputs = np.dot(weights, matrix_inputs)
        tanh_inputs = []
        for array in matrix_inputs:
            tanh_inputs.append(function(array))
        matrix_inputs = np.ones([matrix_inputs.shape[0] + 1, matrix_inputs.shape[1]])
        matrix_inputs[1:, :] = tanh_inputs
    return matrix_inputs[1:, :]


def calc_parallel_a_test(OG_matrix_inputs, weights_all, function):
    matrix_inputs = OG_matrix_inputs
    for weights in weights_all:
        new_inputs = function(np.dot(weights, matrix_inputs))
        matrix_inputs = np.ones([new_inputs.shape[0] + 1, new_inputs.shape[1]])
        matrix_inputs[1:, :] = new_inputs
    return matrix_inputs[1:, :]


def main():
    tanh_vectorized = np.vectorize(math.tanh)
    # Every node needs a weight for every node/input in previous layer, if said node/input is not connected to all nodes then its weight equals 0
    # Test part 1: final output
    print("Test 1")
    # Test No1
    print("No1")
    inputs_1 = [0, 1]
    bias = [1]
    actual_inputs_1 = np.array(np.insert(inputs_1, 0, bias))
    weights_1 = np.array(get_random_weights(2, len(actual_inputs_1)))
    outputs_1 = np.dot(weights_1, actual_inputs_1)
    results_1 = tanh_vectorized(outputs_1)

    inputs_2 = results_1
    actual_inputs_2 = np.array(np.insert(inputs_2, 0, bias))
    weights_2 = np.array(get_random_weights(1, len(actual_inputs_2)))
    outputs_2 = np.dot(weights_2, actual_inputs_2)
    results_2 = tanh_vectorized(outputs_2)
    print(results_2)

    # Test No2
    print("No2")
    results_1_1 = tanh_array(outputs_1)
    inputs_2_1 = results_1_1
    actual_inputs_2_1 = np.array(np.insert(inputs_2_1, 0, bias))
    outputs_2_1 = np.dot(weights_2, actual_inputs_2_1)
    results_2_1 = tanh_array(outputs_2_1)
    print(results_2_1)

    # Test No3
    print("No3")
    print(inputs_1)
    print(calc_a(inputs_1, [weights_1, weights_2], tanh_vectorized)[0])

    # Test No4
    print("No4")
    new_inputs = np.array([[0], [1]])
    print(new_inputs)
    print(calc_a(new_inputs, [weights_1, weights_2], tanh_vectorized)[0])
    
    # Test part 2: Cost
    print("Test 2")
    training_data = [[0, 0], [1, 0], [0, 1], [1, 1]]
    training_solutions = [0, 1, 1, 0]
    results = []
    for training in training_data:
        results.append(calc_a(training, [weights_1, weights_2], tanh_vectorized)[0])
    print(results)
    
    # Test No1
    print("No1")
    cost = 0
    for result in range(0, len(results)):
        cost += (results[result] - training_solutions[result])**2
    print(cost)

    # Test No2
    print("No2")
    print(calc_C(results, training_solutions))

    # Test part 3: further parallelisation
    print("Test 3")
    # Test No1
    print("No1")
    for training in training_data:
        print(calc_a(training, [weights_1, weights_2], tanh_vectorized))

    # Test No2
    print("No2")
    inputs_parallel = np.array([[1, 1, 1, 1], [0, 1, 0, 1], [0, 0, 1, 1]])
    print(calc_parallel_a(inputs_parallel, [weights_1, weights_2], tanh_vectorized))

    # Test No3
    print("No3")
    tanh_vectorized_twice = np.vectorize(tanh_vectorized)
    print(calc_parallel_a_test(inputs_parallel, [weights_1, weights_2], tanh_vectorized_twice))

    # Test part 4: Backwards Propagation


main()