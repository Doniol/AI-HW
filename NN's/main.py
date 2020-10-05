import numpy as np
import random
from typing import List, Tuple

random.seed(0)


def load_data(filename: str) -> Tuple[List[List[float]], List[List[int]]]:
    ''' This function loads the csv files and turns into an ndarray which we use to calculate the neighbour distance
    
    file: String with the file name
    return: A ndarray with the data and one with the corresponding names of the flowers
    '''
    data = np.genfromtxt(filename, delimiter=",", usecols=[0, 1, 2, 3], dtype=float)
    data_answers = np.genfromtxt(filename, delimiter=",", usecols=[4], dtype=str)
    answers = []
    for data_answer in data_answers:
        if data_answer == "Iris-setosa":
            answers.append([1, 0, 0])
        elif data_answer == "Iris-versicolor":
            answers.append([0, 1, 0])
        else:
            answers.append([0, 0, 1])
    return data, answers


def normalize(dataset_to_be_changed: List[List[float]]) -> None:
    ''' This function normalizes the given dataset
    
    dataset_to_be_changed: The dataset which you want to normalize
    '''
    min_values = [float("inf") for data_type in range(0, len(dataset_to_be_changed[0]))]
    max_values = [float("-inf") for data_type in range(0, len(dataset_to_be_changed[0]))]
    for data_type in range(0, len(dataset_to_be_changed[0])):
        for data_entry in dataset_to_be_changed:
            if data_entry[data_type] > max_values[data_type]:
                max_values[data_type] = data_entry[data_type]
            elif data_entry[data_type] < min_values[data_type]:
                min_values[data_type] = data_entry[data_type]

    for data_type in range(0, len(dataset_to_be_changed[0])):
        for day in dataset_to_be_changed:
            day[data_type] = (day[data_type] - min_values[data_type]) / (max_values[data_type] - min_values[data_type])


def sigmoid(z: float) -> float:
    ''' This function runs a sigmoid function over the given value

    z: The given value
    '''
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(z: float) -> float:
    ''' This function runs a derivative sigmoid function over the given value

    z: The given value
    '''
    return sigmoid( z ) * (1 - sigmoid( z ))


class Neuron:
    def __init__(self, name: str) -> None:
        ''' Creates a neuron

        name: A name for the neuron, denoting it's function, position and eventual layer
        '''
        self.previous_neurons = []
        self.previous_weights = []
        self.next_neurons =[]
        self.a = 0
        self.b = 0
        self.d = 0
        self.z = 0
        self.name = name

    def calc_a_z(self) -> None:
        ''' This function calculates the a and z value of the neuron
        The a and z values are calculated and saved in the neuron.
        The z is the sum of all weighted inputs of this neuron.
        The a is the sigmoid of these weighted inputs.
        '''
        sum_weights = 0
        # Calulate the sum of all weighted inputs
        for weigth in range(len(self.previous_weights)):
            sum_weights += self.previous_weights[weigth] * self.previous_neurons[weigth].a
        # Save this in z
        self.z = sum_weights + self.b
        # Run the sigmoid function over z
        self.a = sigmoid(self.z)

    def calc_delta(self, desired_solution: List[int] = []) -> None:
        ''' This function handles the backwards propagation of the neuron
        
        desired_solution: A list containing the expected outcome of the output
        '''
        # If this neuron is a part of a hidden layer
        if self.name[0] == "L":
            self.calc_delta_layer()
        # If this neuron is an output neuron
        else:
            self.calc_delta_output(desired_solution)

    def calc_delta_layer(self) -> None:
        ''' This function calculates the delta of the neuron if it's not the output
        The delta is stored inside the neuron
        '''
        sum_weighted_next_neurons = 0
        for next_neuron in self.next_neurons:
            next_neuron_weight = next_neuron.previous_weights[next_neuron.previous_neurons.index(self)]
            sum_weighted_next_neurons += next_neuron.d * next_neuron_weight
        self.d = sigmoid_derivative(self.z) * sum_weighted_next_neurons

    def calc_delta_output(self, desired_solution: List[int] = []) -> None:
        ''' This function calculates the delta for output neurons
        The delta is calculated using the desired solution and saved in the neuron

        desired_solution: A list containing the expected outcome of the output
        '''
        self.d = sigmoid_derivative(self.z) * (desired_solution[int(self.name[2:])] - self.a)

    def update_w_b(self, learning_rate: float) -> None:
        ''' This function updates the bias and weights of the neuron
        '''
        for previous_neuron_index in range(len(self.previous_neurons)):
            self.previous_weights[previous_neuron_index] += learning_rate * self.d * self.previous_neurons[previous_neuron_index].a
        self.b += learning_rate * self.d


class NeuralNetwork:
    def __init__(self, input_count: int, output_count:int, hidden_layers: List[int]) -> None:
        ''' Create a neural network

        input_count: The amount of input nodes for the network
        output_count: The amount of output nodes for the network
        hidden_layers: A list containing integers, each integer is a hidden layer containing an amount of neurons corresponding to the integer
        '''
        # Add input neurons to self.neurons
        self.neurons = [[Neuron("I." + str(count)) for count in range(0, input_count)]]
        # Add hidden_layer neurons to self.neurons
        for layer_index in range(0, len(hidden_layers)):
            new_layer = []
            for neuron_index in range(0, hidden_layers[layer_index]):
                new_layer.append(Neuron("L." + str(layer_index + 1) + "." + str(neuron_index)))
            self.neurons.append(new_layer)
        # Add output neurons to self.neurons
        self.neurons.append([Neuron("O." + str(count)) for count in range(0, output_count)])

    def connect_all_neurons(self) -> None:
        ''' This function connects the neurons to each other in order to form a coherent network
        '''
        for layer_index in range(1, len(self.neurons)):
            for neuron in self.neurons[layer_index]:
                # Set previous neurons to previous layer
                neuron.previous_neurons = self.neurons[layer_index - 1]
                neuron.previous_weights = [random.uniform(-1, 1) for neuron in range(0, len(self.neurons[layer_index - 1]))]

                # If current neuron is not an output neuron
                if not layer_index == len(self.neurons) - 1:
                    # Set next neurons to next layer
                    neuron.next_neurons = self.neurons[layer_index + 1]

    def set_inputs(self, inputs: List[float]) -> None:
        ''' This function sets the inputs for the input neurons

        inputs: A list with the inputs for the input layer
        '''
        for neuron_index in range(0, len(self.neurons[0])):
            self.neurons[0][neuron_index].a = inputs[neuron_index]
    
    def feed_forward(self) -> None:
        ''' This function is used to feed the information in the neural network forwards to calculate the output values
        '''
        for layer_index in range(1, len(self.neurons)):
            for neuron in self.neurons[layer_index]:
                neuron.calc_a_z()
    
    def get_outputs(self)-> List[float]:
        ''' This function returns the outputs of the output neurons

        return: A list with the outputs
        '''
        outputs = []
        for output in self.neurons[-1]:
            outputs.append([output.name, output.a])
        return outputs
    
    def backwards_propagation(self, desired_solution: List[int]) -> None:
        ''' This function runs the backward propagation principle on the entire network

        desired_solution: A list with the desired outcome for the output neurons
        '''
        for layer_index in range(len(self.neurons) - 1, 0, -1):
            for neuron in self.neurons[layer_index]:
                neuron.calc_delta(desired_solution)
                
    def update_network(self, learning_rate: float) -> None:
        ''' This function will update the weights and biases of all of the neurons in the network

        learning_rate: A float denoting how much the weights and biases should change in accordance with the calculated d-values
        '''
        for layer_index in range(1, len(self.neurons)):
            for neuron in self.neurons[layer_index]:
                neuron.update_w_b(learning_rate)

    def print_network_layout(self) -> None:
        ''' This function shows the topology of the neural network, each neuron is connnected to all the other neurons of the next layer
        '''
        for layer_index in range(0, len(self.neurons)):
            neuron_list = []
            for neuron in self.neurons[layer_index]:
                neuron_list.append(neuron.name)
            print(neuron_list, "\n")
    
    def train_network(self, test_inputs: List[List[float]], test_solutions: List[List[int]], learning_rate: float, iterations: int) -> None:
        ''' This function starts the training of the neural network

        test_inputs: An array with the data to train the network with
        test_solutions: An array with the solutions of the test_inputs
        learning_rate: A float denoting how much the weights and biases should change in accordance with the calculated d-values
        iterations: The amount of iterations the network trains for
        '''
        for iteration in range(0, iterations):
            for test_index in range(0, len(test_solutions)):
                self.set_inputs(test_inputs[test_index])
                self.feed_forward()
                self.backwards_propagation(test_solutions[test_index])
                self.update_network(learning_rate)
            
    def calculate_fault(self, dataset: List[List[float]], desired_solutions: List[List[int]]) -> float:
        ''' This function calculates the error between the desired outcome and the calculated outcome

        dataset: The dataset to run the test with
        desired_solutions: A list with the desired outcomes
        return: The raw difference between the desired and received results
        '''
        fault = 0
        for data_index in range(0, len(dataset)):
            self.set_inputs(dataset[data_index])
            self.feed_forward()
            for solution in desired_solutions[data_index]:
                fault += np.sqrt((np.array(self.get_outputs()) - np.array(solution))**2)
        return fault
        
    def calculate_success_rate(self, dataset: List[List[float]] , desired_solutions: List[List[int]]) -> float:
        ''' This function calculates the succes rate of the neural network

        dataset: The dataset you want to validate
        desired_solutions: A list with the outcomes the dataset should have
        return: The percentage of the correctly answered data 
        '''
        correct_count = 0
        for data_index in range(0, len(dataset)):
            self.set_inputs(dataset[data_index])
            self.feed_forward()
            outputs = [1 if output[1] > 0.5 else 0 for output in self.get_outputs()]
            
            if outputs == desired_solutions[data_index]:
                correct_count += 1
        return (correct_count / len(dataset) * 100)
    
    def print_outputs(self, dataset: List[List[float]]):
        ''' This function prints the outputs of the output neurons

        dataset: The dataset you want to validate
        '''
        for data_index in range(0, len(dataset)):
            self.set_inputs(dataset[data_index])
            self.feed_forward()
            print(self.get_outputs())


def main():
    learning_rate = 1

    # XOR gate with 2 inputs
    inputs_XOR = [[0, 0], [0, 1], [1, 0], [1, 1]]
    solutions_XOR = [[0], [1], [1], [0]]
    XOR = NeuralNetwork(2, 1, [2])
    XOR.connect_all_neurons()

    XOR.train_network(inputs_XOR, solutions_XOR, learning_rate, 1000)
    XOR_success_rate = XOR.calculate_success_rate(inputs_XOR, solutions_XOR)
    # XOR.print_outputs(input_XOR)
    print("XOR: ", XOR_success_rate, "%")

    # ADDER
    inputs_ADDER = [[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]]
    solutions_ADDER = [[0,0],  [0,1],  [0,1],  [1,0],  [0,1],  [1,0],  [1,0],  [1,1]]
    ADDER = NeuralNetwork(3, 2, [3])
    ADDER.connect_all_neurons()

    ADDER.train_network(inputs_ADDER, solutions_ADDER, learning_rate, 1000)
    ADDER_success_rate = ADDER.calculate_success_rate(inputs_ADDER, solutions_ADDER)
    # ADDER.print_outputs(inputs_ADDER)
    print("ADDER: ", ADDER_success_rate,"%" )

    # Iris Dataset
    inputs_Iris, solutions_Iris = load_data("NN's\iris.data")
    normalize(inputs_Iris)
    Iris = NeuralNetwork(4, 3, [2])
    Iris.connect_all_neurons()

    Iris.train_network(inputs_Iris, solutions_Iris, learning_rate, 1300)
    Iris_success_rate = Iris.calculate_success_rate(inputs_Iris, solutions_Iris)
    # Iris.print_outputs(inputs_Iris)
    print("Iris: ", Iris_success_rate, "%")


main()