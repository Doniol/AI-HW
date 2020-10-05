import numpy as np
import random
from typing import List, Tuple

random.seed(0)


def load_data(filename: str) -> Tuple[List[List[float]], List[List[int]]]:
    ''' This function loads the csv files and turns into an ndarray which we use to calculate the neighbour distance
    
    file: String with the file name
    return: A ndarray with the data and one with the corresponding names of the flowers
    '''

def normalize(dataset_to_be_changed: List[List[float]]) -> None:
    ''' This function normalizes the given dataset
    
    dataset_to_be_changed: The dataset which you want to normalize
    '''


def sigmoid(z: float) -> float:
    ''' This function runs a sigmoid function over the given value

    z: The given value
    '''


def sigmoid_derivative(z: float) -> float:
    ''' This function runs a derivative sigmoid function over the given value

    z: The given value
    '''


class Neuron:
    def __init__(self, name: str) -> None:
        ''' Creates a neuron

        name: A name for the neuron, denoting it's function, position and eventual layer
        '''

    def calc_a_z(self) -> None:
        ''' This function calculates the a and z value of the neuron
        The a and z values are calculated and saved in the neuron.
        The z is the sum of all weighted inputs of this neuron.
        The a is the sigmoid of these weighted inputs.
        '''

    def calc_delta(self, desired_solution: List[int] = []) -> None:
        ''' This function handles the backwards propagation of the neuron
        
        desired_solution: A list containing the expected outcome of the output
        '''

    def calc_delta_layer(self) -> None:
        ''' This function calculates the delta of the neuron if it's not the output
        The delta is stored inside the neuron
        '''

    def calc_delta_output(self, desired_solution: List[int] = []) -> None:
        ''' This function calculates the delta for output neurons
        The delta is calculated using the desired solution and saved in the neuron

        desired_solution: A list containing the expected outcome of the output
        '''

    def update_w_b(self, learning_rate: float) -> None:
        ''' This function updates the bias and weights of the neuron
        '''


class NeuralNetwork:
    def __init__(self, input_count: int, output_count:int, hidden_layers: List[int]) -> None:
        ''' Create a neural network

        input_count: The amount of input nodes for the network
        output_count: The amount of output nodes for the network
        hidden_layers: A list containing integers, each integer is a hidden layer containing an amount of neurons corresponding to the integer
        '''

    def connect_all_neurons(self) -> None:
        ''' This function connects the neurons to each other in order to form a coherent network
        '''

    def set_inputs(self, inputs: List[float]) -> None:
        ''' This function sets the inputs for the input neurons

        inputs: A list with the inputs for the input layer
        '''
    
    def feed_forward(self) -> None:
        ''' This function is used to feed the information in the neural network forwards to calculate the output values
        '''
    
    def get_outputs(self)-> List[float]:
        ''' This function returns the outputs of the output neurons

        return: A list with the outputs
        '''
    
    def backwards_propagation(self, desired_solution: List[int]) -> None:
        ''' This function runs the backward propagation principle on the entire network

        desired_solution: A list with the desired outcome for the output neurons
        '''
                
    def update_network(self, learning_rate: float) -> None:
        ''' This function will update the weights and biases of all of the neurons in the network

        learning_rate: A float denoting how much the weights and biases should change in accordance with the calculated d-values
        '''

    def print_network_layout(self) -> None:
        ''' This function shows the topology of the neural network, each neuron is connnected to all the other neurons of the next layer
        '''
    
    def train_network(self, test_inputs: List[List[float]], test_solutions: List[List[int]], learning_rate: float, iterations: int) -> None:
        ''' This function starts the training of the neural network

        test_inputs: An array with the data to train the network with
        test_solutions: An array with the solutions of the test_inputs
        learning_rate: A float denoting how much the weights and biases should change in accordance with the calculated d-values
        iterations: The amount of iterations the network trains for
        '''
            
    def calculate_fault(self, dataset: List[List[float]], desired_solutions: List[List[int]]) -> float:
        ''' This function calculates the error between the desired outcome and the calculated outcome

        dataset: The dataset to run the test with
        desired_solutions: A list with the desired outcomes
        return: The raw difference between the desired and received results
        '''

    def calculate_success_rate(self, dataset: List[List[float]] , desired_solutions: List[List[int]]) -> float:
        ''' This function calculates the succes rate of the neural network

        dataset: The dataset you want to validate
        desired_solutions: A list with the outcomes the dataset should have
        return: The percentage of the correctly answered data 
        '''
    
    def print_outputs(self, dataset: List[List[float]]):
        ''' This function prints the outputs of the output neurons

        dataset: The dataset you want to validate
        '''

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

    Iris.train_network(inputs_Iris, solutions_Iris, learning_rate, 2000)
    Iris_success_rate = Iris.calculate_success_rate(inputs_Iris, solutions_Iris)
    # Iris.print_outputs(inputs_Iris)
    print("Iris: ", Iris_success_rate, "%")


main()