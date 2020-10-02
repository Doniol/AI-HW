from __future__ import annotations
import math
import mpmath
import copy
import random
from datetime import datetime
import numpy

random.seed(1)
# Seed(0) vind ie niet leuk


class neuron_V2:
    def __init__(self, input_count: int, name: str):
        ''' This function creates a neuron

        input_count: The amount of inputs this neuron gets
        name: A string with the name of the neuron
        '''
        self.w = []
        for input in range(0, input_count):
            self.w.append(random.uniform(-1, 1))
        self.d = 0 # Delta
        self.Z = 0 # The sum of all weighted inputs
        self.a = 0 # The answer when puttin Z through tanh
        self.b = 0 # Bias
        self.previous_layer_connections = []
        self.next_layer_connections = []
        self.inputs = []
        self.name = name

    def get_name(self) -> str:
        ''' Get neuron name

        Return: The name of the neuron
        '''
        return self.name
    
    def set_weigts(self, weights: List[float]):
        ''' Set the weights for the inputs, these weights are in the same order of the input

        weights: A list with floats containing the weights of the inputs
        '''
        self.w = weights
    
    def set_inputs(self, inputs: List[float]):
        ''' Set the input for this neuron

        inputs: A list with inputs for this neuron, these can be intial inputs and the output of the neurons in the layer before
        '''
        self.inputs = inputs
    
    def get_specific_w(self, name: str) -> float:
        ''' Get the weight of specific input

        return: A float with the weight
        '''
        for w_index in range(0, len(self.w)):
            if self.previous_layer_connections[w_index].get_name() == name:
                return self.w[w_index]

    def set_previous_layer_connections(self, connections: List[neuron_V2]):
        ''' Set which neurons give their output to this one
        
        connection: The neurons which are the input of this neuron
        '''
        self.previous_layer_connections = connections
    
    def get_previous_layer_connections(self) -> List[neuron_V2]:
        ''' Get the neurons of the previous layer aka this neuron inputs
        
        return: A list containing the neurons
        '''
        return self.previous_layer_connections

    def add_previous_layer_connections(self, neuron: neuron_V2):
        ''' Adds a neuron to the previous_layers_connects
        neuron: A neuron
        '''
        self.previous_layer_connections.append(neuron)

    def set_next_layer_connections(self, connections: List[neuron_V2]):
        ''' Set to which neurons this neurons outputs
        
        connections: A list containing the neurons this neuron gives it output to 
        '''
        self.next_layer_connections = connections
    
    def get_next_layer_connections(self) -> List[neuron_V2]:
        ''' Get the neurons of the next layer aka this neuron outputs
        
        return: A list containing the neurons
        '''
        return self.next_layer_connections

    def add_next_layer_connections(self, neuron: neuron_V2):
        ''' Adds a neuron to the next_layers_connects

        neuron: A neuron
        '''
        self.next_layer_connections.append(neuron)

    def get_b(self) -> float:
        ''' Get the bias of the neuron

        return: The bias in the form of a float
        '''
        return self.b

    def set_b(self, bias: float):
        ''' Set the bias of the neuron

        bias: A float of the bias
        '''
        self.b = bias

    def get_d(self) -> float:
        ''' Get the delta, the difference between the gotten answer and the wanted answer

        return: The delta
        '''
        return self.d

    def set_d(self, delta: float):
        ''' Set the delta of the neuron

        delta: A float of the delta
        '''
        self.d = delta

    def calculate_a(self):
        ''' Calculate the tanh of the weighted inputs
        '''
        self.a = self.tanh(self.calculate_weighted_inputs())
    
    def get_a(self)-> List[float]:
        ''' Get the tanh of the weighted inputs

        return: A list of floats which contains the tanh of the weighted inputs
        '''
        return self.a
    
    def get_differentiated_a(self) -> List[float]:
        ''' Get the derivative of the a, aka the tanh of the weighted inputs

        return: the derivative of a
        '''
        return 1- self.tanh(self.tanh(self.calculate_weighted_inputs()))

    def tanh(self, x: float) -> float:
        ''' This function calculates the tanh of the given input
        
        return: The float of the calculated tanh
        '''
        return (mpmath.exp(x) - mpmath.exp(-x)) / (mpmath.exp(x) + mpmath.exp(-x)) 
    
    def calculate_weighted_inputs(self) -> float:
        ''' This function calculates the weighted inputs by multiplying the input with the weights
        
        return: A list with the weighted inputs and it returns the bias
        '''
        output = 0
        for input_index in range(0, len(self.inputs)):
            output += self.inputs[input_index] * self.w[input_index]
        return output + self.b
    
    def calculate_last_a(self) -> float:
        ''' This function calculates a for this neuron, using the newly calculated a's for the previous neurons
        This function loops through every neuron this neuron gets an input from, and through the neurons they get a input from until the actual input layer is reached,
        and calculates a (the tanh) for them. Because of this loop every neuron gets an updated output in the order of the layers.
        The output is the fully updated tanh of this neuron.
        
        return: The float of the tanh with the updated neurons
        '''
        if self.previous_layer_connections:
            self.inputs = []
            for connection in self.previous_layer_connections:
                self.inputs.append(connection.calculate_last_a())
        self.calculate_a()
        return self.get_a()

    def calc_delta(self):
        ''' Calulates the delta for this neuron
        The delta is the difference between the gotten answer and the wanted answer
        '''
        total_calc = 0
        for connection in self.next_layer_connections:
            total_calc += (connection.get_d() * connection.get_specific_w(self.name))
        self.d = self.get_differentiated_a() * total_calc

    def calc_all_delta_w_b(self, learning_rate, all_output_nodes):
        ''' This function calculates the delta, weights and biases for every neuron
        This function calculates the deltas, weights and biases for every neuron connected to this neuron and their inputs until the input layer is reached.
        This function is bassicaly a update function to update the information in every neuron.

        desired_outcome: The outcome which you want to get
        learning_rate: The step to take to get a local mininum
        '''
        # Get a list of all of the nodes in the previous layer
        names = []
        total_previous_layer = []
        for output in all_output_nodes:
            for node in output.get_previous_layer_connections():
                if node.get_name() not in names:
                    names.append(node.get_name())
                    total_previous_layer.append(node)
        self.calc_delta_per_layer(total_previous_layer, learning_rate)

    def calc_output_delta_w_b(self, desired_outcome: float, learning_rate):
        ''' Calculates and stored the delta, w and b values for this output node
        '''
        self.d = self.get_differentiated_a() * (desired_outcome - self.get_a())
        self.adapt_w_b(learning_rate)

    def calc_delta_per_layer(self, connections: List[neuron_V2], learning_rate: float):
        ''' This function calculates the delta for every neuron connected and applies this delta to the neuron.
        This function checks the input neurons of this neuron and calculates the delta for them.
        This function will continue doing this for each layer until the input layer is reached

        connections: A list with neurons which output towards this neuron
        learning_rate: The step to take to get a local mininum
        '''
        names = []
        new_layer_names = []
        new_layer = []
        for connection in connections:
            if connection.get_name() not in names:
                names.append(connection.get_name())
                connection.calc_delta()
                connection.adapt_w_b(learning_rate)
                if connection.get_previous_layer_connections():
                    for entry in connection.get_previous_layer_connections():
                        if entry.get_name() not in new_layer_names:
                            new_layer_names.append(entry.get_name())
                            new_layer.append(entry)
        if new_layer:
            self.calc_delta_per_layer(new_layer, learning_rate)

    def adapt_w_b(self, learning_rate: float):
        ''' This function updates the weights and bias of the neuron.

        learning_rate: The step to take towards the local mininum 
        '''
        for w_index in range(0, len(self.w)):
            self.w[w_index] += learning_rate * self.d * self.inputs[w_index]
        self.b += learning_rate * self.d

    def update(self, desired_solution: List[float], learning_rate: float, output_nodes_in_order_of_solution):
        ''' This function causes the neural network to learn
        This function calls the functions neccesary for the neural network to learn

        desired_solution: A list of the expected outcomes for final neuron
        learning_rate: the step to take towards the local mininum
        '''        
        for node_index in range(0, len(output_nodes_in_order_of_solution)):
            output_nodes_in_order_of_solution[node_index].calculate_last_a()
            output_nodes_in_order_of_solution[node_index].calc_output_delta_w_b(desired_solution[node_index], learning_rate)
        self.calc_all_delta_w_b(learning_rate, output_nodes_in_order_of_solution)


def load_data(filename):
    ''' This function loads the csv files and turns into an ndarray which we use to calculate the neigbour distance
    
    file: String with the file name
    return: A ndarray with the data, a list with names of the flowers
    '''
    data = numpy.genfromtxt(filename, delimiter=",", usecols=[0, 1, 2, 3], dtype=float)
    data_answers = numpy.genfromtxt(filename, delimiter=",", usecols=[4], dtype=str)
    answers = []
    for data_answer in data_answers:
        if data_answer == "Iris-setosa":
            answers.append([1, 0, 0])
        elif data_answer == "Iris-versicolor":
            answers.append([0, 1, 0])
        else:
            answers.append([0, 0, 1])
    return data, answers

# def calculate_fail_rate(training_data: np.ndarray, training_solutions: List[str], learning_rate: float, test_amount: int = 150) -> float:
#     # Create necessary neurons
#     # Layer 1
#     n_0_0 = neuron_V2(4, "0_0")
#     n_0_1 = neuron_V2(4, "0_1")
#     n_0_2 = neuron_V2(4, "0_2")
#     n_0_3 = neuron_V2(4, "0_3")
#     # Layer 2
#     n_1_0 = neuron_V2(4, "1_0")
#     n_1_1 = neuron_V2(4, "1_1")
#     # Layer 3
#     n_2_0 = neuron_V2(2, "2_0")

#     # Connect the neurons
#     # Layer 1
#     n_0_0.set_next_layer_connections([n_1_0, n_1_1])
#     n_0_1.set_next_layer_connections([n_1_0, n_1_1])
#     n_0_2.set_next_layer_connections([n_1_0, n_1_1])
#     n_0_3.set_next_layer_connections([n_1_0, n_1_1])
#     # Layer 2
#     n_1_0.set_previous_layer_connections([n_0_0, n_0_1, n_0_2, n_0_3])
#     n_1_0.set_next_layer_connections([n_2_0])
#     n_1_1.set_previous_layer_connections([n_0_0, n_0_1, n_0_2, n_0_3])
#     n_1_1.set_next_layer_connections([n_2_0])
#     # Layer 3
#     n_2_0.set_previous_layer_connections([n_1_0, n_1_1])

#     # Train the NN and print results
#     for i in range(0, 800):
#         print(i)
#         for training_index in range(0, len(training_data)):
#             n_0_0.set_inputs(training_data[training_index])
#             n_0_1.set_inputs(training_data[training_index])
#             n_0_2.set_inputs(training_data[training_index])
#             n_0_3.set_inputs(training_data[training_index])
#             n_2_0.update(training_solutions[training_index], learning_rate)

#     # test_data_index = []
#     # for i in range(test_amount):
#     #     test_data_index.append(random.randint(0, len(training_data)-1))

#     correct_answer = 0
#     for test_index in range(len(training_solutions)):
#         print(test_index)
#         n_0_0.set_inputs(training_data[test_index])
#         n_0_1.set_inputs(training_data[test_index])
#         n_0_2.set_inputs(training_data[test_index])
#         n_0_3.set_inputs(training_data[test_index])
#         last_a = n_2_0.calculate_last_a()
#         if training_solutions[test_index] == round(last_a):
#             print("Correct answer, last a = " , last_a , " The answer was " , training_solutions[test_index])
#             correct_answer += 1
#         else:
#             print("Incorrect answer, last a = ", last_a ," The answer was ", training_solutions[test_index])

#     return correct_answer / test_amount *100           

def main():
    bool_fail_rate = False
    data, answers = load_data("iris.data")
    training_data = data
    training_solutions = answers
    learning_rate = 0.1

    # Create necessary neurons
    # Layer 1
    n_0_0 = neuron_V2(4, "0_0")
    n_0_1 = neuron_V2(4, "0_1")
    n_0_2 = neuron_V2(4, "0_2")
    n_0_3 = neuron_V2(4, "0_3")
    # Layer 2
    n_1_0 = neuron_V2(4, "1_0")
    n_1_1 = neuron_V2(4, "1_1")
    # Layer 3
    n_2_0 = neuron_V2(2, "2_0")
    n_2_1 = neuron_V2(2, "2_1")
    n_2_2 = neuron_V2(2, "2_2")

    # Connect the neurons
    # Layer 1
    n_0_0.set_next_layer_connections([n_1_0, n_1_1])
    n_0_1.set_next_layer_connections([n_1_0, n_1_1])
    n_0_2.set_next_layer_connections([n_1_0, n_1_1])
    n_0_3.set_next_layer_connections([n_1_0, n_1_1])
    # Layer 2
    n_1_0.set_previous_layer_connections([n_0_0, n_0_1, n_0_2, n_0_3])
    n_1_0.set_next_layer_connections([n_2_0, n_2_1, n_2_2])
    n_1_1.set_previous_layer_connections([n_0_0, n_0_1, n_0_2, n_0_3])
    n_1_1.set_next_layer_connections([n_2_0, n_2_1, n_2_2])
    # Layer 3
    n_2_0.set_previous_layer_connections([n_1_0, n_1_1])
    n_2_1.set_previous_layer_connections([n_1_0, n_1_1])
    n_2_2.set_previous_layer_connections([n_1_0, n_1_1])

    if(bool_fail_rate):
        # print(calculate_fail_rate(training_data, training_solutions, learning_rate))
        return
    else:
        # Train the NN and print results
        for i in range(0, 50):
            print(i)
            for training_index in range(0, len(training_data)):
                n_0_0.set_inputs(training_data[training_index])
                n_0_1.set_inputs(training_data[training_index])
                n_0_2.set_inputs(training_data[training_index])
                n_0_3.set_inputs(training_data[training_index])
                n_2_0.update(training_solutions[training_index], learning_rate, [n_2_0, n_2_1, n_2_2])
        
        for training_index in range(0, len(training_data)):
            n_0_0.set_inputs(training_data[training_index])
            n_0_1.set_inputs(training_data[training_index])
            n_0_2.set_inputs(training_data[training_index])
            n_0_3.set_inputs(training_data[training_index])
            print("Input:", training_data[training_index], "Answer: ", training_solutions[training_index], "Final a: ", 
                  n_2_0.calculate_last_a(), n_2_1.calculate_last_a(), n_2_2.calculate_last_a())
        print(n_2_0.w, n_2_1.w, n_2_2.w)



main()