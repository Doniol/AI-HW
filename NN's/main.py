import math
import copy
import random

class neuron:
    def __init__(self, input_weights):
        self.input_weights = input_weights
        self.input_values = []
        self.bias = 0

    def set_input(self, input_values):
        self.input_values = input_values

    def get_input(self):
        return self.input_values

    def get_output(self):
        # return 1 / (1 + math.exp(-self.calculate_weighted_input()))
        return self.tanh(self.calculate_weighted_input())

    def get_differentiated_output(self):
        return 1- self.tanh(self.tanh(self.calculate_weighted_input()))
       
    def tanh(self, x):
        return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))   

    def calculate_weighted_input(self):
        output = 0
        for input_index in range(0, len(self.input_values)):
            output += self.input_values[input_index] * self.input_weights[input_index]
        return output + self.bias
    
    def update(self, desired_outcomes, training_inputs, learning_rate, weightdiff):
        # Loop through each weight
        for weight_index in range(0, len(self.input_weights)):
            # Run every test for each weight
            for training_index in range(0, len(training_inputs)):
                # Create new weights for calculation purposes
                test_weights = copy.deepcopy(self.input_weights)
                test_weights[weight_index] += weightdiff
                # Calculate the difference between C with the current weights, and C with the new weights
                dC = (self.calc_C(training_inputs[training_index], desired_outcomes[training_index], self.input_weights) - 
                      self.calc_C(training_inputs[training_index], desired_outcomes[training_index], test_weights))
                # If C is negative, that means the new weights are better than the old ones
                if dC < 0:
                    # Update the old weights
                    self.input_weights[weight_index] -= learning_rate * dC / weightdiff
                else:
                    # If C is positive, that might mean that a low C lies on the other side of the current weights
                    # Create new weights that check for lower weights than the current ones
                    test_weights[weight_index] -= 2 * weightdiff
                    # Calculate the difference of C again
                    dC = (self.calc_C(training_inputs[training_index], desired_outcomes[training_index], self.input_weights) - 
                          self.calc_C(training_inputs[training_index], desired_outcomes[training_index], test_weights))
                    # Once again, a negative C is a good thing
                    if dC < 0:
                        self.input_weights[weight_index] += learning_rate * dC / weightdiff


        # Weight derivative = original weight - (learning rate * deltaC / deltaWeight)

        # Bereken voor elke weight de verandering in C als je de weight een beetje veranderd
        # Als de verandering in C - is, dan gaat t goed en kan je de weight aanpassen, anders kijk je andere kant op


        # delta rule:
        # Weight tussen node j en k = bestaande weight + (???)

    def calc_C(self, training_input, desired_outcome, weights):
        mean = 0
        test_neuron = neuron(weights)
        test_neuron.set_input(training_input)
        C = (desired_outcome - test_neuron.get_output())**2 / 2
        # print("CalC: ", training_input, desired_outcome, test_neuron.get_output(), weights, C)
        return C


class neuron_V2:
    def __init__(self, input_count, previous_layer_connections ,input_weights = []):
        self.weights = input_weights
        if len(input_weights) >= 1:
            for input in input_count:
                input_weights.append(random.uniform(-1, 1))
        self.d = 0 # Delta
        self.Z = 0 # The sum of all weighted inputs
        self.a = 0 # The answer when puttin Z through tanh
        self.b = 0 # Bias
        self.previous_layer_connections = previous_layer_connections
        self.inputs = []
    
    def set_inputs(inputs):
        self.inputs = inputs

    def set_previous_layer_connections(connections):
        self.previous_layer_connections = connections
    
    def get_b(self):
        return self.b

    def set_b(self, bias):
        self.b = bias

    def get_d(self):
        return self.d

    def calculate_a(self):
        self.a = self.tanh(self.calculate_weighted_inputs())
    
    def get_a(self):
        return self.a
    
    def get_differentiated_a(self):
        return 1- self.tanh(self.tanh(self.calculate_weighted_inputs()))

    def tanh(self, x):
        return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x)) 
    
    def calculate_weighted_inputs(self):
        output = 0
        for input_index in range(0, len(self.inputs)):
            output += self.inputs[input_index] * self.weights[input_index]
        return output + self.b
    
    def calculate_last_a(self):
        if not self.inputs:
            for connection in self.previous_layer_connections:
                inputs.append(connection.calculate_last_a())
        return self.calculate_a()


    def calc_delta(self):
        self.
        


        neuron_o.input_weights[0] += learning_rate * o_delta * neuron_h.get_output()
        neuron_o.input_weights[1] += learning_rate * o_delta * neuron_l.get_output()
        print(neuron_o.input_weights)
        neuron_h.input_weights[0] += learning_rate * h_delta * training_data[0][0]
        neuron_h.input_weights[1] += learning_rate * h_delta * training_data[0][1]
        print(neuron_h.input_weights)
        neuron_l.input_weights[0] += learning_rate * l_delta * training_data[0][0]
        neuron_l.input_weights[1] += learning_rate * l_delta * training_data[0][1]
        print(neuron_l.input_weights)

    def update(self):
        



def main():
    training_data = [[1, 1], [1, 0], [0, 1], [0, 0]]
    training_solutions= [0, 1, 1, 0]
    learning_rate = 0.1

    neuron_h = neuron([0.2, -0.4])
    neuron_l = neuron([0.7, 0.1])
    neuron_o = neuron([0.6, 0.9])

    # test = neuron([0.7, 0.1])
    # test.set_input([1, 1])
    # print(test.get_output())
    
    # Connect neurons and calc last a
    neuron_h.set_input([training_data[0][0], training_data[0][1]])
    print(neuron_h.get_output())
    neuron_l.set_input([training_data[0][0], training_data[0][1]])
    print(neuron_l.get_output())
    neuron_o.set_input((neuron_h.get_output(), neuron_l.get_output()))
    print(neuron_o.get_output())

    # Calc all deltas
    o_delta = neuron_o.get_differentiated_output() * (training_solutions[0] - neuron_o.get_output())
    h_delta = neuron_h.get_differentiated_output() * (o_delta * neuron_o.input_weights[0])
    l_delta = neuron_l.get_differentiated_output() * (o_delta * neuron_o.input_weights[1])
    print(o_delta, h_delta, l_delta)

    # Change neuron weights based on deltas
    neuron_o.input_weights[0] += learning_rate * o_delta * neuron_h.get_output()
    neuron_o.input_weights[1] += learning_rate * o_delta * neuron_l.get_output()
    print(neuron_o.input_weights)
    neuron_h.input_weights[0] += learning_rate * h_delta * training_data[0][0]
    neuron_h.input_weights[1] += learning_rate * h_delta * training_data[0][1]
    print(neuron_h.input_weights)
    neuron_l.input_weights[0] += learning_rate * l_delta * training_data[0][0]
    neuron_l.input_weights[1] += learning_rate * l_delta * training_data[0][1]
    print(neuron_l.input_weights)

    # Change neuron biases based on deltas
    neuron_o.bias += learning_rate * o_delta
    print(neuron_o.bias)
    neuron_h.bias += learning_rate * h_delta
    print(neuron_h.bias)
    neuron_l.bias += learning_rate * l_delta
    print(neuron_l.bias)


main()