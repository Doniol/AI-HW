class perceptor:
    def __init__(self, input_weights, threshold):
        self.input_weights = input_weights
        self.threshold = threshold
        self.input_values = []

    def set_input(self, input_values):
        self.input = input_values

    def get_input(self):
        return self.input

    def get_output(self):
        return self.calculate_weighted_input() >= self.threshold
        
    def calculate_weighted_input(self):
        output = 0
        for input in range(0, len(self.input)):
            output += self.input[input] * self.input_weights[input]
        return output


class NOR_gate(perceptor):
    def __init__(self, input_count):
        perceptor.__init__(self, [-1] * input_count, 0)


def main():
    test_gate = NOR_gate(3)
    test_gate.set_input([1, 1, 1])
    print(test_gate.get_output())


main()