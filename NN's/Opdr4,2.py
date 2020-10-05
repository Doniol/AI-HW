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


class AND_gate(perceptor):
    def __init__(self, input_count):
        perceptor.__init__(self, [1] * input_count, input_count)


class OR_gate(perceptor):
    def __init__(self, input_count):
        perceptor.__init__(self, [1] * input_count, 1)


class XOR_gate:
    def __init__(self):
        self.input_values = []

    def set_input(self, input_values):
        self.input_values = input_values

    def get_input(self):
        return self.input
    
    def get_output(self):
        NOR_A1 = NOR_gate(2)
        NOR_A1.set_input(self.input_values)
        
        NOR_B1 =  NOR_gate(2)
        NOR_B1.set_input([NOR_A1.get_output(), self.input_values[0]])
    
        NOR_B2 =  NOR_gate(2)
        NOR_B2.set_input([NOR_A1.get_output(), self.input_values[1]])
    
        NOR_C1 =  NOR_gate(2)
        NOR_C1.set_input([NOR_B1.get_output(), NOR_B2.get_output()])

        NOR_D1 =  NOR_gate(2)
        NOR_D1.set_input([NOR_C1.get_output(), NOR_C1.get_output()])
        return NOR_D1.get_output()


class ADDER:
    def __init__(self):
        self.input_values = []
        self.input_carry = 0
    
    def set_input(self, input_values, input_carry):
        self.input_values = input_values
        self.input_carry = input_carry

    def get_input(self):
        return self.input, self.input_carry
    
    def get_output(self):
        XOR_A1 = XOR_gate()
        XOR_A1.set_input(self.input_values)
        
        AND_B1 = AND_gate(2)
        AND_B1.set_input([XOR_A1.get_output(), self.input_carry])

        AND_B2 = AND_gate(2)
        AND_B2.set_input(self.input_values)

        XOR_C1 = XOR_gate()
        XOR_C1.set_input([XOR_A1.get_output(), self.input_carry])

        OR_D1 = OR_gate(2)
        OR_D1.set_input([AND_B1.get_output(), AND_B2.get_output()])
        
        return XOR_C1.get_output(), OR_D1.get_output()


def main():
    input_1 = 0
    input_2 = 1
    input_carry = 1
    
    test = XOR_gate()
    test.set_input([input_1, input_2])
    print(test.get_output())


main()