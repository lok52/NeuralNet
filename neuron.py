from random import random
from math import tanh


class Connection:
    def __init__(self, weight, delta_weight = 0):
        self.weight = weight
        self.delta_weight = delta_weight


class Neuron:
    def __init__(self, next_layer_neuron_num, index, output_val=0):
        # net learning rate
        self.nlt = 0.20
        # momentum
        self.alpha = 0.5
        self.neuron_index = index
        self.output_val = output_val
        self.output_weights = [Connection(random()) for _ in range(next_layer_neuron_num)]
        self.gradient = 0

    def feed(self, prev_layer):
        sum = 0
        for i in range(len(prev_layer)):
            sum += prev_layer[i].output_val * prev_layer[i].output_weights[self.neuron_index].weight
        self.output_val = self.activation_function(sum)

    @staticmethod
    def activation_function(x):
        return tanh(x)

    @staticmethod
    def activation_function_derivative(x):
        return 1 - x * x

    def der_sum(self, next_layer):
        sum = 0
        for i in range(len(next_layer) - 1):
            sum += self.output_weights[i].weight * next_layer[i].gradient
        return sum

    def calc_output_gradient(self, target_val):
        delta = float(target_val) - self.output_val
        self.gradient = delta * self.activation_function_derivative(self.output_val)

    def calc_hidden_gradient(self, next_layer):
        dow = self.der_sum(next_layer)
        self.gradient = dow * self.activation_function_derivative(self.output_val)

    def update_input_weights(self, prev_layer):
        for neuron in prev_layer:
            old_delta_weight = neuron.output_weights[self.neuron_index].delta_weight
            new_delta_weight = self.nlt * neuron.output_val * self.gradient + self.alpha * old_delta_weight

            neuron.output_weights[self.neuron_index].delta_weight = new_delta_weight
            neuron.output_weights[self.neuron_index].weight += new_delta_weight
