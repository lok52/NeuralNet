from neuron import Neuron
from math import sqrt


class Net:
    def __init__(self, topology):
        self.numLayers = len(topology)
        # layers is dependent on net topology (may vary)
        # layers - vector of layers
        # layer - vector of neuron
        # self.layers = [[Neuron(0 if layer_num == self.numLayers - 1 else topology[layer_num + 1], neuron_num)
        #                 for neuron_num in range(topology[layer_num] + 1)]
        #                for layer_num in range(self.numLayers)]
        self.layers = []
        for layer_num in range(self.numLayers):
            self.layers.append([])
            num_outputs = 0 if layer_num == self.numLayers - 1 else topology[layer_num + 1]
            for neuron_num in range(topology[layer_num] + 1):
                self.layers[-1].append(Neuron(num_outputs, neuron_num))
            self.layers[-1][-1].output_val = 1
        self.error = 0
        self.recent_average_error = 0
        self.recent_average_smoothing_factor = 2000

    def feed_forward(self, input_vals):
        for i in range(len(input_vals)):
            self.layers[0][i].output_val = input_vals[i]

        for layer_num in range(1, len(self.layers)):
            prev_layer = self.layers[layer_num - 1]
            for i in range(len(self.layers[layer_num]) - 1):
                self.layers[layer_num][i].feed(prev_layer)

    def back_prop(self, target_vals):
        error = 0
        output_layer = self.layers[-1]
        for i in range(len(output_layer) - 1):
            delta = float(target_vals[i]) - output_layer[i].output_val
            error += delta * delta
        error /= (len(output_layer) - 1)
        error = sqrt(error)

        self.recent_average_error = (self.recent_average_error * self.recent_average_smoothing_factor + self.error)/(self.recent_average_smoothing_factor + 1)
        for i in range(len(output_layer) - 1):
            output_layer[i].calc_output_gradient(target_vals[i])

        for layer_num in range(len(self.layers) - 2, 0, -1):
            hidden_layer = self.layers[layer_num]
            next_layer = self.layers[layer_num + 1]
            for i in range(len(hidden_layer)):
                hidden_layer[i].calc_hidden_gradient(next_layer)

        for layer_num in range(len(self.layers) - 1, 0, -1):
            layer = self.layers[layer_num]
            prev_layer = self.layers[layer_num - 1]

            for i in range(len(layer) - 1):
                layer[i].update_input_weights(prev_layer)

        return error

    def get_results(self):
        results = []
        for neuron in self.layers[-1]:
            results.append(neuron.output_val)
        return results
