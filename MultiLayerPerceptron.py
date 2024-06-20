import numpy as np
import sys
import json
from Layer import Layer  

class MLP:
    def __init__(self, layers: list[int], input_dimensions: int, output_dimensions: int, activation_function='linear', beta=100, learning_rate=0.01, optimizer='', id_offset=0):
        self.layers: list[Layer] = []
        self.input_dimensions = input_dimensions

        i = 0   
        while i < (len(layers) - 1):
            input_dimensions = layers[i]
            output_dimensions = layers[i + 1]

            self.layers.append(Layer((input_dimensions, output_dimensions), i+id_offset, activation_function, beta, learning_rate, optimizer))
            i += 1
            
        self.min_error = sys.maxsize

    def predict(self, data_input):
        next_layer_input = data_input
        for layer in self.layers:
            next_layer_input = layer.forward(next_layer_input)
        return next_layer_input
    
    def backward(self, gradient, iteration):
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient, iteration)
        return gradient
    
    def update(self):
        for layer in self.layers:
            layer.update()
    
    def dump_weights_to_file(self, filename):
        weights = [layer.weights.tolist() for layer in self.layers]
        bias = [layer.biases.tolist() for layer in self.layers]
        data = {
            'weights': weights,
            'biases': bias
        }
        with open(filename, 'w') as f:
            json.dump(data, f)

    def load_weights_from_file(self, filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        for i, layer in enumerate(self.layers):
            layer.weights = np.array(data['weights'][i])
            layer.biases = np.array(data['biases'][i])
