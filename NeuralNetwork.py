import numpy as np
import sys
import json

from multiprocessing import Pool
from Layer import Layer  

class NeuralNetwork:
    def __init__(self, layers: list[int], input_dimensions: int, output_dimensions: int, activation_function='linear', beta=100, learning_rate=0.01, optimizer=''):
        
        self.latent_layer = len(layers) + 1
        all_layers = [input_dimensions] + layers + [output_dimensions] + [layer for layer in reversed(layers)] + [input_dimensions]
        self.layers: list[Layer] = []

        self.input_dimensions = input_dimensions

        i = 0   
        while i < (len(all_layers) - 1):
            input_dimensions = all_layers[i]
            output_dimensions = all_layers[i + 1]

            self.layers.append(Layer((input_dimensions, output_dimensions), i, activation_function, beta, learning_rate, optimizer))
            i += 1

        print(f'Layer: {self.layers[-1].weights.shape}')
            
        self.min_error = sys.maxsize

    # TODO: Check if this is correct
    def predict_latent_space(self, data_input):
        next_layer_input = data_input
        for i, layer in enumerate(self.layers):
            next_layer_input = layer.forward(next_layer_input)
            if i == self.latent_layer - 1:
                return next_layer_input
        return next_layer_input

    def predict(self, data_input):
        next_layer_input = data_input
        for layer in self.layers:
            next_layer_input = layer.forward(next_layer_input)
        return next_layer_input
    
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
    
    def train(self, data_input, expected_output, iters):
        iteration = 0

        batch_size = data_input.shape[0]
        num_complete_batches, remainder = divmod(len(data_input), batch_size)
        batches = [data_input[i * batch_size:(i+1) + batch_size] for i in range(num_complete_batches)]

        data_input = np.array(batches)
        expected_output = np.array(batches)

        input_len = len(data_input)

        best_pixel_error = sys.maxsize
        best_pixel_error_sum = sys.maxsize

        with open('error_history.csv', 'w') as f:
            f.write('iteration,max_pixel_error,pixel_error,error\n')
        
        with open('error_history.csv', 'a') as f:
            while iteration < iters:
                err = 0
                # mu = np.random.randint(0, input_len)
                for mu in range(input_len):
                    result = self.predict(data_input[mu])
                    err += calculate_error(result, expected_output[mu])
                    gradient = calculate_error_derivative(result, expected_output[mu])

                    for layer in reversed(self.layers):
                        gradient = layer.backward(gradient, iteration)

                error = err / len(data_input)
                            
                if iteration % 500 == 0:
                    pixel_error = self.calculate_pixel_error(data_input, expected_output)
                    max_pixel_error = max(pixel_error)

                    pixel_error = sum(pixel_error)

                    print(f'Iteration: {iteration} - Pixel Error Sum: {pixel_error} - Error: {error:0.5f} - Max Pixel Error: {max_pixel_error} - Max Ever Pixel Error: {best_pixel_error}')
                    f.write(f'{iteration}, {max_pixel_error}, {pixel_error}, {error}\n')

                    if max_pixel_error < best_pixel_error:
                        best_pixel_error = max_pixel_error
                        self.dump_weights_to_file('last_weights.txt')

                    if pixel_error < best_pixel_error_sum:
                        best_pixel_error_sum = pixel_error
                        if max_pixel_error == best_pixel_error:
                            self.dump_weights_to_file('last_weights.txt')

                if error < self.min_error:
                    self.min_error = error

                if pixel_error == 0:
                    break

                for layer in reversed(self.layers):
                    layer.update()

                iteration += 1
        return self.min_error, iteration
    
    def calculate_pixel_error(self, data_input, expected_output):
        results = self.predict(data_input)[0]
    
        errors = [get_different_pixel_count(clean_results(results[i]), expected_output[0][i]) for i in range(len(results))]
        return errors

def calculate_error(predictions, expected_output):
    return np.mean(np.power(expected_output - predictions, 2))

def calculate_error_derivative(predictions, expected_output):
    return 2 * (predictions - expected_output) / np.size(expected_output)

def clean_results(results):
    cleaned = np.array([1 if i > 0.5 else 0 for i in results])
    return cleaned.reshape(results.shape)

def get_different_pixel_count(data1, data2):
    clean_data1 = clean_results(data1)
    return sum([1 for i in range(len(clean_data1)) if clean_data1[i] != data2[i]])
