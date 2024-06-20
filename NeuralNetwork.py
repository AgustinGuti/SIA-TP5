import numpy as np
import sys
import json
import time
from Layer import Layer, VariationalLayer

class NeuralNetwork:
    def __init__(self, layers: list[int], input_dimensions: int, output_dimensions: int, activation_function='linear', beta=100, learning_rate=0.01, 
                 optimizer='', variational=False):
        
        self.latent_layer = len(layers) + 1
        all_layers = [input_dimensions] + layers + [4] + [layer for layer in reversed(layers)] + [input_dimensions]
        self.layers: list[Layer] = []

        self.input_dimensions = input_dimensions

        i = 0   
        # while i < (len(all_layers) - 1):
            # input_dimensions = all_layers[i]
            # output_dimensions = all_layers[i + 1]
            # if i == self.latent_layer:
            #     self.layers.append(VariationalLayer((4, 2), i, activation_function, beta, learning_rate, optimizer))
            # else:
            #     if variational and i == self.latent_layer + 1:
            #         self.layers.append(Layer((2, output_dimensions), i, activation_function, beta, learning_rate, optimizer))
            #         self.layers.append(Layer((input_dimensions, output_dimensions), i, activation_function, beta, learning_rate, optimizer))
            #     else:
            #         self.layers.append(Layer((input_dimensions, output_dimensions), i, activation_function, beta, learning_rate, optimizer))
            # i += 1
        self.layers.append(Layer((35, 30), 0, activation_function, beta, learning_rate, optimizer))
        self.layers.append(Layer((30, 20), 1, activation_function, beta, learning_rate, optimizer))
        self.layers.append(Layer((20, 10), 2, activation_function, beta, learning_rate, optimizer))
        self.layers.append(Layer((10, 4), 3, activation_function, beta, learning_rate, optimizer))
        self.layers.append(VariationalLayer((4, 2), 4, activation_function, beta, learning_rate, optimizer))
        self.layers.append(Layer((2, 4), 5, activation_function, beta, learning_rate, optimizer))
        self.layers.append(Layer((4, 10), 6, activation_function, beta, learning_rate, optimizer))
        self.layers.append(Layer((10, 20), 7, activation_function, beta, learning_rate, optimizer))
        self.layers.append(Layer((20, 30), 8, activation_function, beta, learning_rate, optimizer))
        self.layers.append(Layer((30, 35), 9, activation_function, beta, learning_rate, optimizer))

        print(f'Layer: {self.layers[-1].weights.shape}')
            
        self.min_error = sys.maxsize

    def predict_latent_space(self, data_input):
        next_layer_input = data_input
        for i, layer in enumerate(self.layers):
            next_layer_input = layer.forward(next_layer_input)
            if i == self.latent_layer - 1:
                return next_layer_input
        return next_layer_input

    def predict_from_latent_space(self, data_input):
        next_layer_input = data_input
        for layer in self.layers[self.latent_layer + 1:]:
            next_layer_input = layer.forward(next_layer_input)
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
            last_time = time.time()
            while iteration < iters:
                err = 0
                # mu = np.random.randint(0, input_len)
                for mu in range(input_len):
                    # print(f'Input: {data_input[mu].shape} - mu: {mu}')
                    result = self.predict(data_input[mu])
                    err += calculate_error(result, expected_output[mu])
                    gradient = calculate_error_derivative(result, expected_output[mu])

                    for i, layer in enumerate(reversed(self.layers)):
                        if i == self.latent_layer + 1:
                            gradient = layer.backward(gradient, iteration, gradient)
                        else:
                            gradient = layer.backward(gradient, iteration)

                error = err / len(data_input)
                            
                if iteration % 500 == 0:
                    # print(f'Data Input: {data_input.shape}')
                    pixel_error = self.calculate_pixel_error(data_input[0], expected_output)
                    max_pixel_error = max(pixel_error)

                    pixel_error = sum(pixel_error)

                    time_diff = time.time() - last_time
                    last_time = time.time()

                    print(f'Iteration: {iteration} - Pixel Error Sum: {pixel_error} - Error: {error:0.5f} - Max Pixel Error: {max_pixel_error} - Max Ever Pixel Error: {best_pixel_error} - Time: {time_diff:0.3f} seconds')
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
        results = self.predict(data_input)

        # print(f'Results: {results.shape} - Expected: {expected_output[0].shape}')
    
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
