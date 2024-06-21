import numpy as np
import sys
import json
import time
from Layer import Layer, VariationalLayer
from MultiLayerPerceptron import MLP

class Autoencoder:
    def __init__(self, layers: list[int], input_dimensions: int, output_dimensions: int, activation_function='linear', beta=100, learning_rate=0.01, optimizer='', variational=False):
        
        self.latent_layer = len(layers) + 1

        self.encoder: MLP = MLP([input_dimensions] + layers, input_dimensions, output_dimensions, activation_function, beta, learning_rate, optimizer)
        self.encoder.layers[-1].use_backtracking_as_final_gradient = True
        if variational:
            self.latent: VariationalLayer = VariationalLayer((layers[-1], output_dimensions), len(layers), activation_function, beta, learning_rate, optimizer)
        else:
            self.latent: Layer = Layer((layers[-1], output_dimensions), len(layers), activation_function, beta, learning_rate, optimizer)
        self.decoder: MLP = MLP([output_dimensions] + list(reversed(layers)) + [input_dimensions], output_dimensions, input_dimensions, activation_function, beta, learning_rate, optimizer, id_offset=len(layers)+1)
        self.variational = variational
        self.input_dimensions = input_dimensions            
        self.min_error = sys.maxsize
        self.iteration = 0

    def predict_latent_space(self, data_input):
        encoder_output = self.encoder.predict(data_input)
        latent_output = self.latent.forward(encoder_output)[0]
        return latent_output

    def predict_from_latent_space(self, data_input):
        decoder_output = self.decoder.predict(data_input)
        return decoder_output

    def predict(self, data_input):
        return self.predict_with_latent_data(data_input)[0]
    
    def predict_with_latent_data(self, data_input):
        encoder_output = self.encoder.predict(data_input)
        latent_output = self.latent.forward(encoder_output)
        decoder_output = self.decoder.predict(latent_output)
        return decoder_output

    def dump_weights_to_file(self, filename):
        self.encoder.dump_weights_to_file(f'{filename}_encoder.txt')
        self.latent.dump_weights_to_file(f'{filename}_latent.txt')
        self.decoder.dump_weights_to_file(f'{filename}_decoder.txt')
        with open(f'{filename}_iteration.txt', 'w') as f:
            f.write(str(self.iteration))

    def load_weights_from_file(self, filename):
        # TODO fix continue training. Need iterations data for learning rate decay
        self.encoder.load_weights_from_file(f'{filename}_encoder.txt')
        self.latent.load_weights_from_file(f'{filename}_latent.txt')
        self.decoder.load_weights_from_file(f'{filename}_decoder.txt')
        with open(f'{filename}_iteration.txt', 'r') as f:
            self.iteration = int(f.read())

    def save_latent_space(self, data_input, filename):
        latent_space = self.predict_latent_space(data_input)
        with open(filename, 'a') as f:
            for i in range(len(latent_space)):
                f.write(f'{",".join([str(i) for i in latent_space[i]])}\n')
    
    def train(self, data_input, expected_output, iters):
        batch_size = data_input.shape[0]
        num_complete_batches, remainder = divmod(len(data_input), batch_size)
        batches = [data_input[i * batch_size:(i+1) + batch_size] for i in range(num_complete_batches)]

        data_input = np.array(batches)

        output_batch_size = expected_output.shape[0]
        num_complete_batches, remainder = divmod(len(expected_output), output_batch_size)
        output_batches = [expected_output[i * output_batch_size:(i+1) + output_batch_size] for i in range(num_complete_batches)]

        expected_output = np.array(output_batches)

        input_len = len(data_input)

        best_pixel_error = sys.maxsize
        best_pixel_error_sum = sys.maxsize

        with open('error_history.csv', 'w') as f:
            f.write('iteration,max_pixel_error,pixel_error,error\n')
        
        with open('error_history.csv', 'a') as f:
            last_time = time.time()
            while self.iteration < iters:
                err = 0
                # mu = np.random.randint(0, input_len)
                for mu in range(input_len):
                    result= self.predict_with_latent_data(data_input[mu])
                    err += calculate_error(result, expected_output[mu])
                    gradient = calculate_error_derivative(result, expected_output[mu])

                    delta_outputs = self.decoder.backward(gradient, self.iteration)
                    delta_latent = self.latent.backward(delta_outputs, self.iteration)
                    self.encoder.backward(delta_latent, self.iteration)

                error = err / len(data_input)
                            
                if self.iteration % 500 == 0:
                    pixel_error = self.calculate_pixel_error(data_input[0], expected_output[0])
                    max_pixel_error = max(pixel_error)

                    pixel_error = sum(pixel_error)

                    time_diff = time.time() - last_time
                    last_time = time.time()

                    if max_pixel_error < best_pixel_error:
                        best_pixel_error = max_pixel_error
                        self.dump_weights_to_file('last_weights')

                    if pixel_error < best_pixel_error_sum:
                        best_pixel_error_sum = pixel_error
                        if max_pixel_error == best_pixel_error:
                            self.dump_weights_to_file('last_weights')

                    print(f'Iteration: {self.iteration} - Pixel Error Sum: {pixel_error} - Error: {error:0.5f} - Max Pixel Error: {max_pixel_error} - Max Ever Pixel Error: {best_pixel_error} - Time: {time_diff:0.3f} seconds')
                    f.write(f'{self.iteration}, {max_pixel_error}, {pixel_error}, {error}\n')

                
                if error < self.min_error:
                    self.min_error = error

                if pixel_error == 0:
                    break

                self.encoder.update()
                self.latent.update()
                self.decoder.update()

                self.iteration += 1
        return self.min_error, self.iteration
    
    def calculate_pixel_error(self, data_input, expected_output):
        results = self.predict(data_input)
    
        errors = [get_different_pixel_count(clean_results(results[i]), expected_output[i]) for i in range(len(results))]
        return errors

def calculate_error(predictions, expected_output, mean_std=None):
    kl = 0
    # if mean_std:
    #     kl = calculate_KL_divergence(mean_std[0], mean_std[1])
    return np.mean(np.power(expected_output - predictions, 2)) + kl

def calculate_KL_divergence(mean, std):
    return - 0.5 * np.sum(1 + std - np.power(mean, 2) - np.exp(std))

def calculate_error_derivative(predictions, expected_output):
    return 2 * (predictions - expected_output) / np.size(expected_output)

def clean_results(results):
    cleaned = np.array([1 if i > 0.5 else 0 for i in results])
    return cleaned.reshape(results.shape)

def get_different_pixel_count(data1, data2):
    clean_data1 = clean_results(data1)
    return sum([1 for i in range(len(clean_data1)) if clean_data1[i] != data2[i]])
