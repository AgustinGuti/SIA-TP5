import numpy as np
import sys
import json
import copy

from multiprocessing import Pool

activation_functions = {
    'linear': lambda x, b=0: x,
    'sigmoid': lambda x, b: 1 / (1 + np.exp(-2*b*x)),
    'tan_h': lambda x, b: np.tanh(b*x),
}

derivative_activation_functions = {
    'linear': lambda x, b=0: 1,
    'sigmoid': lambda x, b: 2*b*np.exp(-2*b*x) / (1 + np.exp(-2*b*x))**2,
    'tan_h': lambda x, b: b*(1 - np.tanh(b*x)**2),
}    

# class NeuronParams:
#     def __init__(self, dimensions, learning_rate=0.1, activation_function='linear', beta=100, optimizer=''):
#         self.dimensions = dimensions
#         self.learning_rate = learning_rate
#         self.activation_function = activation_function
#         self.beta = beta
#         self.optimizer = optimizer

#     def __str__(self):
#         return f'Dimensions: {self.dimensions} - Learning Rate: {self.learning_rate} - Activation Function: {self.activation_function} - Beta: {self.beta} - Optimizer: {self.optimizer}\n'
   
#     def __repr__(self):
#         return self.__str__()

class LayerParams:
    def __init__(self, dimensions, learning_rate=0.1, activation_function='linear', beta=100, optimizer='', neurons=1):        
        self.dimensions = dimensions
        self.neurons = neurons
        self.learning_rate = learning_rate
        self.activation_function = activation_function
        self.beta = beta
        self.optimizer = optimizer

    def __str__(self):
        return f'Dimensions: {self.dimensions} - Learning Rate: {self.learning_rate} - Activation Function: {self.activation_function} - Beta: {self.beta} - Optimizer: {self.optimizer} - Neurons: {self.neurons}\n'
        
    def __repr__(self):
        return self.__str__()

class NeuralNetwork:
    def __init__(self, params: list[LayerParams]):
        print(f'Params: {params}')
        # self.layers = [Layer(params[i], i, params[i+2].neurons) for i in range(len(params)-2)] + [Layer(params[-1], len(params), 35)]
        self.layers = []

        i = 0   
        while i < (len(params) - 1):
            self.layers.append(Layer(params[i], i))
            i += 1
            
        # self.layers = [Layer(params[i], i, len(params[i+1])) for i in range(len(params)-1)] + [Layer(params[-1], len(params), 35)]
        self.min_error = sys.maxsize

    def predict(self, data_input):
        next_layer_input = data_input
        for layer in self.layers:
            next_layer_input = layer.process(next_layer_input)
        return next_layer_input
    
    def predict_until_layer(self, data_input, layer):
        results = []
        next_layer_input = data_input
        for i in range(layer):
            print(f'Layer {i}- {self.layers[i].weights.shape}')
            next_layer_input = self.layers[i].process(next_layer_input)
            results.append(next_layer_input)
        return results[-1]
    
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
            layer.set_min_weights()
    
    def process(self, data_input):
        return self.predict(data_input)

    # def print_weights(self):
    #     for layer in self.layers:
    #         print(f'Layer: {layer.id}')
    #         print(f'Weights: {layer.weights}')
    def train_one_sample(self, data):
        data_input, expected_output, iteration = data
        result = self.predict(data_input)
        err = calculate_error(result, expected_output)
        gradient = calculate_error_derivative(result, expected_output)

        for layer in reversed(self.layers):
            gradient = layer.backward(gradient, iteration)

        return err
    
    def train(self, data_input, expected_output, iters):
        iteration = 0
        error_history = []
        pixel_error_history = []

        input_len = len(data_input)
        while iteration < iters:
            for mu in range(input_len):
                result = self.predict(data_input[mu])
                err = calculate_error(result, expected_output[mu])
                gradient = calculate_error_derivative(result, expected_output[mu])

                for layer in reversed(self.layers):
                    gradient = layer.backward(gradient, iteration)

            error = err / len(data_input)

            pixel_error = self.calculate_pixel_error(data_input, expected_output)
                        
            if iteration % 100 == 0:
                print(f'Iteration: {iteration} - Pixel Error: {pixel_error} - Error: {error}')

                error_history.append(error)
                pixel_error_history.append(pixel_error)

            if error < self.min_error:
                self.min_error = error

            if pixel_error == 0:
                break

            for layer in self.layers:
                layer.update()


            # print(f'Iteration: {iteration} - Error: {error} - Min Error: {self.min_error} {improved}')

            iteration += 1
        return self.min_error, iteration, error_history, pixel_error_history
    
    def calculate_pixel_error(self, data_input, expected_output):
        results = self.predict(data_input)
        errors = [get_different_pixel_count(clean_results(results[i]), expected_output[i]) for i in range(len(results))]
        return np.max(errors)
        # max_error = 0
        # results = self.predict(data_input)
        # for i in range(len(results)):
        #     result = clean_results(results[i])

        #     different_pixels = get_different_pixel_count(result, expected_output[i])
        #     if different_pixels > max_error:
        #         max_error = different_pixels
        # return max_error

def calculate_error(predictions, expected_output):
    cleaned = clean_results(predictions)
    return np.mean(np.power(expected_output - cleaned, 2))

    return get_different_pixel_count(predictions, expected_output)

    return sum(sum((expected_output[i] - predictions[i])**2 for i in range(len(predictions)))/len(predictions))

def calculate_error_derivative(predictions, expected_output):
    cleaned = clean_results(predictions)
    return 2 * (cleaned - expected_output) / np.size(predictions)

def clean_results(results):
    return [1 if i > 0.95 else 0 for i in results]

def get_different_pixel_count(data1, data2):
    clean_data1 = clean_results(data1)
    return sum([1 for i in range(len(clean_data1)) if clean_data1[i] != data2[i]])

class Layer:
    def __init__(self, params: LayerParams, id):
        self.id = id
        self.weights = np.random.uniform(size=(params.dimensions, params.neurons))
        
        print(f'Layer: {self.id} - Weights: {self.weights.shape}')

        # self.weights = np.array([np.random.rand(params.dimensions) for _ in range(params.neurons)]) 
        # print(f'Layer: {self.id} - Weights: {self.weights.shape}')
        # self.weights = np.random.rand(params[0].dimensions

        self.biases = np.random.uniform(size=(1, params.neurons))

        self.learning_rate = params.learning_rate
        self.activation_function = params.activation_function

        self.beta = params.beta

        self.momentum_params = MomentumParams()
        self.rms_prop_params = RMSPropParams()
        self.adam_params = AdamParams()

        self.optimizer = params.optimizer
        self.last_input = None
        self.last_excitement = None
        self.last_output = None

        self.weight_acum = 0
        self.bias_acum = 0
    
    def compute_activation(self, excitement):
        return activation_functions[self.activation_function](excitement, self.beta)
    
    def compute_activation_prime(self, excitement):
        return derivative_activation_functions[self.activation_function](excitement, self.beta)

    def process(self, data_input):
        self.last_input = data_input
        self.last_excitement = np.dot(data_input, self.weights)
        # return np.dot(self.weights, data_input) + self.biases
        self.last_output = self.compute_activation(self.last_excitement)
        return self.last_output
    
    def backward(self, gradient, iteration):
        # print(f'Layer: {self.id} - Grad Shape: {gradient.shape} - Weights: {self.weights.shape} - Last Input: {self.last_data.shape}')
        weights_gradient = np.multiply(self.compute_activation_prime(self.last_excitement), gradient)
        new_gradient = np.dot(weights_gradient, self.weights.T)


        # print(f'Layer: {self.id} - last_input: {self.last_input.T.shape} - Weights: {self.weights.shape} - Weights Grad: {weights_gradient.shape}')
        self.last_input = self.last_input.reshape(-1, 1)
        weights_gradient = weights_gradient.reshape(1, -1)
        # print(f'Layer: {self.id} - Last Input: {self.last_input.shape} - Weights Grad: {weights_gradient.shape}')

        weights_error = np.dot(self.last_input, weights_gradient) 

        # print(f'Layer: {self.id} - Weights: {self.weights.shape} - Weights Grad: {weights_error.shape}')

        if self.optimizer == 'adam':
            weights_error = self.adam_params.get_delta(iteration, weights_error)

        self.weight_acum += weights_error * self.learning_rate
        self.bias_acum += np.average(weights_gradient) * self.learning_rate

        # self.bias_acum += gradient

        # return np.clip(new_gradient, -1, 1)
        return new_gradient
    
    def update(self):
        self.weights -= self.weight_acum
        self.biases -= self.bias_acum
        self.weight_acum = 0
        self.bias_acum = 0

class MomentumParams:
    def __init__(self, beta=0.9):
        self.beta = beta
        self.last_m = 0

    def get_m(self, iteration, gradient):
        m = self.beta * self.last_m + (1 - self.beta) * gradient
        self.last_m = m
        return m # / (1 - self.beta**iteration)
    
class RMSPropParams:
    def __init__(self, beta=0.9):
        self.beta = beta
        self.last_v = 0

    def get_v(self, iteration, gradient):
        v = self.beta * self.last_v + (1 - self.beta) * gradient**2
        self.last_v = v
        return v # / (1 - self.beta**iteration)
    
    def get_delta(self, iteration, gradient):
        return gradient / np.sqrt(self.get_v(iteration, gradient) + 1e-8)

class AdamParams:
    def __init__(self, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.m = None
        self.v = None
        
    def get_delta(self, iteration, gradient):
        if self.m is None:
            self.m = np.zeros(np.shape(gradient))
            self.v = np.zeros(np.shape(gradient))

        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        self.v = self.beta2 * self.v + (1 - self.beta2) * np.power(gradient, 2)

        m = np.divide(self.m, 1 - self.beta1 ** (iteration + 1))
        v = np.divide(self.v, 1 - self.beta2 ** (iteration + 1))
        return np.divide(m, (np.sqrt(v) + self.epsilon))