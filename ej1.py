import numpy as np
import matplotlib.pyplot as plt
import re
import yaml
import matplotlib.animation as animation
from NeuralNetwork import NeuralNetwork, NeuronParams, calculate_error

def main():
    # Open yaml config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)


    font_data = parse_font_file('font.h')

    example_data_input = np.array(font_data)
    example_data_output = np.array(font_data)

    previous_dimension = len(example_data_input[0])

    layers = []
    for layer in config['network']['layers']:
        layers.append([NeuronParams(previous_dimension, layer['lr'], layer['function'], layer['beta'], layer['opt']) for _ in range(layer['neuron'])])
        previous_dimension = layer['neuron']


    neural_network = NeuralNetwork(layers)
    train_errors = []
      
    best_weights_history = []
    best_biases_history = []
    error_history = []
    for _ in range(50000):
        min_error, iterations, best_weights, best_biases, error = neural_network.train(example_data_input, example_data_output, 1)
        best_weights_history.append(best_weights)
        best_biases_history.append(best_biases)
        error_history.append(error)
        train_errors.append(min_error)

    plt.figure()

    plt.plot(range(len(train_errors)), train_errors, label='Training Error')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.title(f'Training Error over Epochs')

    print(f'Min error: {min_error}')

    for i, input_data in enumerate(example_data_input):
        result = neural_network.predict(input_data)
        pixel_error = get_different_pixel_count(result, example_data_output[i])
        print(f'Number: {i} - Pixel Error: {pixel_error} - Percentage: {pixel_error/len(result)*100}')

        print_number(i, result)
        print_number(i+100, input_data)

    latent_results = neural_network.predict_until_layer(example_data_input, 4, True)
    print(latent_results)



    # fig, ax = plt.subplots()
    # x = np.linspace(-2, 2, 100)
    # y = np.linspace(-2, 2, 100)
    # X, Y = np.meshgrid(x, y)

    # Z = np.array([[neural_network.predict([x_val, y_val])[0] for x_val in x] for y_val in y])
    
    # cp = ax.contourf(X, Y, Z, levels=0,  cmap='coolwarm')

    # ax.scatter(example_data_input[:, 0], example_data_input[:, 1], c=example_data_output, cmap='coolwarm')

    plt.show()

def clean_results(results):
    return [1 if x > 0.5 else 0 for x in results]

def get_different_pixel_count(data1, data2):
    clean_data1 = clean_results(data1)
    return sum([1 for i in range(len(clean_data1)) if clean_data1[i] != data2[i]])

def parse_font_file(filename):
    font_data = []
    with open(filename, 'r') as f:
        for line in f:
            match = re.search(r'\{(.+?)\}', line)  # match content inside braces
            if match:
                # split the matched string by comma, convert each item to binary array
                font_data.append([int(x, 16) for x in match.group(1).split(',')])

    for i, data in enumerate(font_data):
        font_data[i] = [int(x) for x in ''.join([format(byte, '05b') for byte in data])]

    return font_data


def print_number(number, data, dims=(7,5)):
    # Sample 2D list
    # Convert the list of lists to a numpy array
    array_data = np.array(data).reshape(dims[0], dims[1])

    # Display the data as an image
    plt.figure()
    plt.title(f'Number: {number}')
    plt.imshow(array_data, cmap='gray_r')  # 'gray_r' is reversed grayscale: 0=white, 1=black
    plt.axis('off')  # Turn off axis numbers and ticks
    plt.savefig(f'results/number_{number}.png')
    plt.close()



if __name__ == "__main__":
    main()
