import numpy as np
import matplotlib.pyplot as plt
import re
import yaml
import csv
import pandas as pd
import matplotlib.animation as animation
from NeuralNetwork import NeuralNetwork, calculate_error, get_different_pixel_count, clean_results

def main():
    # Open yaml config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    font_data, font_tags = parse_font_file('font.h')

    example_data_input = np.array(font_data)
    example_data_output = np.array(font_data)

    neural_network = NeuralNetwork(config['network']['layers'],35, 2, config['network']['function'], config['network']['beta'], config['network']['learning_rate'], config['network']['optimizer'])

    train = config['train']

    if train:
        min_error, iterations = neural_network.train(example_data_input, example_data_output, 1000000)
        print(f'Training finished in {iterations} iterations')

        neural_network.dump_weights_to_file('weights.txt')
    
    else:
        neural_network.load_weights_from_file('last_weights.txt')

    with open('error_history.csv', 'r') as f:
        df = pd.read_csv('error_history.csv')

    plt.figure()
    plt.plot(df['iteration'], df['pixel_error'])
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.title(f'Pixel Training Error over Epochs')
    plt.savefig('results/pixel_error.png')

    plt.figure()
    plt.plot(df['iteration'], df['error'])
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.title(f'Training Error over Epochs')
    plt.savefig('results/error.png')

    plt.figure()
    plt.plot(df['iteration'], df['max_pixel_error'])
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.title(f'Max Pixel Training Error over Epochs')
    plt.savefig('results/max_pixel_error.png')

    batch_size = example_data_input.shape[0]
    num_complete_batches, remainder = divmod(len(example_data_input), batch_size)
    batches = [example_data_input[i * batch_size:(i+1) + batch_size] for i in range(num_complete_batches)]

    example_data_input = np.array(batches)
    example_data_output = np.array(batches)

    total_error = 0

    for i, input_data in enumerate(example_data_input[0]):
        result = neural_network.predict(input_data)

        pixel_error = get_different_pixel_count(result, example_data_output[0][i])
        letter = font_tags[i]
        print(f'Letter: {letter} - Pixel Error: {pixel_error}')
        total_error += pixel_error

        print_number(letter, result)
        print_number(f'{letter}_clean', clean_results(result))
        print_number(f'{letter}_expected', input_data)

    print(f'Total error: {total_error}')

    latent_results = neural_network.predict_latent_space(example_data_input)

    # Scatter plot
    plt.figure()

    plt.scatter([x[0] for x in latent_results[0]], [x[1] for x in latent_results[0]])

    # Add a tag to each point
    for i, txt in enumerate(font_tags):
        plt.text(latent_results[0][i][0], latent_results[0][i][1], txt)

    plt.xlabel('Number')
    plt.ylabel('Latent Value')
    plt.title('Latent Values')
    plt.savefig('results/latent_values.png')

    # Predict new letter
    new_input = [0.3, 0.8]
    result = neural_network.predict_from_latent_space(new_input)
    print_number('new_letter', result, folder='results')
    print_number(f'new_letter_clean', clean_results(result), folder='results')

    min_latent_x = min([x[0] for x in latent_results[0]]) - 0.1
    max_latent_x = max([x[0] for x in latent_results[0]]) + 0.1

    min_latent_y = min([x[1] for x in latent_results[0]]) - 0.1
    max_latent_y = max([x[1] for x in latent_results[0]]) + 0.1


    grid_size = 100
    x = np.linspace(min_latent_x, max_latent_x, grid_size)
    y = np.linspace(min_latent_y, max_latent_y, grid_size)
    X, Y = np.meshgrid(x, y)

    Z = np.array([[neural_network.predict_from_latent_space([x_val, y_val]) for y_val in y] for x_val in x])

    def find_error_to_letter(x, y, letter):
        letter_index = font_tags.index(letter)
        return get_different_pixel_count(Z[x][y], example_data_input[0][letter_index])

    errors = np.array([[find_error_to_letter(i, j, 'm') for i in range(grid_size)] for j in range(grid_size)])

    fig = plt.figure()
    ax = fig.add_subplot(111)

    cp = ax.contourf(X, Y, errors, levels=3,  cmap='coolwarm')
    fig.colorbar(cp)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Error')
    plt.scatter([x[0] for x in latent_results[0]], [x[1] for x in latent_results[0]], color='white')
    for i, txt in enumerate(font_tags):
        plt.text(latent_results[0][i][0], latent_results[0][i][1], txt)
    plt.savefig('results/error_map.png')

    def find_less_error(x, y):
        letter_index = 0
        min_error = np.inf
        for letter, data in zip(font_tags, example_data_input[0]):
            difference = get_different_pixel_count(Z[x][y], data)
            if difference < min_error:
                min_error = difference
                letter_index = font_tags.index(letter)
        return letter_index, min_error

    errors = np.array([[find_less_error(i, j)[1] for i in range(grid_size)] for j in range(grid_size)])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    cp = ax.contourf(X, Y, errors, cmap='coolwarm')
    fig.colorbar(cp)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Error')
    plt.scatter([x[0] for x in latent_results[0]], [x[1] for x in latent_results[0]], color='white')
    for i, txt in enumerate(font_tags):
        plt.text(latent_results[0][i][0], latent_results[0][i][1], txt)
    plt.savefig('results/error_map.png')    

    plt.show()

def parse_font_file(filename):
    font_data = []
    font_tags = []
    with open(filename, 'r') as f:
        for line in f:
            match = re.search(r'\{(.+?)\}', line)  # match content inside braces
            if match:
                # split the matched string by comma, convert each item to binary array
                font_data.append([int(x, 16) for x in match.group(1).split(',')])

            match = re.search('// 0x.., (.+)', line)  # match content after '// 0x.., '
            if match:
                font_tags.append(match.group(1))

    for i, data in enumerate(font_data):
        font_data[i] = [int(x) for x in ''.join([format(byte, '05b') for byte in data])]

    return font_data, font_tags


def print_number(title, data, dims=(7,5), folder='results/letters'):

    # Sample 2D list
    # Convert the list of lists to a numpy array
    array_data = np.array(data).reshape(dims[0], dims[1])

    # Display the data as an image
    plt.figure()
    plt.title(f'{title}')
    plt.imshow(array_data, cmap='gray_r', vmin=0, vmax=1)  # 'gray_r' is reversed grayscale: 0=white, 1=black
    plt.axis('off')  # Turn off axis numbers and ticks
    plt.savefig(f'{folder}/{title}.png')
    plt.close()



if __name__ == "__main__":
    main()