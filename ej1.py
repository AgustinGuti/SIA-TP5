import numpy as np
import matplotlib.pyplot as plt
import re
import yaml
import csv
import copy
import pandas as pd
import matplotlib.animation as animation
from Autoencoder import Autoencoder, calculate_error, get_different_pixel_count, clean_results
from noise import salt_and_pepper_noise, gaussian_noise

def main():
    # Open yaml config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    if config['denoising']['enable']:
        denoising_tests()
    else:
        font_data, font_tags = parse_font_file('font.h')
        example_data_input = np.array(font_data)
        example_data_output = np.array(font_data)
        neural_network = Autoencoder(config['network']['layers'],35, 2, config['network']['function'], config['network']['beta'], config['network']['learning_rate'], config['network']['optimizer'])
        train_neural_network(neural_network, example_data_input, example_data_output, font_tags, config['run_config'])
        
        if config['test']['show_errors_graph']:
            with open('error_history.csv', 'r') as f:
                df = pd.read_csv('error_history.csv')

            generate_pixel_error_graphs(df)

        batch_size = example_data_input.shape[0]
        num_complete_batches, remainder = divmod(len(example_data_input), batch_size)
        batches = [example_data_input[i * batch_size:(i+1) + batch_size] for i in range(num_complete_batches)]

        example_data_input = np.array(batches)
        example_data_output = np.array(batches)

        if config['test']['generate_alphabet']:
            predict_and_print(neural_network, example_data_input, example_data_output, font_tags)

        latent_results = neural_network.predict_latent_space(example_data_input)
        generate_latent_values_graph(neural_network, example_data_input, font_tags)
        
        if(config['new_letter']['generate']):
            generate_new_letter(config, neural_network)
        
        generate_error_map(neural_network, example_data_input, font_tags, latent_results)

        latent_space_data_generation(neural_network, fig_size=(7, 5))

    plt.show()

def denoising_tests():
    # Open yaml config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    font_data, font_tags = parse_font_file('font.h')
    example_data_input = np.array(font_data)

    data_input = copy.deepcopy(example_data_input)

    repetitions = 5
    if config['denoising']['noise_type'] == 'salt_and_pepper':
        data_input = [[salt_and_pepper_noise(data_input[i], config['denoising']['noise_level'], shape=(7, 5)) for i in range(len(example_data_input))] for _ in range(repetitions)]
    elif config['denoising']['noise_type'] == 'gaussian':
        data_input = [[gaussian_noise(data_input[i], config['denoising']['noise_level'], shape=(7, 5)) for i in range(len(example_data_input))] for _ in range(repetitions)]

    data_input = np.vstack(data_input)

    example_data_output = np.vstack([font_data for _ in range(repetitions)])
    neural_network = Autoencoder(config['network']['layers'],35, 2, config['network']['function'], config['network']['beta'], config['network']['learning_rate'], config['network']['optimizer'])
    train_neural_network(neural_network, data_input, example_data_output, font_tags, config['train'])

    new_data_input = copy.deepcopy(example_data_input)
    if config['denoising']['noise_type'] == 'salt_and_pepper':
        new_data_input = [salt_and_pepper_noise(new_data_input[i], config['denoising']['noise_level'], shape=(7, 5)) for i in range(len(example_data_input))]
    elif config['denoising']['noise_type'] == 'gaussian':
        new_data_input = [gaussian_noise(new_data_input[i], config['denoising']['noise_level'], shape=(7, 5)) for i in range(len(example_data_input))]
    
    new_data_input = np.array([new_data_input])
    if config['test']['generate_alphabet']:
        predict_and_print(neural_network, new_data_input, [example_data_output], font_tags, show_original=True)

    latent_results = neural_network.predict_latent_space(new_data_input)
    generate_error_map(neural_network, [font_data], font_tags, latent_results)

    return


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
    if "|" in title:
        title = title.replace("|", "pipe")
    plt.savefig(f'{folder}/{title}.png')
    plt.close()

def generate_latent_values_graph(neural_network, example_data_input, font_tags):
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


def generate_new_letter(config, neural_network):
    new_input = [config['new_letter']['x'], config['new_letter']['y']]
    result = neural_network.predict_from_latent_space(new_input)
    print_number(f'new_letter_{new_input[0]}_{new_input[1]}', result, folder='results')
    print_number(f'new_letter_{new_input[0]}_{new_input[1]}_clean', clean_results(result), folder='results')

def generate_error_map(neural_network, example_data_input, font_tags, latent_results):
    min_latent_x = min([x[0] for x in latent_results[0]]) - 0.1
    max_latent_x = max([x[0] for x in latent_results[0]]) + 0.1

    min_latent_y = min([x[1] for x in latent_results[0]]) - 0.1
    max_latent_y = max([x[1] for x in latent_results[0]]) + 0.1

    grid_size = 100
    x = np.linspace(min_latent_x, max_latent_x, grid_size)
    y = np.linspace(min_latent_y, max_latent_y, grid_size)
    X, Y = np.meshgrid(x, y)

    Z = np.array([[neural_network.predict_from_latent_space([x_val, y_val]) for y_val in y] for x_val in x])
    
    errors = np.array([[find_less_error(i, j, font_tags, example_data_input, Z)[1] for i in range(grid_size)] for j in range(grid_size)])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    cp = ax.contourf(X, Y, errors, cmap='coolwarm')
    fig.colorbar(cp)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Pixel difference to closest letter')
    plt.scatter([x[0] for x in latent_results[0]], [x[1] for x in latent_results[0]], color='white')
    for i, txt in enumerate(font_tags):
        plt.text(latent_results[0][i][0], latent_results[0][i][1], txt)
    plt.savefig('results/error_map2.png')    

def find_error_to_letter(x, y, letter, font_tags, example_data_input, Z):
        letter_index = font_tags.index(letter)
        return get_different_pixel_count(Z[x][y], example_data_input[0][letter_index])

def find_less_error(x, y, font_tags, example_data_input, Z):
    letter_index = 0
    min_error = np.inf
    for letter, data in zip(font_tags, example_data_input[0]):
        difference = get_different_pixel_count(Z[x][y], data)
        if difference < min_error:
            min_error = difference
            letter_index = font_tags.index(letter)
    return letter_index, min_error


def latent_space_data_generation(neural_network, fig_size=(7, 5)):
    min_latent_x = -0.4
    max_latent_x = 0.6

    min_latent_y = -0.6
    max_latent_y = 0.4

    grid_size = 15
    x = np.linspace(min_latent_x, max_latent_x, grid_size)
    y = np.linspace(min_latent_y, max_latent_y, grid_size)

    Z = np.array([[neural_network.predict_from_latent_space(np.array([x_val, y_val])) for y_val in y] for x_val in x])
    figure_size_x, figure_size_y = fig_size

    plot_figures(Z, grid_size, figure_size_x, figure_size_y)

    Z_clean = np.array([[clean_results(Z[i][j]) for j in range(len(Z[i]))] for i in range(len(Z))])

    plot_figures(Z_clean, grid_size, figure_size_x, figure_size_y)


def plot_figures(data, grid_size, figure_size_x, figure_size_y):
    figure = np.zeros((figure_size_x * grid_size, figure_size_y * grid_size))
    # TODO fix 90 degree rotation
    for i in range(len(data)):
        for j in range(len(data[i])):
            target_x_start = (i) * figure_size_x
            target_x_end = (i + 1) * figure_size_x
            target_y_start = (j) * figure_size_y
            target_y_end = (j + 1) * figure_size_y
            figure[target_x_start:target_x_end, target_y_start:target_y_end] = data[i][j].reshape(figure_size_x, figure_size_y)

    plt.figure(figsize=(15, 15))
    plt.imshow(figure, cmap='gray_r')
    plt.axis('off')

def generate_pixel_error_graphs(df):
    # Pixel Training Error Graph
    plt.figure()
    plt.plot(df['iteration'], df['pixel_error'])
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.title(f'Pixel Training Error over Epochs')
    last_iter = df['iteration'].iloc[-1]
    last_pixel_error = df['pixel_error'].iloc[-1]
    plt.text(last_iter, last_pixel_error, f'{last_pixel_error:.2f}', ha='right')
    plt.savefig('results/pixel_error.png')

    # Training Error Graph
    plt.figure()
    plt.plot(df['iteration'], df['error'])
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.title(f'Training Error over Epochs')
    last_error = df['error'].iloc[-1]
    plt.text(last_iter, last_error, f'{last_error:.2f}', ha='right')
    plt.savefig('results/error.png')

    # Max Pixel Error Graph
    plt.figure()
    plt.plot(df['iteration'], df['max_pixel_error'])
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.title(f'Max Pixel Error over Epochs')
    last_max_pixel_error = df['max_pixel_error'].iloc[-1]
    plt.text(last_iter, last_max_pixel_error, f'{last_max_pixel_error:.2f}', ha='right')
    plt.savefig('results/max_pixel_error.png')
    
def predict_and_print(neural_network: Autoencoder, example_data_input, example_data_output, font_tags, show_original=False):
    total_error = 0

    letters_data = []

    for i, input_data in enumerate(example_data_input[0]):
        result = neural_network.predict(input_data)

        pixel_error = get_different_pixel_count(result, example_data_output[0][i])
        letter = font_tags[i]
        print(f'Letter: {letter} - Pixel Error: {pixel_error}')
        total_error += pixel_error

        letters_data.append((letter, input_data, result))

        # if show_original:
        #     print_number(f'{letter}_expected', example_data_output[0][i])
        # print_number(letter, result)
        # print_number(f'{letter}_clean', clean_results(result))
        # print_number(f'{letter}_input', input_data)

    fig, axs = plt.subplots(4, 8)
    fig.suptitle('Letters')
    for i, (letter, input_data, result) in enumerate(letters_data):
        ax = axs[i // 8, i % 8]
        ax.imshow(result.reshape(7, 5), cmap='gray_r', vmin=0, vmax=1)
        ax.axis('off')
        ax.set_title(letter)
    plt.savefig('results/letters/letters.png')
    plt.close()

    fig, axs = plt.subplots(4, 8)
    fig.suptitle('Letters')
    for i, (letter, input_data, result) in enumerate(letters_data):
        ax = axs[i // 8, i % 8]
        ax.imshow(input_data.reshape(7, 5), cmap='gray_r', vmin=0, vmax=1)
        ax.axis('off')
        ax.set_title(letter)
    plt.savefig('results/letters/letters_input.png')
    plt.close()

    fig, axs = plt.subplots(4, 8)
    fig.suptitle('Letters')
    for i, (letter, input_data, result) in enumerate(letters_data):
        ax = axs[i // 8, i % 8]
        ax.imshow(clean_results(result).reshape(7, 5), cmap='gray_r', vmin=0, vmax=1)
        ax.axis('off')
        ax.set_title(letter)
    plt.savefig('results/letters/letters_clean.png')
    plt.close()

    print(f'Total error: {total_error}')

def train_neural_network(neural_network: Autoencoder, example_data_input, example_data_output, font_tags, run_config):
    if run_config['train']:
        if run_config['continue']:
            neural_network.load_weights_from_file('last_weights')
        min_error, iterations = neural_network.train(example_data_input, example_data_output, 1000000)
        print(f'Training finished in {iterations} iterations')
        neural_network.dump_weights_to_file('weights')    
    else:
        neural_network.load_weights_from_file('last_weights')

if __name__ == "__main__":
    main()