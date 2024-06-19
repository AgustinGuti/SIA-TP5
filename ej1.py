import numpy as np
import matplotlib.pyplot as plt
import re
import yaml
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

    if True:
        min_error, iterations, error_history, pixel_error_history = neural_network.train(example_data_input, example_data_output, 1000000)
        print(f'Training finished in {iterations} iterations')

        neural_network.dump_weights_to_file('weights.txt')

        plt.figure()

        plt.plot(range(len(pixel_error_history)), pixel_error_history, label='Training Error')
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Error')
        plt.title(f'Pixel Training Error over Epochs')
        plt.savefig('results/pixel_error.png')

        plt.figure()

        plt.plot(range(len(error_history)), error_history, label='Training Error')
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Error')
        plt.title(f'Training Error over Epochs')
        plt.savefig('results/error.png')

        print(f'Min error: {min_error}')
    
    else:
        neural_network.load_weights_from_file('best_weights.txt')

    example_data_input = np.reshape(example_data_input, (len(example_data_input), 35, 1))

    for i, input_data in enumerate(example_data_input):
        result = neural_network.predict(input_data)
        pixel_error = get_different_pixel_count(result, example_data_output[i])
        print(f'Number: {i} - Pixel Error: {pixel_error} - Percentage: {pixel_error/len(result)*100}')

        print_number(i, result)
        print_number(f'{i}_clean', clean_results(result))
        print_number(f'{i}_expected', input_data)

    latent_results = [neural_network.predict_latent_space(example_data_input[i]) for i in range(len(example_data_input))]
    print(latent_results)

    # Scatter plot
    plt.figure()

    plt.scatter([x[0] for x in latent_results], [x[1] for x in latent_results])

    # Add a tag to each point
    for i, txt in enumerate(font_tags):
        plt.text(latent_results[i][0], latent_results[i][1], txt)

    plt.xlabel('Number')
    plt.ylabel('Latent Value')
    plt.title('Latent Values')
    plt.savefig('results/latent_values.png')


    # fig, ax = plt.subplots()
    # x = np.linspace(-2, 2, 100)
    # y = np.linspace(-2, 2, 100)
    # X, Y = np.meshgrid(x, y)

    # Z = np.array([[neural_network.predict([x_val, y_val])[0] for x_val in x] for y_val in y])
    
    # cp = ax.contourf(X, Y, Z, levels=0,  cmap='coolwarm')

    # ax.scatter(example_data_input[:, 0], example_data_input[:, 1], c=example_data_output, cmap='coolwarm')

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


def print_number(title, data, dims=(7,5)):
    # Sample 2D list
    # Convert the list of lists to a numpy array
    array_data = np.array(data).reshape(dims[0], dims[1])

    # Display the data as an image
    plt.figure()
    plt.title(f'{title}')
    plt.imshow(array_data, cmap='gray_r')  # 'gray_r' is reversed grayscale: 0=white, 1=black
    plt.axis('off')  # Turn off axis numbers and ticks
    plt.savefig(f'results/{title}.png')
    plt.close()



if __name__ == "__main__":
    main()