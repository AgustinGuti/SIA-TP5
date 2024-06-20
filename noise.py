import numpy as np
import matplotlib.pyplot as plt

def salt_and_pepper_noise(vector, intensity=0.1, shape=(5, 5)):
    # noise_matrix = vector.reshape(shape[0], shape[1])
    for i in range(len(vector)):
        if np.random.random() < intensity:
            if vector[i] == 1:
                vector[i] = -1
            else:
                vector[i] = 1

    return vector

def gaussian_noise(vector, intensity=0.1, shape=(5, 5)):
    noise_matrix = np.array(vector).reshape(shape[0], shape[1])

    noise_matrix = noise_matrix + np.random.normal(0, intensity, (shape[0], shape[1]))

    noise_matrix = noise_matrix - np.min(noise_matrix)
    noise_matrix = noise_matrix / np.max(noise_matrix)

    return noise_matrix.flatten()

def print_letter(number, data, dims=(5,5)):
    # Sample 2D list
    # Convert the list of lists to a numpy array
    array_data = np.array(data).reshape(dims[0], dims[1])

    # Display the data as an image
    plt.figure()
    plt.title(f'{number}')
    plt.imshow(array_data, cmap='gray_r')  # 'gray_r' is reversed grayscale: 0=white, 1=black
    plt.axis('off')  # Turn off axis numbers and ticks
    plt.savefig(f'results/plot{number}.png', dpi=300, bbox_inches='tight')  # Adjust dpi and bbox as needed
    plt.close()
    # plt.show()

def print_group(group, title, individual_titles=None, dims=(5,5)):
    fig, axs = plt.subplots(1, len(group), figsize=(len(group)*3, 3))
    
    for i, letter in enumerate(group):
        axs[i].imshow(letter.reshape(dims), cmap='gray_r')
        if individual_titles:
            axs[i].set_title(individual_titles[i])
        axs[i].axis('off')
    
    fig.suptitle(title)
    plt.savefig(f'results/{title}.png', dpi=300, bbox_inches='tight')