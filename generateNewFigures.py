from matplotlib import pyplot as plt
from keras.models import load_model
from scipy.stats import norm
import numpy as np

generator = load_model('my_model.h5')

def generate_2d_manifold(generator):
    n = 35
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    grid_x = norm.ppf(np.linspace(0.01, 0.99, n))
    grid_y = norm.ppf(np.linspace(0.01, 0.99, n))
    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z_sample = np.array([[xi, yi]])
            x_decoded = generator.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size, j * digit_size: (j + 1) * digit_size] = digit
    return figure

def disp_generated(figure):
    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap='GnBu_r')
    plt.show()

figure  = generate_2d_manifold(generator)
disp_generated(figure)