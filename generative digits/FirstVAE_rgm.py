from __future__ import print_function

import numpy as np

from scipy.stats import norm

from keras.layers import Input, Dense, Lambda, Dropout,LeakyReLU
from keras.models import Model

from keras import backend as K
from keras import metrics
from keras.datasets import mnist
from keras.callbacks import  TensorBoard
batch_size = 100
original_dim = 784
latent_dim = 2
intermediate_dim = 256
epochs = 50
epsilon_std = 1.0

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,
                              stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon






def create_encoding_flow(encoder_input , encoder_layers, meanloglatentdim=2):
    #only create one of these
    layer_input = encoder_input
    for layer in encoder_layers:
        layer_input = layer(layer_input)

    z_mean_layer = Dense(meanloglatentdim)
    z_log_var_layer = Dense(meanloglatentdim)

    z_mean = z_mean_layer(layer_input)
    z_log_var = z_log_var_layer(layer_input)
    return z_mean,z_log_var


def create_decoding_flow(dec_input, decoder_layers):

    inp = dec_input
    for layer in decoder_layers:
        outp = layer(inp)
        inp = outp

    return outp


# ENCODER
x = Input(shape=(original_dim,))

encoder_layers = [

                    Dense(256),
                    LeakyReLU(alpha=0.5),
                    Dense(256),
                    LeakyReLU(alpha=0.4),
                    Dense(256),
                    LeakyReLU(alpha=0.3),
                    Dense(256),
                    LeakyReLU(alpha=0.2),
                    Dense(256),
                    LeakyReLU(alpha=0.1),


]

z_mean,z_log_var = create_encoding_flow(x,encoder_layers, meanloglatentdim=latent_dim)
encoder = Model(x, z_mean)



# DECODER
decoder_layers = [
                    Dense(256),
                    LeakyReLU(alpha=0.3),
                    Dense(256),
                    LeakyReLU(alpha=0.2),
                    Dense(256*2),
                    LeakyReLU(alpha=0.1),
                    Dense(256*3),
                    LeakyReLU(alpha=0.1),
                    Dense(256*2),
                    LeakyReLU(alpha=0.1),


                    Dropout(.20),

                    Dense(256 * 2),
                    LeakyReLU(alpha=0.3),
                    Dense(256 * 4),
                    LeakyReLU(alpha=0.2),
                    Dense(256 * 8),
                    LeakyReLU(alpha=0.1),

                    Dropout(.20),

                    Dense(256 * 2),
                    LeakyReLU(alpha=0.1),
                    Dense(256 * 2),
                    LeakyReLU(alpha=0.1),
                    Dense(256 * 2),
                    LeakyReLU(alpha=0.1),
                    Dense(256 * 2),
                    LeakyReLU(alpha=0.1),

                    Dense(original_dim, activation='sigmoid') ]

sampling_layer = Lambda(sampling, output_shape=(latent_dim,))
z = sampling_layer([z_mean, z_log_var])
decoder_out = create_decoding_flow(z,decoder_layers)
vae = Model(x, decoder_out)


# GENERATOR
generator_input = Input(shape=(latent_dim,))
generator_out = create_decoding_flow(generator_input,decoder_layers)
generator = Model(generator_input, generator_out)


def compute_loss(x,x_decoded_mean,z_log_var,z_mean):
    xent_loss = original_dim * metrics.binary_crossentropy(x, x_decoded_mean)
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    vae_loss = K.mean(xent_loss + kl_loss)
    return vae_loss
vae_loss = compute_loss(x, decoder_out, z_log_var, z_mean)
vae.add_loss(vae_loss)
vae.compile(optimizer='rmsprop', loss=None, metrics=['mse','mape'])



# LOAD,  NORMALIZE AND RESHAPE DATA
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))


vae.fit(x_train,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, None),

        callbacks=[TensorBoard(log_dir='longleakyreluDropout')])


def plot_2d_digit_classes_latent(encoder):
    x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
    plt.figure(figsize=(6, 6))
    plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
    plt.colorbar()
    plt.show()

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
    plt.imshow(figure, cmap='Greys_r')
    plt.show()


# with open('generator_architecture.json', 'w') as f: f.write(generator.to_json())
# generator.save_weights('generator.h5')
# encoder.save_weights('encoder.h5')

generator.save('my_model.h5')
is_server =True
if not is_server:
    import matplotlib.pyplot as plt
    plot_2d_digit_classes_latent(encoder)
    figure = generate_2d_manifold(generator)
    with open('figures.h','wb') as o: np.save(o,figure)
    with open('figures.h','rb') as o:  figures = np.load(o)
    disp_generated(figures)
else:
    figure = generate_2d_manifold(generator)
    with open('figures.h','wb') as o: np.save(o,figure)
