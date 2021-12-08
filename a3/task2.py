from tqdm import tqdm
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import argparse

def load_real_samples(scale=False):
    X = np.load('./data/face_dataset_64x64.npy')[:20000, :, :, :]
    if scale:
        X = (X - 127.5) * 2
    return X / 255.

# We will use this function to display the output of our models throughout this notebook
def grid_plot(images, epoch='', name='', n=3, save=False, scale=False):
    if scale:
        images = (images + 1) / 2.0
    for index in range(n * n):
        plt.subplot(n, n, 1 + index)
        plt.axis('off')
        plt.imshow(images[index])
    fig = plt.gcf()
    fig.suptitle(name + '  '+ str(epoch), fontsize=14)
    if save:
        filename = 'results/generated_plot_e%03d_f.png' % (epoch+1)
        plt.savefig(filename)
        plt.close()
    plt.show()

from tensorflow.keras.layers import Dense, Flatten, Conv2D, Conv2DTranspose, Reshape

def build_conv_net(in_shape, out_shape, n_downsampling_layers=4, out_activation='sigmoid'):
    """
    Build a basic convolutional network
    """
    model = tf.keras.Sequential()
    default_args=dict(kernel_size=(3,3), strides=(2,2), padding='same', activation='relu')

    model.add(Conv2D(input_shape=in_shape, filters=128, **default_args))

    for _ in range(n_downsampling_layers):
        model.add(Conv2D(**default_args, filters=128))

    model.add(Flatten())
    model.add(Dense(out_shape, activation=out_activation) )
    model.summary()
    return model


def build_deconv_net(latent_dim, n_upsampling_layers=4, activation_out='sigmoid'):
    """
    Build a deconvolutional network for decoding latent vectors

    When building the deconvolutional architecture, usually it is best to use the same layer sizes that 
    were used in the downsampling network, however the Conv2DTranspose layers are used instead. 
    Using identical layers and hyperparameters ensures that the dimensionality of our output matches the
    input. 
    """

    model = tf.keras.Sequential()
    model.add(Dense(4 * 4 * 64, input_dim=latent_dim))
    model.add(Reshape((4, 4, 64)))
    default_args=dict(kernel_size=(3,3), strides=(2,2), padding='same', activation='relu')
    
    for i in range(n_upsampling_layers):
        model.add(Conv2DTranspose(**default_args, filters=128))

    # This last convolutional layer converts back to 3 channel RGB image
    model.add(Conv2D(filters=3, kernel_size=(3,3), activation=activation_out, padding='same'))
    model.summary()
    return model

def build_convolutional_autoencoder(data_shape, latent_dim):
    encoder = build_conv_net(in_shape=data_shape, out_shape=latent_dim)
    decoder = build_deconv_net(latent_dim, activation_out='sigmoid')

    # We connect encoder and decoder into a single model
    autoencoder = tf.keras.Sequential([encoder, decoder])
    
    # Binary crossentropy loss - pairwise comparison between input and output pixels
    autoencoder.compile(loss='binary_crossentropy', optimizer='adam')

    return autoencoder


if __name__ == "__main__":

    # Argument parser.
    p = argparse.ArgumentParser()
    p.add_argument("--create_model", action="store_true", help="specify whether to create the model: if not specified, we load from disk")
    p.add_argument("--create_dataset", action="store_true", help="specify whether to create the dataset: if not specified, we load from disk")
    p.add_argument("--bidirectional", action="store_true", help="specify whether the LSTM is bidirectional or not")
    p.add_argument('--epochs', default=25, type=int, help='number of epochs')
    FLAGS = p.parse_args()

    dataset = load_real_samples()
    # grid_plot(dataset[np.random.randint(0, 1000, 4)], name='Fliqr dataset (64x64x3)', n=2)

    image_size = dataset.shape[1:]
    latent_dim = 256

    cae = build_convolutional_autoencoder(image_size, latent_dim)

    for epoch in range(10):
        print('\nEpoch: ', epoch)

        # Note that (X=y) when training autoencoders
        # In this case we only care about qualitative performance, we don't split into train/test sets
        cae.fit(dataset, dataset, epochs=1, batch_size=64)
        
        samples = dataset[:9]
        reconstructed = cae.predict(samples)
        grid_plot(samples, epoch, name='Original', n=3, save=False)
        grid_plot(reconstructed, epoch, name='Reconstructed', n=3, save=False)
