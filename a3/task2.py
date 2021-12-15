from tqdm import tqdm
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import argparse

from tensorflow.keras.layers import Dense, Flatten, Conv2D, Conv2DTranspose, Reshape


#######
# CAE #
#######
def load_real_samples_face_dataset(scale=False):
    X = np.load('./face_dataset_64x64.npy')[:20000, :, :, :]
    if scale:
        X = (X - 127.5) * 2
    return X / 255.

def load_real_samples_cats_dataset(scale=False):
    X = np.load('./cats.npy')
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
    model.add(Reshape((4, 4 , 64)))
    default_args=dict(kernel_size=(3,3), strides=(2,2), padding='same', activation='relu')
    
    for i in range(n_upsampling_layers):
        model.add(Conv2DTranspose(**default_args, filters=128))

    # This last convolutional layer converts back to 3 channel RGB image
    model.add(Conv2D(filters=3, kernel_size=(3,3), activation=activation_out, padding='same'))
    model.summary()

    # exit(0)

    return model

def build_convolutional_autoencoder(data_shape, latent_dim):
    encoder = build_conv_net(in_shape=data_shape, out_shape=latent_dim)
    decoder = build_deconv_net(latent_dim, activation_out='sigmoid')

    # We connect encoder and decoder into a single model
    autoencoder = tf.keras.Sequential([encoder, decoder])
    
    # Binary crossentropy loss - pairwise comparison between input and output pixels
    autoencoder.compile(loss='binary_crossentropy', optimizer='adam')

    return autoencoder

#######
# VAE #
#######
class Sampling(tf.keras.layers.Layer):
    """
    Custom layer for the variational autoencoder
    It takes two vectors as input - one for means and other for variances of the latent variables described by a multimodal gaussian
    Its output is a latent vector randomly sampled from this distribution
    """
    def call(self, inputs):
        z_mean, z_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_var) * epsilon

def build_vae(data_shape, latent_dim):

    # Building the encoder - starts with a simple downsampling convolutional network  
    encoder = build_conv_net(data_shape, latent_dim*2)
    
    # Adding special sampling layer that uses the reparametrization trick 
    z_mean = Dense(latent_dim)(encoder.output)
    z_var = Dense(latent_dim)(encoder.output)
    z = Sampling()([z_mean, z_var])
    
    # Connecting the two encoder parts
    encoder = tf.keras.Model(inputs=encoder.input, outputs=z)

    # Defining the decoder which is a regular upsampling deconvolutional network
    decoder = build_deconv_net(latent_dim, activation_out='sigmoid')
    vae = tf.keras.Model(inputs=encoder.input, outputs=decoder(z))
    
    # Adding the special loss term
    kl_loss = -0.5 * tf.reduce_sum(z_var - tf.square(z_mean) - tf.exp(z_var) + 1)
    vae.add_loss(kl_loss/tf.cast(tf.keras.backend.prod(data_shape), tf.float32))

    vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='binary_crossentropy')

    return encoder, decoder, vae

#######
# GAN #
#######
def build_gan(data_shape, latent_dim, lr=0.0002, beta_1=0.5):
    optimizer = tf.optimizers.Adam(learning_rater=lr, beta_1=beta_1)

    # Usually thew GAN generator has tanh activation function in the output layer
    generator = build_deconv_net(latent_dim, activation_out='tanh')
    
    # Build and compile the discriminator
    discriminator = build_conv_net(in_shape=data_shape, out_shape=1) # Single output for binary classification
    discriminator.compile(loss='binary_crossentropy', optimizer=optimizer)
    
    # End-to-end GAN model for training the generator
    discriminator.trainable = False
    true_fake_prediction = discriminator(generator.output)
    GAN = tf.keras.Model(inputs=generator.input, outputs=true_fake_prediction)
    GAN = tf.keras.models.Sequential([generator, discriminator])
    GAN.compile(loss='binary_crossentropy', optimizer=optimizer)
    
    return discriminator, generator, GAN

def run_generator(generator, n_samples=100):
    """
    Run the generator model and generate n samples of synthetic images using random latent vectors
    """
    latent_dim = generator.layers[0].input_shape[-1]
    generator_input = np.random.randn(n_samples, latent_dim)

    return generator.predict(generator_input)

def get_batch(generator, dataset, batch_size=64):
    """
    Gets a single batch of samples (X) and labels (y) for the training the discriminator.
    One half from the real dataset (labeled as 1s), the other created by the generator model (labeled as 0s).
    """
    batch_size //= 2 # Split evenly among fake and real samples

    fake_data = run_generator(generator, n_samples=batch_size)
    real_data = dataset[np.random.randint(0, dataset.shape[0], batch_size)]

    X = np.concatenate([fake_data, real_data], axis=0)
    y = np.concatenate([np.zeros([batch_size, 1]), np.ones([batch_size, 1])], axis=0)

    return X, y

def train_gan(generator, discriminator, gan, dataset, latent_dim, n_epochs=20, batch_size=64):

    batches_per_epoch = int(dataset.shape[0] / batch_size / 2)
    for epoch in range(n_epochs):
        for batch in tqdm(range(batches_per_epoch)):
            
            # 1) Train discriminator both on real and synthesized images
            X, y = get_batch(generator, dataset, batch_size=batch_size)
            discriminator_loss = discriminator.train_on_batch(X, y)

            # 2) Train generator (note that now the label of synthetic images is reversed to 1)
            X_gan = np.random.randn(batch_size, latent_dim)
            y_gan = np.ones([batch_size, 1])
            generator_loss = gan.train_on_batch(X_gan, y_gan)

        noise = np.random.randn(16, latent_dim)
        images = generator.predict(noise)
        grid_plot(images, epoch, name='GAN generated images', n=3, save=False, scale=True)


if __name__ == "__main__":

    # Argument parser.
    p = argparse.ArgumentParser()
    p.add_argument("--cae", action="store_true", help="specify to run the convolutional autoencoder")
    p.add_argument("--vae", action="store_true", help="specify to run the variational autoencoder")
    p.add_argument("--gan", action="store_true", help="specify to run the generative adversarial networks")

    # p.add_argument("--bidirectional", action="store_true", help="specify whether the LSTM is bidirectional or not")
    # p.add_argument('--epochs', default=25, type=int, help='number of epochs')
    FLAGS = p.parse_args()

    # Load dataset ### Change functions here to switch between datasets
    dataset = load_real_samples_cats_dataset()
    dataset_scaled = load_real_samples_cats_dataset(scale=True)

    # Cats images are still in 1d vector format, needs reshaping
    dataset = dataset.reshape((dataset.shape[0], 64, 64, 3))
    dataset_scaled = dataset_scaled.reshape((dataset_scaled.shape[0], 64, 64, 3))

    print('dataset shape: ' + str(dataset.shape))
    grid_plot(dataset[np.random.randint(0, 1000, 4)], name='dataset images', n=2)

    print(dataset[0])
    print(dataset_scaled[0])

    if FLAGS.cae:
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

    if FLAGS.vae:
        # Training the VAE model
        latent_dim = 32
        encoder, decoder, vae = build_vae(dataset.shape[1:], latent_dim)

        # Generate random vectors that we will use to sample our latent space
        latent_vectors = np.random.randn(9, latent_dim)
        for epoch in range(20):
            vae.fit(dataset, dataset, epochs=1, batch_size=4)
            
            images = decoder(latent_vectors)
            grid_plot(images, epoch, name='VAE generated images', n=3, save=True)

    if FLAGS.gan:
        latent_dim = 128
        discriminator, generator, gan = build_gan(dataset.shape[1:], latent_dim)
        dataset_scaled = load_real_samples_cats_dataset(scale=True)

        train_gan(generator, discriminator, gan, dataset_scaled, latent_dim)