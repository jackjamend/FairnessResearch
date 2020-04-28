#import tensorflow.keras as keras
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

class DataGAN:
    def __init__(self, data_dim, latent_dim=100, n_classes=2, lr=2e-4):
        """
        Define generator, discriminator, and combined GAN models.
        Inputs:
        - data_dim: how many attributes are in the vector to be generated.
        - latent_dim: size of noise vector to be input
          (i.e., dimensionality of latent space).
        - n_classes: how many different classes are possible for the conditional input
          (e.g., 2 for male/female)
        - lr: learning rate (same for gen and discrim for now)
        """
        n_nodes = 128

        # Generator
        in_label_gen = keras.layers.Input(shape=(1,)) # Define input for conditional input
        in_noise = keras.layers.Input(shape=(latent_dim,)) # Define input for noise vector
        in_gen = keras.layers.Concatenate()([in_label_gen, in_noise])

        x = keras.layers.Dense(n_nodes, activation="relu")(in_gen)
        x = keras.layers.Dense(n_nodes*2, activation="relu")(x)
        x = keras.layers.Dense(n_nodes*4, activation="relu")(x)
        out_gen = keras.layers.Dense(data_dim, activation="sigmoid")(x)
        model_gen = keras.models.Model([in_noise, in_label_gen], out_gen, name='generator_model')

        # Discriminator
        in_label_discrim = keras.layers.Input(shape=(1,)) # Define input for conditional input
        in_data = keras.layers.Input(shape=data_dim, name="adversary_input")
        in_discrim = keras.layers.Concatenate()([in_label_discrim, in_data])
        x = keras.layers.Dense(n_nodes*4, activation="relu")(in_discrim)
#         x = keras.layers.Dropout(0.1)(x)
        x = keras.layers.Dense(n_nodes*2, activation="relu")(x)
#         x = keras.layers.Dropout(0.1)(x)
        x = keras.layers.Dense(n_nodes, activation="relu")(x)
        out_discrim = keras.layers.Dense(1, activation="sigmoid")(x)
        model_discrim = keras.models.Model(inputs=[in_data, in_label_discrim], outputs=out_discrim, name='adversary_model')

        adam = keras.optimizers.Adam(lr)
        model_discrim.compile(optimizer=adam, loss="binary_crossentropy", metrics=['accuracy'])

        # Combined GAN
        model_discrim.trainable = False
        # get noise and label inputs from generator model
        gen_noise, gen_label = model_gen.input
        # get output from the generator model
        gen_output = model_gen.output
        # connect output and label input from generator as inputs to discriminator
        out_gan = model_discrim([gen_output, gen_label])
        # define gan model as taking noise and label and outputting a classification
        model_gan = keras.models.Model([gen_noise, gen_label], out_gan, name='gan_model')
        # compile model
        opt = keras.optimizers.Adam(lr)
        model_gan.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

        self.model = model_gan
        self.generator = model_gen
        self.adversary = model_discrim
        self.output_size = data_dim
        self.latent_dim = latent_dim

    def retrieve_data(self, data, label, idx):
        return data[idx], label[idx]

    # generate points in latent space as input for the generator
    def generate_latent_points(self, latent_dim, n_samples, n_classes=2):
        # generate points in the latent space
        z_input = np.random.standard_normal(size=(n_samples, latent_dim))
        # generate conditional input
        cond_input = np.random.randint(0, n_classes, n_samples)
        return [z_input, cond_input]

    # use the generator to generate n fake examples, with class labels
    def generate_fake_samples(self, latent_dim, n_samples):
        # generate points in latent space
        z_input, cond_input = self.generate_latent_points(latent_dim, n_samples)
        # predict outputs
        data = self.generator.predict([z_input, cond_input])
        # create class labels (zeros for fake)
        y = np.zeros((n_samples, 1))
        return [data, cond_input], y

    def train(self, data, cond_label, batch_size=128, n_classes=2):
        """
        Train for 1 epoch.
        Inputs:
        - data: real examples, N x D.
        - cond_label: conditional attribute (e.g., gender), N x 1
        - batch_size: examples per batch
        - n_classes: how many distinct categories there are for the conditional attribute (e.g., 2)
        """
        num_batches = data.shape[0] // batch_size
        half_batch = batch_size // 2

        d_loss = []
        d_acc = []
        g_loss = []
        g_acc = []

        for batch_idx in range(num_batches):
            idx = np.random.randint(0, data.shape[0], half_batch)  # Look at better batching

            # First train the discriminator with real data
            x_real, cond_input = self.retrieve_data(data, cond_label, idx)
            y_real = np.ones(shape=(len(x_real), 1)) # Discrim wants to predict real data as 1
            # update discriminator model weights
            # d_loss1, _ = self.adversary.train_on_batch([x_real, cond_input], y_real)
            d_results1 = self.adversary.train_on_batch([x_real, cond_input], y_real)

            # generate 'fake' examples
            [X_fake, cond_input], y_fake = self.generate_fake_samples(self.latent_dim, half_batch)
            # update discriminator model weights
            d_results2 = self.adversary.train_on_batch([X_fake, cond_input], y_fake) # Discrim wants to predict fake data as 0

            # Next we train the generator by way of the entire GAN
            # prepare points in latent space as input for the generator
            [z_input, cond_input] = self.generate_latent_points(self.latent_dim, batch_size, n_classes)
            # create inverted labels for the fake samples
            # (since we're training the generator, we want to fool the discriminator)
            y_gan = np.ones((batch_size, 1))
            # update the generator via the discriminator's error
            g_results = self.model.train_on_batch([z_input, cond_input], y_gan)
            
            # Average discriminator loss
            d_loss_curr = (d_results1[0] + d_results2[0]) / 2
            g_loss_curr = g_results[0]

            # summarize loss on this batch
            if (batch_idx+1) % 25 == 0:
                print('> {}/{}, d={:.3f}, g={:.3f}'.format(
                    batch_idx+1, num_batches, d_loss_curr, g_loss_curr))

            d_loss.append(d_loss_curr)
            g_loss.append(g_loss_curr)
            d_acc.append((d_results1[1] + d_results2[1]) / 2)
            g_acc.append(g_results[1])

        return d_loss, g_loss, d_acc, g_acc
