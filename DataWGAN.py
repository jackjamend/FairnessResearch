"""
WGAN: Generative Adversarial Network using Wasserstein distance metric.
Adapted from https://colab.research.google.com/github/timsainb/tensorflow2-generative-models/blob/master/3.0-WGAN-GP-fashion-mnist.ipynb
4/23/2020
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class DataWGAN:
    def __init__(self, dims, n_Z=64, lr_gen=1e-4, lr_disc=5e-4, gpw=10.0):
        """

        Inputs:
        - dims: size of generated example/input to discriminator
        """
        self.dims = dims
        self.lr_gen = lr_gen
        self.lr_disc = lr_disc
        self.n_Z = n_Z
        self.gradient_penalty_weight = gpw

        self.gen = self.get_generator()
        self.disc = self.get_discriminator()

        # optimizers
        self.gen_optimizer = tf.keras.optimizers.Adam(lr_gen, beta_1=0.5)
        self.disc_optimizer = tf.keras.optimizers.RMSprop(lr_disc)# train the model


    def get_generator(self):
        n_nodes = 128
        generator = [
            tf.keras.layers.Dense(units=n_nodes, name='gen_fc1', activation="relu", dtype='float32'),
            tf.keras.layers.Dense(units=2*n_nodes, name='gen_fc2', activation="relu", dtype='float32'),
            tf.keras.layers.Dense(units=4*n_nodes, name='gen_fc3', activation="relu", dtype='float32'),
            tf.keras.layers.Dense(units=self.dims[0], name='gen_out', activation="sigmoid", dtype='float32'),
        ]

        return tf.keras.Sequential(generator)

    def get_discriminator(self):
        n_nodes = 128
        discriminator = [
            tf.keras.layers.InputLayer(input_shape=self.dims, name='dis_input', dtype='float32'),
            tf.keras.layers.Dense(units=4*n_nodes, name='dis_fc1', activation="relu", dtype='float32'),
            tf.keras.layers.Dense(units=2*n_nodes, name='dis_fc2', activation="relu", dtype='float32'),
            tf.keras.layers.Dense(units=1*n_nodes, name='dis_fc3', activation="relu", dtype='float32'),
            tf.keras.layers.Dense(units=1, name='dis_out', activation="sigmoid", dtype='float32'),
        ]

        return tf.keras.Sequential(discriminator)
        
    def generate(self, z):
        return self.gen(z)

    def discriminate(self, x):
        return self.disc(x)

    def compute_loss(self, x):
        """ passes through the network and computes loss
        """
        ### pass through network
        # generating noise from a uniform distribution
        z_samp = tf.random.normal([x.shape[0], self.n_Z], dtype=tf.dtypes.float32)

        # run noise through generator
        x_gen = self.generate(z_samp)

        # discriminate x and x_gen
        logits_x = self.discriminate(x)
        logits_x_gen = self.discriminate(x_gen)

        # gradient penalty
        d_regularizer = self.gradient_penalty(x, x_gen)
        ### losses
        disc_loss = (
            tf.reduce_mean(logits_x)
            - tf.reduce_mean(logits_x_gen)
            + d_regularizer * self.gradient_penalty_weight
        )

        # losses of fake with label "1"
        gen_loss = tf.reduce_mean(logits_x_gen)

        return disc_loss, gen_loss

    def compute_gradients(self, x):
        """ passes through the network and computes loss
        """
        ### pass through network
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            disc_loss, gen_loss = self.compute_loss(x)

        # compute gradients
        gen_gradients = gen_tape.gradient(gen_loss, self.gen.trainable_variables)

        disc_gradients = disc_tape.gradient(disc_loss, self.disc.trainable_variables)

        return gen_gradients, disc_gradients

    def apply_gradients(self, gen_gradients, disc_gradients):

        self.gen_optimizer.apply_gradients(
            zip(gen_gradients, self.gen.trainable_variables)
        )
        self.disc_optimizer.apply_gradients(
            zip(disc_gradients, self.disc.trainable_variables)
        )

    def gradient_penalty(self, x, x_gen):
        epsilon = tf.random.uniform([x.shape[0], 1, 1, 1], 0.0, 1.0)
        x_hat = epsilon * x + (1 - epsilon) * x_gen
        with tf.GradientTape() as t:
            t.watch(x_hat)
            d_hat = self.discriminate(x_hat)
        gradients = t.gradient(d_hat, x_hat)
        ddx = tf.sqrt(tf.reduce_sum(gradients ** 2, axis=[1, 2]))
        d_regularizer = tf.reduce_mean((ddx - 1.0) ** 2)
        return d_regularizer

    @tf.function
    def train(self, train_x):
        gen_gradients, disc_gradients = self.compute_gradients(train_x)
        self.apply_gradients(gen_gradients, disc_gradients)
