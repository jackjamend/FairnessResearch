"""
Class to implement the Mixed Density Network
"""
import numpy as np
import tensorflow.keras as keras
from tensorflow.math import exp

class MDN:
    def __init__(self, num_input):
        num_output = 2  # Based on two genders given (K)

        in_ = keras.layers.Input(shape=(num_input,))
        fc1 = keras.layers.Dense(num_input // 2, activation="relu", name="fully_connected1")(in_)  # They use different activation tanh
        fc2 = keras.layers.Dense(num_input // 4, activation="relu", name="fully_connected2")(fc1)
        fc3 = keras.layers.Dense(num_input // 8, activation="relu")(fc2)

        # Mixture Density Outputs
        mu_output = keras.layers.Dense((num_input*num_output), activation=None, name="mean_layer")(fc3)
        variance_layer = keras.layers.Dense(num_output, activation=None, name="variance_layer")(fc3)
        var_output = keras.layers.Lambda(lambda x: exp(x), output_shape=(num_output,), name="exp_var_layer")(variance_layer)
        pi_output = keras.layers.Dense(num_output, acitvation="softmax", name="pi_layer")(fc3)

        model = keras.models.Model(in_, [mu_output, var_output, pi_output], name="MDN")
        adam = keras.optimizers.Adam()

        # TODO: Can I compile the model here? Is loss the custom loss function? Similar to CAN
        model.compile(optimizer="adam")
        self.model = model

    def display_model(self):
        print("Model:")
        print(self.model.summary())
        pass

    def train(self, data, protected, labels, batch_size):
        pass

    def test(self, data, protected, labels, batch_size):
        pass

    def create_figs(self, epoch, folder, fold):
        pass

    @staticmethod
    def result_graph_info():
        pass
