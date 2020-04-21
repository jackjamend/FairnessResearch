import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt

class DataGAN:
    def __init__(self, num_variables, input_size_noise=100):
        # Generator
        in_ = keras.layers.Input(shape=(input_size_noise,))
        fc1 = keras.layers.Dense(64, activation="relu")(in_)  # Make variables for testing
        fc2 = keras.layers.Dense(128, activation="relu")(fc1)
        fc3 = keras.layers.Dense(64, activation="relu")(fc2)
        fc4 = keras.layers.Dense(128, activation="relu")(fc3)
        fc5 = keras.layers.Dense(64, activation="relu")(fc4)
        out = keras.layers.Dense(num_variables, activation="sigmoid")(fc5)  # Needs to be bigger

        generator = keras.models.Model(in_, out, name='generator_model')

        # Discriminator
        in_discrim = keras.layers.Input(shape=num_variables, name="adversary_input")
        dfc1 = keras.layers.Dense(25, activation="relu")(in_discrim)
        dfc2 = keras.layers.Dense(16, activation="relu")(dfc1)
        dout = keras.layers.Dense(1, activation="sigmoid")(dfc2)
        discrim = keras.models.Model(inputs=in_discrim, outputs=dout, name='adversary_model')

        adam = keras.optimizers.Adam(2e-4)
        discrim.compile(optimizer=adam, loss="binary_crossentropy", metrics=['accuracy'])
        discrim.trainable = False

        model_input = keras.layers.Input(shape=input_size_noise)
        class_pred = generator(model_input)
        discrim_out = discrim(class_pred)
        gan_model = keras.models.Model(inputs=[model_input], outputs=[discrim_out],
                                        name="bias_model")

        loss_dict = {'adversary_model': 'binary_crossentropy'}
            # 'generator_model': 'binary_crossentropy', this makes error
                     # MSE for regression bce for classification
                       # for multiclass, categorical_crossentropy

        loss_wts = {'adversary_model': 1}
        gan_model.compile(optimizer="adam", loss=loss_dict, loss_weights=loss_wts, metrics=['accuracy'])

        self.model = gan_model
        self.generator = generator
        self.adversary = discrim
        self.input_size = num_variables
        self.gan_input_size = input_size_noise

    @staticmethod
    def make_noise(input_size, num_entries):
        gender = np.array([0,1] * (num_entries // 2)).reshape((num_entries, 1))
        noise = np.random.normal(size=(num_entries, input_size-1))
        noise_data = np.concatenate((noise, gender), axis=1)
        return noise_data

    def create_data(self, input_size, num_entries):
        noise_data = self.make_noise(input_size, num_entries)
        created_data = self.generator.predict(noise_data)
        return created_data

    def retrieve_data(self, data, idx):
        return data[idx]

    def train(self, data, num_fake_data, batch_size=128):
        num_batches = data.shape[0] // batch_size

        can_batch_loss = np.array([])
        classifier_batch_accuracy = np.array([])
        adversary_batch_accuracy = np.array([])
        for batch_idx in range(num_batches):
            idx = np.random.randint(0, data.shape[0], batch_size)  # Look at better batching

            x_real = self.retrieve_data(data, idx)
            x_fake = self.create_data(self.gan_input_size, num_fake_data)
            y_real = np.zeros(shape=(len(x_real), 1))
            y_fake = np.ones(shape=(len(x_fake), 1))

            x = np.vstack((x_real, x_fake))
            y = np.vstack((y_real, y_fake))
            adv_train_results = self.adversary.train_on_batch(x,y)  # Train discriminator
            print("adv: ",adv_train_results)
            gan_x = self.make_noise(self.gan_input_size, batch_size)

            gan_y = np.ones(shape=(len(gan_x), 1))

            model_train_results = self.model.train_on_batch(gan_x, gan_y)
            print("model: ", model_train_results)
            # Set up variables for recording training stats
            # model_loss = batch_results[0]
            # classifier_acc = batch_results[3]
            # adversary_acc = batch_results[4]
            #
            # can_batch_loss = np.append(can_batch_loss, model_loss)
            # classifier_batch_accuracy = np.append(classifier_batch_accuracy, classifier_acc)
            # adversary_batch_accuracy = np.append(adversary_batch_accuracy, adversary_acc)

        # can_loss = can_batch_loss.mean()
        # class_acc = classifier_batch_accuracy.mean()
        # adv_acc = adversary_batch_accuracy.mean()
        # self.__update_epoch_vars(can_loss, class_acc, adv_acc)