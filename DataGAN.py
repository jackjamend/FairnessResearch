import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt

class DataGAN:
    def __init__(self, num_input):
        # Generator
        in_ = keras.layers.Input(shape=(num_input+1,))
        fc1 = keras.layers.Dense(num_input // 2, activation="relu")(in_)
        fc2 = keras.layers.Dense(num_input // 4, activation="relu")(fc1)
        fc3 = keras.layers.Dense(num_input // 8, activation="relu")(fc2)
        out = keras.layers.Dense(1, activation="sigmoid")(fc3)

        generator = keras.models.Model(in_, out, name='generator_model')

        # Discriminator
        in_discrim = keras.layers.Input(shape=num_input, name="adversary_input")
        dfc1 = keras.layers.Dense(25, activation="relu")(in_discrim)
        dfc2 = keras.layers.Dense(16, activation="relu")(dfc1)
        dout = keras.layers.Dense(1, activation="sigmoid")(dfc2)
        discrim = keras.models.Model(inputs=in_discrim, outputs=dout, name='adversary_model')

        adam = keras.optimizers.Adam(2e-4)
        discrim.compile(optimizer=adam, loss="binary_crossentropy", metrics=['accuracy'])
        discrim.trainable = False

        model_input = keras.layers.Input(shape=num_input)
        class_pred = generator(model_input)
        protect_pred = discrim(class_pred)
        gan_model = keras.models.Model(inputs=[model_input], outputs=[class_pred, protect_pred],
                                        name="bias_model")

        loss_dict = {'classifier_model': 'binary_crossentropy',
                     # MSE for regression bce for classification
                     'adversary_model': 'binary_crossentropy'}  # for multiclass, categorical_crossentropy

        loss_wts = {'classifier_model': 5, 'adversary_model': 1}
        gan_model.compile(optimizer="adam", loss=loss_dict, loss_weights=loss_wts, metrics=['accuracy'])

        self.model = gan_model
        self.generator = generator
        self.adversary = discrim
        self.input_size = num_input

    def create_data(self, input_size, batch_size, gender):
        rand_vals = np.random.random(input_size * batch_size)
        gender = np.array([[gender]*batch_size])
        rand_vals = np.concatenate((rand_vals, gender.T), axis=1)
        created_data = self.generator.predict(rand_vals)
        return created_data

    def retrieve_data(self, data, idx):
        return data[idx]


    def train(self, data, gender, pad_size, batch_size=128):
        num_batches = data.shape[0] // batch_size

        can_batch_loss = np.array([])
        classifier_batch_accuracy = np.array([])
        adversary_batch_accuracy = np.array([])
        for batch_idx in range(num_batches):
            idx = np.random.randint(0, data.shape[0], batch_size)  # Look at better batching

            x_real = self.retrieve_data(data, idx)
            x_fake = self.create_data(self.input_size, pad_size, gender)
            y_real = np.zeros(shape=(len(x_real), 1))
            y_fake = np.ones(shape=(len(x_fake), 1))

            x = np.vstack((x_real, x_fake))
            y = np.vstack((y_real, y_fake))
            predicted_y = self.generator.predict([x])
            self.adversary.train_on_batch(x,y)


            self.adversary.train_on_batch(predicted_y, z)
            batch_results = self.model.train_on_batch(x, [y, self.__flip_label(z)])

            model_loss = batch_results[0]
            classifier_acc = batch_results[3]
            adversary_acc = batch_results[4]

            can_batch_loss = np.append(can_batch_loss, model_loss)
            classifier_batch_accuracy = np.append(classifier_batch_accuracy, classifier_acc)
            adversary_batch_accuracy = np.append(adversary_batch_accuracy, adversary_acc)

        can_loss = can_batch_loss.mean()
        class_acc = classifier_batch_accuracy.mean()
        adv_acc = adversary_batch_accuracy.mean()
        self.__update_epoch_vars(can_loss, class_acc, adv_acc)