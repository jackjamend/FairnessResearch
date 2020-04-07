"""
Class file to create the split classifier model. Train two separate models on the data by gender
"""
import matplotlib.pyplot as plt
import tensorflow.keras as keras
import numpy as np


class BasicModel:
    def __init__(self, num_input):
        # Male Model
        male_in_ = keras.layers.Input(shape=(num_input + 1,))
        male_fc1 = keras.layers.Dense(num_input // 2, activation="relu")(male_in_)
        male_fc2 = keras.layers.Dense(num_input // 4, activation="relu")(male_fc1)
        male_fc3 = keras.layers.Dense(num_input // 8, activation="relu")(male_fc2)
        male_out = keras.layers.Dense(1, activation="sigmoid")(male_fc3)

        male_classifier = keras.models.Model(male_in_, male_out, name='classifier_model')
        male_classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        self.male_model = male_classifier

        female_in_ = keras.layers.Input(shape=(num_input + 1,))
        female_fc1 = keras.layers.Dense(num_input // 2, activation="relu")(female_in_)
        female_fc2 = keras.layers.Dense(num_input // 4, activation="relu")(female_fc1)
        female_fc3 = keras.layers.Dense(num_input // 8, activation="relu")(female_fc2)
        female_out = keras.layers.Dense(1, activation="sigmoid")(female_fc3)

        female_classifier = keras.models.Model(female_in_, female_out, name='classifier_model')
        female_classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        self.male_model = male_classifier
        self.female_model = female_classifier

        # Variables for tracking information
        self.epoch_loss = np.array([])
        self.epoch_accuracy = np.array([])
        self.num_trains = 0

    def display_models(self):
        print("Male Model:")
        print(self.male_model.summary())

        print("Female Model:")
        print(self.male_model.summary())

    def train(self, data, protected, labels, batch_size=128):
        # Get idxs for males and females
        male_idx = protected == 0
        female_idx = protected == 1

        self.num_trains += 1
        num_batches = data.shape[0] // batch_size

        batch_loss = np.array([])
        batch_accuracy = np.array([])
        for batch_idx in range(num_batches):
            idx = np.random.randint(0, data.shape[0], batch_size)  # Look at better batching

            # get values from data
            x = data[idx]
            y = labels[idx]
            z = protected[idx]

            # Since only interested in classifier, we can use protected

            z = np.reshape(z, (len(z), 1))
            x = np.append(x, z, 1)

            loss, acc = self.model.train_on_batch(x, y)
            batch_loss = np.append(batch_loss, loss)
            batch_accuracy = np.append(batch_accuracy, acc)

        loss = batch_loss.mean()
        acc = batch_accuracy.mean()
        self.__update_epoch_vars(loss, acc)

    def test(self, data, protected, labels, batch_size=128):
        protected = np.reshape(protected, (len(protected), 1))
        data = np.append(data, protected, 1)

        results = self.model.evaluate(data, labels, batch_size=batch_size)
        return results

    def confusion_matrix(self, data, protected, labels, batch_size=128):
        protected = np.reshape(protected, (len(protected), 1))
        data = np.append(data, protected, 1)

        raw_prediction = self.model.predict(data, batch_size=batch_size)
        predictions = np.where(raw_prediction > .5, 1, 0)

        labels = np.reshape(labels, (len(labels), 1))

        # Confusion Matrix set up variables
        only_ones = predictions == 1
        only_zero = predictions == 0

        true_positive = sum(labels[only_ones] == 1)
        true_negative = sum(labels[only_zero] == 0)
        false_positive = sum(labels[only_ones] == 0)  # Type 1
        false_negative = sum(labels[only_zero] == 1)  # Type 2

        return true_positive, true_negative, false_positive, false_negative

    def __update_epoch_vars(self, loss, acc):
        self.epoch_loss = np.append(self.epoch_loss, loss)
        self.epoch_accuracy = np.append(self.epoch_accuracy, acc)

    @staticmethod
    def __reshape_1d(arr):
        return np.reshape(arr, (len(arr), 1))

    def create_figs(self, epoch, folder):
        # Loss figure
        loss_path = folder + 'basic_model_loss_epoch{}.png'.format(epoch)
        plt.figure(figsize=(10, 15))
        plt.plot(self.epoch_loss)
        plt.title("Basic Model Loss for {} Epochs".format(epoch))
        plt.savefig(loss_path)
        plt.close()

        # Accuracy figure
        acc_path = folder + 'basic_model_acc_epoch{}.png'.format(epoch)
        plt.figure(figsize=(10, 15))
        plt.plot(self.epoch_accuracy)
        plt.title("Basic Model Accuracy for {} Epochs".format(epoch))
        plt.savefig(acc_path)
        plt.close()

    @staticmethod
    def result_graph_info():
        result_names = ["Basic Model Loss", "Basic Model Accuracy"]
        plot_paths = ['overall_loss.png', 'overall_acc.png']
        return result_names, plot_paths
