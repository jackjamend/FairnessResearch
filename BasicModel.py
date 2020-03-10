"""
Class file to create the basic classifier model
"""
import matplotlib.pyplot as plt
import tensorflow.keras as keras
import numpy as np


class BasicModel:
    def __init__(self, num_input):
        in_ = keras.layers.Input(shape=(num_input+1,))
        fc1 = keras.layers.Dense(num_input // 2, activation="relu")(in_)
        fc2 = keras.layers.Dense(num_input // 4, activation="relu")(fc1)
        fc3 = keras.layers.Dense(num_input // 8, activation="relu")(fc2)
        out = keras.layers.Dense(1, activation="sigmoid")(fc3)

        classifier = keras.models.Model(in_, out, name='classifier_model')
        classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        self.model = classifier

        # Variables for tracking information
        self.epoch_loss = np.array([])
        self.epoch_accuracy = np.array([])
        self.num_trains = 0

    def display_models(self):
        print("Model:")
        print(self.model.summary())

    def train(self, data, protected, labels, batch_size=128):
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

        # Confusion Matrix set up variables
        only_ones = predictions == 1
        only_zero = predictions == 0

        # True Positive
        true_positive = sum(labels[only_ones] == 1)
        true_negative = sum(labels[only_zero] == 0)
        false_positive = sum(labels[only_ones] == 0)  # Type 1
        false_negative = sum(labels[only_zero] == 1)  # Type 2

        return true_positive, true_negative, false_positive, false_negative

    def __update_epoch_vars(self, loss, acc):
        self.epoch_loss = np.append(self.epoch_loss, loss)
        self.epoch_accuracy = np.append(self.epoch_accuracy, acc)

    def create_figs(self, epoch, folder, fold):
        # Loss figure
        loss_path = folder + 'basic_model_loss_fold{}.png'.format(fold)
        plt.figure(figsize=(10, 15))
        plt.plot(self.epoch_loss)
        plt.title("Basic Model Loss for {} Epochs".format(epoch))
        plt.savefig(loss_path)
        plt.close()

        # Accuracy figure
        acc_path = folder + 'basic_model_acc_fold{}.png'.format(fold)
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
