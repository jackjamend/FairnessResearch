"""
Class file to create the split classifier model. Train two separate models on the data by gender
"""
import matplotlib.pyplot as plt
import tensorflow.keras as keras
import numpy as np
from sklearn.metrics import confusion_matrix

class SplitModel:
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
        self.epoch_loss_male = np.array([])
        self.epoch_loss_female = np.array([])
        self.epoch_accuracy_male = np.array([])
        self.epoch_accuracy_female = np.array([])
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

        male_data = data[male_idx]
        male_protected = protected[male_idx]
        male_labels = labels[male_idx]

        female_data = data[female_idx]
        female_protected = protected[female_idx]
        female_labels = labels[female_idx]

        self.num_trains += 1
        num_batches = data.shape[0] // batch_size

        batch_loss_male = np.array([])
        batch_loss_female = np.array([])
        batch_accuracy_male = np.array([])
        batch_accuracy_female= np.array([])
        for batch_idx in range(num_batches):
            rand_male_idx = np.random.randint(0, male_data.shape[0], batch_size)  # Look at better batching
            rand_female_idx = np.random.randint(0, female_data.shape[0], batch_size)

            # get values from data
            x_male = male_data[rand_male_idx]
            x_female = female_data[rand_female_idx]
            y_male = male_labels[rand_male_idx]
            y_female = female_labels[rand_female_idx]
            z_male = male_protected[rand_male_idx]
            z_female = female_protected[rand_female_idx]

            z_male = np.reshape(z_male, (len(z_male), 1))
            x_male = np.append(x_male, z_male, 1)
            z_female = np.reshape(z_female, (len(z_female), 1))
            x_female = np.append(x_female, z_female, 1)

            # Predict
            loss_male, acc_male = self.male_model.train_on_batch(x_male, y_male)
            loss_female, acc_female = self.female_model.train_on_batch(x_female, y_female)
            batch_loss_male = np.append(batch_loss_male, loss_male)
            batch_loss_female = np.append(batch_loss_female, loss_female)
            batch_accuracy_male = np.append(batch_accuracy_male, acc_male)
            batch_accuracy_female = np.append(batch_accuracy_female, acc_female)

        loss_male = batch_loss_male.mean()
        loss_female = batch_loss_female.mean()
        acc_male = batch_accuracy_male.mean()
        acc_female = batch_accuracy_female.mean()
        self.__update_epoch_vars(loss_male, loss_female, acc_male, acc_female)

    def test(self, data, protected, labels, batch_size=128):
        male_idx = protected == 0
        female_idx = protected == 1

        male_data = data[male_idx]
        male_protected = protected[male_idx]
        male_labels = labels[male_idx]

        female_data = data[female_idx]
        female_protected = protected[female_idx]
        female_labels = labels[female_idx]

        male_protected = np.reshape(male_protected, (len(male_protected), 1))
        female_protected = np.reshape(female_protected, (len(female_protected), 1))

        male_data = np.append(male_data, male_protected, 1)
        female_data = np.append(female_data, female_protected, 1)

        results_male = self.male_model.evaluate(male_data, male_labels, batch_size=batch_size)
        results_female = self.female_model.evaluate(female_data, female_labels, batch_size=batch_size)
        return results_male, results_female

    def confusion_matrix(self, data, protected, labels, batch_size=128):
        male_idx = protected == 0
        female_idx = protected == 1

        male_data = data[male_idx]
        male_protected = protected[male_idx]
        male_labels = labels[male_idx]

        female_data = data[female_idx]
        female_protected = protected[female_idx]
        female_labels = labels[female_idx]

        male_protected = np.reshape(male_protected, (len(male_protected), 1))
        female_protected = np.reshape(female_protected, (len(female_protected), 1))

        male_data = np.append(male_data, male_protected, 1)
        female_data = np.append(female_data, female_protected, 1)

        raw_male = self.male_model.predict(male_data, batch_size=batch_size)
        raw_female = self.female_model.predict(female_data, batch_size=batch_size)

        male_predictions = np.where(raw_male > .5, 1, 0)
        female_predictions = np.where(raw_female > .5, 1, 0)

        male_labels = np.reshape(male_labels, (len(male_labels), 1))
        female_labels = np.reshape(female_labels, (len(female_labels), 1))

        cm_male = confusion_matrix(male_labels, male_predictions)
        cm_female = confusion_matrix(female_labels, female_predictions)

        cms = np.array([cm_male.flatten() / male_labels.shape[0],
                        cm_female.flatten() / female_labels.shape[0]]).flatten()

        return cms

    def __update_epoch_vars(self, loss_male, loss_female, acc_male, acc_female):

        self.epoch_loss_male = np.append(self.epoch_loss_male, loss_male)
        self.epoch_loss_female = np.append(self.epoch_loss_male, loss_female)
        self.epoch_accuracy_male = np.append(self.epoch_accuracy_male, acc_male)
        self.epoch_accuracy_female = np.append(self.epoch_accuracy_female, acc_female)

    @staticmethod
    def __reshape_1d(arr):
        return np.reshape(arr, (len(arr), 1))

    def create_figs(self, epoch, folder):
        # Loss figure
        loss_path = folder + 'male_model_loss_epoch{}.png'.format(epoch)
        plt.figure(figsize=(10, 15))
        plt.plot(self.epoch_loss_male)
        plt.title("Male Model Loss for {} Epochs".format(epoch))
        plt.savefig(loss_path)
        plt.close()

        loss_path = folder + 'female_model_loss_epoch{}.png'.format(epoch)
        plt.figure(figsize=(10, 15))
        plt.plot(self.epoch_loss_female)
        plt.title("Female Model Loss for {} Epochs".format(epoch))
        plt.savefig(loss_path)
        plt.close()

        # Accuracy figure
        acc_path = folder + 'male_model_acc_epoch{}.png'.format(epoch)
        plt.figure(figsize=(10, 15))
        plt.plot(self.epoch_accuracy_male)
        plt.title("Male Model Accuracy for {} Epochs".format(epoch))
        plt.savefig(acc_path)
        plt.close()

        acc_path = folder + 'female_model_acc_epoch{}.png'.format(epoch)
        plt.figure(figsize=(10, 15))
        plt.plot(self.epoch_accuracy_female)
        plt.title("Female Model Accuracy for {} Epochs".format(epoch))
        plt.savefig(acc_path)
        plt.close()

    def model_save(self, path, epoch):
        self.male_model.save(path+'male_model_at_{}.h5'.format(epoch))
        self.female_model.save(path + 'female_model_at_{}.h5'.format(epoch))

    @staticmethod
    def result_graph_info():
        result_names = ["Basic Model Loss", "Basic Model Accuracy"]
        plot_paths = ['overall_loss.png', 'overall_acc.png']
        return result_names, plot_paths
