"""
Class file to create the basic classifier model
"""
import matplotlib.pyplot as plt
import tensorflow.keras as keras
import numpy as np
from sklearn.metrics import confusion_matrix


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
        cms = self.__gender_confusion_matrix(data, protected, labels)
        protected = np.reshape(protected, (len(protected), 1))
        data = np.append(data, protected, 1)

        raw_prediction = self.model.predict(data, batch_size=batch_size)
        predictions = np.where(raw_prediction > .5, 1, 0)
        
        labels = np.reshape(labels, (len(labels), 1))

        cm = confusion_matrix(labels, predictions)

        return np.append(cm.flatten() / labels.shape[0], cms).flatten()

    def __gender_confusion_matrix(self, data, protected, labels, batch_size=128):
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

        raw_male = self.model.predict(male_data, batch_size=batch_size)
        raw_female = self.model.predict(female_data, batch_size=batch_size)

        male_predictions = np.where(raw_male > .5, 1, 0)
        female_predictions = np.where(raw_female > .5, 1, 0)

        cm_male = confusion_matrix(male_labels, male_predictions)
        cm_female = confusion_matrix(female_labels, female_predictions)

        cms = np.append(cm_male.flatten() / male_data.shape[0],
                        cm_female.flatten() / female_data.shape[0]).flatten()
        return cms



    def __update_epoch_vars(self, loss, acc):
        self.epoch_loss = np.append(self.epoch_loss, loss)
        self.epoch_accuracy = np.append(self.epoch_accuracy, acc)

    @staticmethod
    def __format_data(data, labels, idxs):
        print(data.shape)
        print(labels.shape)
        print(idxs.shape)
        print(data[idxs])
        # print(labels[idxs])
        return data[idxs, :], labels[idxs]

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

    def model_save(self, path, epoch):
        self.model.save(path+'basic_at_{}.h5'.format(epoch))

    @staticmethod
    def result_graph_info():
        result_names = ["Basic Model Loss", "Basic Model Accuracy"]
        plot_paths = ['overall_loss.png', 'overall_acc.png']
        return result_names, plot_paths
