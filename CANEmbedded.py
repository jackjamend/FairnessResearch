"""
Class for a CAN Embedded model.

The model will use a layer of size 2 before the output for the input into the adversary.

Jack J Amend
"""

import matplotlib.pyplot as plt
import tensorflow.keras as keras
import numpy as np


class CANEmbedded:
    def __init__(self, num_input, embedding_size=2):

        # Classifier
        in_ = keras.layers.Input(shape=(num_input,))
        fc1 = keras.layers.Dense(num_input // 2, activation="relu")(in_)
        fc2 = keras.layers.Dense(num_input // 4, activation="relu")(fc1)
        fc3 = keras.layers.Dense(num_input // 8, activation="relu")(fc2)
        adv_out = keras.layers.Dense(embedding_size, activation="relu")(fc3)
        out = keras.layers.Dense(1, activation="sigmoid")(adv_out)

        classifier = keras.models.Model(in_, [out, adv_out], name='classifier_model')

        # Discriminator
        in_discrim = keras.layers.Input(shape=embedding_size, name="classifer_output")
        dfc1 = keras.layers.Dense(25, activation="relu")(in_discrim)
        dout = keras.layers.Dense(1, activation="sigmoid")(dfc1)
        discrim = keras.models.Model(inputs=in_discrim, outputs=dout, name='adversary_model')

        adam = keras.optimizers.Adam(2e-4)
        discrim.compile(optimizer=adam, loss="binary_crossentropy", metrics=['accuracy'])
        discrim.trainable = False

        model_input = keras.layers.Input(shape=num_input)
        class_pred, class_embed = classifier(model_input)
        protect_pred = discrim(class_embed)
        bias_model = keras.models.Model(inputs=[model_input], outputs=[class_pred, protect_pred],
                                        name="bias_model")

        loss_dict = {'classifier_model': 'binary_crossentropy',
                     # MSE for regression bce for classification
                     'adversary_model': 'binary_crossentropy'}  # for multiclass, categorical_crossentropy

        loss_wts = {'classifier_model': 5, 'adversary_model': 1}
        bias_model.compile(optimizer="adam", loss=loss_dict, loss_weights=loss_wts, metrics=['accuracy'])

        self.model = bias_model
        self.classifier = classifier
        self.adversary = discrim

        # Models for tracking
        self.epoch_can_loss = np.array([])
        self.epoch_classifier_accuracy = np.array([])
        self.epoch_adversary_accuracy = np.array([])

    def display_models(self):
        print("Model:")
        print(self.model.summary())

        print("Classifier:")
        print(self.classifier.summary())

        print("Adversary:")
        print(self.adversary.summary())

    def train(self, data, protected, labels, batch_size=128):
        num_batches = data.shape[0] // batch_size

        can_batch_loss = np.array([])
        classifier_batch_accuracy = np.array([])
        adversary_batch_accuracy = np.array([])
        for batch_idx in range(num_batches):
            idx = np.random.randint(0, data.shape[0], batch_size)  # Look at better batching

            X = data[idx]
            Y = labels[idx]
            Z = protected[idx]

            predicted_y, out_embedd = self.classifier.predict([X])

            self.adversary.train_on_batch(out_embedd, Z)
            batch_results = self.model.train_on_batch(X, [Y, self.__flip_label(Z)])

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

    def test(self, data, protected, labels, batch_size=128):
        results = self.model.evaluate(data, [labels, protected], batch_size=batch_size)

        return np.take(results, [0,3,4])

    def create_figs(self, epoch, folder):
        # CAN Loss
        can_loss_title = 'CAN Embed Model Loss for {} Epochs'.format(epoch)
        can_loss_path = folder + 'can_embed_model_loss_epoch{}.png'.format(epoch)
        self.__line_graph(self.epoch_can_loss, can_loss_title, can_loss_path)

        # Classifier Accuracy
        can_class_loss_title = 'CAN Embed Classifier Accuracy for {} Epochs'.format(epoch)
        can_class_loss_path = folder + 'can_embed_classifier_model_loss_epoch{}.png'.format(epoch)
        self.__line_graph(self.epoch_classifier_accuracy, can_class_loss_title, can_class_loss_path)

        # Adversary Accuracy
        can_adv_loss_title = 'CAN Embed Adversary Accuracy for {} Epochs'.format(epoch)
        can_adv_loss_path = folder + 'can_embed_adversary_model_loss_epoch{}.png'.format(epoch)
        self.__line_graph(self.epoch_adversary_accuracy, can_adv_loss_title, can_adv_loss_path)

    def __update_epoch_vars(self, can_loss, classifier_acc, adversary_loss):
        self.epoch_can_loss = np.append(self.epoch_can_loss, can_loss)
        self.epoch_classifier_accuracy = np.append(self.epoch_classifier_accuracy, classifier_acc)
        self.epoch_adversary_accuracy = np.append(self.epoch_adversary_accuracy, adversary_loss)

    def confusion_matrix(self, data, protected, labels, batch_size=128):
        raw_class_preds, raw_protected_preds = self.model.predict(data, batch_size=batch_size)

        protected = self.__reshape_1d(protected)
        labels = self.__reshape_1d(labels)

        vals = np.zeros((8))

        # Label class predicitons
        class_preds = np.where(raw_class_preds > .5, 1, 0)
        class_only_ones = class_preds == 1
        class_only_zero = class_preds == 0

        vals[0] = sum(labels[class_only_ones] == 1)  # True Pos
        vals[1] = sum(labels[class_only_zero] == 0)  # True Neg
        vals[2] = sum(labels[class_only_ones] == 0)  # Type 1
        vals[3] = sum(labels[class_only_zero] == 1)  # Type 2

        # Protected class predictions
        protected_preds = np.where(raw_class_preds > .5, 1, 0)
        prot_only_ones = protected_preds == 1
        prot_only_zero = protected_preds == 0

        vals[4] = sum(protected[prot_only_ones] == 1)  # True Pos
        vals[5] = sum(protected[prot_only_zero] == 0)  # True Neg
        vals[6] = sum(protected[prot_only_ones] == 0)  # Type 1
        vals[7] = sum(protected[prot_only_zero] == 1)  # Type 2

        return vals / labels.shape[0]

    def model_save(self, path, epoch):
        self.model.save(path + 'bias_model_at_{}.h5'.format(epoch))
        self.adversary.save(path + 'adversary_at_{}.h5'.format(epoch))
        self.classifier.save(path + 'classifier_at_{}.h5'.format(epoch))

    @staticmethod
    def __reshape_1d(arr):
        return np.reshape(arr, (len(arr), 1))

    @staticmethod
    def result_graph_info():
        result_names = ["CAN Loss", "Classifier Accuracy", "Adversary Accuracy"]
        plot_paths = ['overall_can_loss.png', 'overall_classifier_acc.png', 'overall_adversary_acc.png']
        return result_names, plot_paths

    @staticmethod
    def __flip_label(protected_label):
        """
        Used to flip the labels so that the values can be fed into the adversary
        :param protected_label: numpy array of 1s and 0s for the incorrect label
        :return: reversed labels
        """
        return 1 - protected_label

    @staticmethod
    def __line_graph(vals, title, path):
        plt.figure(figsize=(10, 15))
        plt.plot(vals)
        plt.title(title)
        plt.savefig(path)
        plt.close()
