'''
Used to run individualize experiments
'''

import numpy as np
from sklearn.model_selection import KFold
from BasicModel import BasicModel
from CAN import CAN
from pathlib import Path
from datetime import datetime
from DataLoader import DataLoader
import matplotlib.pyplot as plt
import csv

def run_experiment(model_type, data_type, epochs=100, num_folds=5, batch_size=128, testing_inv=10):
    '''
    Runs experiment for given different configurations.
    :param model_type: type of model. 0 for basic. 1 for CAN
    :param data_type: type of data. 0 for orginial. 1 for balanced data.
    :return:
    '''

    def get_data(file_path, data_type):
        data_loader = DataLoader(file_path)
        if data_type == 0:
            data, labels, protected = data_loader.get_numeric_data()
        else:
            data, labels, protected = data_loader.get_numeric_data(True)

        return data, labels, protected

    def fold_bars(values, title, path):
        # Fold Test Classifier Accuracy
        plt.figure(figsize=(10, 15))
        plt.title(title)
        plt.bar([str(x) for x in range(len(values))], values)
        plt.savefig(path)
        plt.close()

    def write_to_file(file_name, information):
        with open(file_name, 'a') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(information)

    def record_test_stats(model, data, protected, labels, test_idx, batch_size, epoch,
                          fold_fig_folder, start, start_string, i, fold_model_folder,
                          data_type, fold_time_str):
        # Testing data
        test_data = data[test_idx]
        test_attr = protected[test_idx]
        test_label = labels[test_idx]

        # Model test statistics
        test_stats = model.test(test_data, test_attr, test_label, batch_size)

        # Confusion Matrix from model
        confusion_mat = model.confusion_matrix(test_data, test_attr, test_label, batch_size)

        # Recording Information
        model.create_figs(epoch, fold_fig_folder)

        # Variables to write to CSV
        curr_time = datetime.now()
        curr_time_string = curr_time.strftime("%d/%m/%Y %H:%M:%S")
        diff_time = curr_time - start

        record_vars = [start_string, curr_time_string, diff_time, i, epoch, *test_stats,
                       *confusion_mat, batch_size, fold_fig_folder, fold_model_folder, data_type]

        write_to_file(fold_time_str + 'overview.csv', record_vars)

    # Sets information for saving data
    fig_folder = 'model_testing/'
    if model_type == 0:
        fig_folder += 'basic/'
    elif model_type == 1:
        fig_folder += 'can/'

    # Record datetime
    start = datetime.now()
    start_string = start.strftime("%d/%m/%Y %H:%M:%S")
    fold_time_str = fig_folder + 'epochs{}_'.format(epochs) + start.strftime("%d-%m-%Y_%H-%M-%S") + '/'

    csv_file_str = fold_time_str
    model_folder = fold_time_str + 'models/'
    fig_folder = fold_time_str + 'graphs/'

    data, labels, protected = get_data('adult.data', data_type)

    # Training information
    input_size = data.shape[1]
    kf = KFold(n_splits=num_folds)

    test_results = np.array([])

    for i, (train_idx, test_idx) in enumerate(kf.split(data)):
        if model_type == 0:
            model = BasicModel(input_size)
        elif model_type == 1:
            model = CAN(input_size)

        model.display_models()

        # Create folders for fold
        fold_fig_folder = fig_folder + 'fold{}/'.format(i)
        fold_model_folder = model_folder + 'fold{}/'.format(i)
        Path(fold_fig_folder).mkdir(parents=True, exist_ok=True)
        Path(fold_model_folder).mkdir(parents=True, exist_ok=True)
        # Get training data
        train_data = data[train_idx]
        train_prtd = protected[train_idx]
        train_label = labels[train_idx]
        for epoch in range(epochs):

            print("Epoch: {}".format(epoch))

            # Train model
            model.train(train_data, train_prtd, train_label, batch_size)

            # Testing at defined interval
            if epoch % testing_inv == 0 or epoch == epochs - 1:
                record_test_stats(model, data, protected, labels, test_idx, batch_size, epoch,
                                  fold_fig_folder, start, start_string, i, fold_model_folder,
                                  data_type, fold_time_str)
        result_names, fig_files = model.result_graph_info()
        incr = len(test_results) // epochs
        for i in range(incr):
            graph_vals = test_results[i::incr]
            fold_bars(graph_vals, result_names[i], fig_folder + fig_files[i])