"""
File used to test models being used.

To test a model and use this script, model needs to implement:
    __init__(input_size)
    display_model()
    train(data, protected, labels, batch_size)
    test(data, protected, labels, batch_size)
    confusion_matrix(data, protected, labels, batch_size)
    create_figs(epoch, folder)
    result_graph_info() * static

model_types:
    Basic model - 0
    CAN - 1


Jack Amend
3/2/2020
"""
import csv

import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import KFold

from DataLoader import get_data_labels
from BasicModel import BasicModel
from CAN import CAN
from pathlib import Path
from datetime import datetime


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


# Parameters for the script
model_type = 0
verbose = True
batch_size = 128
num_folds = 5
testing_inv = 10
epochs = 201
fig_folder = 'model_testing/'

if model_type == 0:
    fig_folder += 'basic/'
elif model_type == 1:
    fig_folder += 'can/'
else:
    print("Saving graphs to home directory")


# Record datetime
start = datetime.now()
start_string = start.strftime("%d/%m/%Y %H:%M:%S")
fold_time_str = fig_folder + 'epochs{}_'.format(epochs) + start.strftime("%d-%m-%Y_%H-%M-%S") +'/'

csv_file_str = fold_time_str
model_folder = fold_time_str + 'models/'
fig_folder = fold_time_str + 'graphs/'

# Loading data
if verbose:
    print("Loading data.")
data, labels, protected = get_data_labels("adult.data")

# Testing basic model
if verbose:
    print("Creating model.")
input_size = data.shape[1]
kf = KFold(n_splits=num_folds)

test_results = np.array([])

for i, (train_idx, test_idx) in enumerate(kf.split(data)):
    if model_type == 0:
        model = BasicModel(input_size)
    elif model_type == 1:
        model = CAN(input_size)

    if verbose:
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
        if epoch % testing_inv == 0:
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
                           *confusion_mat, batch_size, fold_fig_folder, fold_model_folder]

            write_to_file(fold_time_str + 'overview.csv', record_vars)

    # Testing
    test_data = data[test_idx]
    test_attr = protected[test_idx]
    test_label = labels[test_idx]

    results = model.test(test_data, test_attr, test_label)
    test_results = np.concatenate([test_results, results])

    # Confusion Matrix from model
    confusion_mat = model.confusion_matrix(test_data, test_attr, test_label, batch_size)

    # Recording Information
    model.create_figs(epochs, fold_fig_folder)

    # Variables to write to CSV
    curr_time = datetime.now()
    curr_time_string = curr_time.strftime("%d/%m/%Y %H:%M:%S")
    diff_time = curr_time - start

    record_vars = [start_string, curr_time_string, diff_time, i, epochs, results,
                   *confusion_mat, batch_size, fold_fig_folder, fold_model_folder]

    write_to_file(fold_time_str + 'overview.csv', record_vars)


result_names, fig_files = model.result_graph_info()
incr = len(test_results) // epochs
for i in range(incr):
    graph_vals = test_results[i::incr]
    fold_bars(graph_vals, result_names[i], fig_folder + fig_files[i])


