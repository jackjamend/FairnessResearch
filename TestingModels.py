"""
File used to test models being used.

To test a model and use this script, model needs to implement:
    __init__(input_size)
    display_model()
    train(data, protected, labels, batch_size)
    test(data, protected, labels, batch_size)
    create_figs(epoch, folder, fold)
    result_graph_info() * static

model_types:
    Basic model - 0
    CAN - 1


Jack Amend
3/2/2020
"""
import numpy as np

import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import KFold

from DataLoader import get_data_labels
from BasicModel import BasicModel
from CAN import CAN
from pathlib import Path


def fold_bars(values, title, path):
    # Fold Test Classifier Accuracy
    plt.figure(figsize=(10, 15))
    plt.title(title)
    plt.bar([str(x) for x in range(len(values))], values)
    plt.savefig(path)
    plt.close()


# Parameters for the script
verbose = True
batch_size = 128
num_folds = 5
epochs = 25
model_type = 1

if model_type == 0:
    fig_folder = 'model_testing/basic/'
elif model_type == 1:
    fig_folder = 'model_testing/can/'
else:
    fig_folder = ''
    print("Saving graphs to home directory")

fig_folder = fig_folder+'epochs_{}/'.format(epochs)
Path(fig_folder).mkdir(parents=True, exist_ok=True)

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

    # Get training data
    train_data = data[train_idx]
    train_prtd = protected[train_idx]
    train_label = labels[train_idx]

    for epoch in range(epochs):
        print("Epoch: {}".format(epoch))

        # Train model
        model.train(train_data, train_prtd, train_label, batch_size)

    # Testing
    test_data = data[test_idx]
    test_attr = protected[test_idx]
    test_label = labels[test_idx]

    results = model.test(test_data, test_attr, test_label)
    test_results = np.concatenate([test_results, results])

    model.create_figs(epochs, fig_folder, i)

result_names, fig_files = model.result_graph_info()
incr = len(test_results) // epochs
for i in range(incr):
    graph_vals = test_results[i::incr]
    fold_bars(graph_vals, result_names[i], fig_folder + fig_files[i])


