"""
Interface for models to be used in TestingModels.py

Jack Amend

3/3/2020
"""


class Model:
    def __init__(self, input_size):
        pass

    def display_model(self):
        pass

    def train(self, data, protected, labels, batch_size):
        pass

    def test(self, data, protected, labels, batch_size):
        pass

    def create_figs(self, epoch, folder, fold):
        pass

    @staticmethod
    def result_graph_info():
        pass
