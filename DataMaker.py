from DataGAN import DataGAN
import numpy as np
from DataLoader import DataLoader

epochs = 5

data = DataLoader("adult.data")
numeric_data, labels, protected = data.get_numeric_data()
data_gan = DataGAN(numeric_data.shape[1])

for epoch in range(epochs):
    print('Epoch {}/{}'.format(epoch+1, epochs))
    data_gan.train(numeric_data, labels, batch_size=64, n_classes=2)
