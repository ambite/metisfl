import pandas as pd
import numpy as np

import torch

from torch.nn import MSELoss
from torch.optim import SGD
from typing import Dict
from numpy import vstack
from sklearn.metrics import mean_squared_error


class HousingMLP(torch.nn.Module):
    def __init__(self, params_per_layer=10, hidden_layers_num=1, learning_rate=0.0, data_type="float64"):
        super(HousingMLP, self).__init__()
        self.params_per_layer = params_per_layer
        self.hidden_layers_num = hidden_layers_num

        if data_type == "float32":
            self.data_type = torch.float32
        elif data_type == "float64":
            self.data_type = torch.float64
        else:
            raise RuntimeError("Not a supported data type. Please pass float32 or float64")

        layers = list()
        layers.append(torch.nn.Linear(13, self.params_per_layer, dtype=self.data_type))
        layers.append(torch.nn.ReLU())
        for i in range(self.hidden_layers_num):
            layers.append(torch.nn.Linear(self.params_per_layer, self.params_per_layer, dtype=self.data_type))
            layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(self.params_per_layer, 1, dtype=self.data_type))
        self.model = torch.nn.Sequential(*layers)

    def forward(self, x):
        output = self.model(x)
        return output

    def fit(self, dataset, epochs, *args, **kwargs) -> Dict:
        # define the optimization
        criterion = MSELoss()
        optimizer = SGD(self.parameters(), lr=0.01, momentum=0.9)
        # enumerate epochs
        for epoch in range(epochs):
            print("MLP Epoch: ", epoch, flush=True)
            # enumerate mini batches
            for i, (inputs, targets) in enumerate(dataset):
                # clear the gradients
                optimizer.zero_grad()
                # compute the model output
                yhat = self.forward(inputs)
                # calculate loss
                loss = criterion(yhat, targets)
                # credit assignment
                loss.backward()
                # update model weights
                optimizer.step()
        return {}

    def evaluate(self, dataset, *args, **kwargs) -> Dict:
        predictions, actuals = list(), list()
        for i, (inputs, targets) in enumerate(dataset):
            # evaluate the model on the test set
            yhat = self.forward(inputs)
            # retrieve numpy array
            yhat = yhat.detach().numpy()
            actual = targets.numpy()
            actual = actual.reshape((len(actual), 1))
            # round to class values
            yhat = yhat.round()
            # store
            predictions.append(yhat)
            actuals.append(actual)
        predictions, actuals = vstack(predictions), vstack(actuals)
        # calculate MSE
        mse = mean_squared_error(actuals, predictions)
        return {"mse": mse}
