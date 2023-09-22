import pandas as pd
import numpy as np
import torch
import os

from metisfl.controller import Controller

from mlp import HousingMLP

if __name__ == "__main__":
    script_cwd = os.path.dirname(__file__)
    housing_np = pd.read_csv(script_cwd + "/housing_data.csv").to_numpy()
    housing_np = housing_np[~np.isnan(housing_np).any(axis=1)]
    housing_np_x = np.array([list(row[:-1]) for row in housing_np])
    housing_np_y = np.array([list(row[-1:]) for row in housing_np])
    tensor_x = torch.Tensor(housing_np_x)  # transform to torch tensor
    tensor_y = torch.Tensor(housing_np_y)
	
    housing_mlp = HousingMLP()
    print(housing_mlp)
    housing_mlp.fit(dataset=[(tensor_x, tensor_y)], epochs=4)
    print(housing_mlp.evaluate(dataset=[(tensor_x, tensor_y)]))

    Controller()
    
