import unittest
import numpy as np
from metisfl.controller.aggregation.federated_average import FederatedAverage
from metisfl.learner.message import MessageHelper


class TestFederatedAverage(unittest.TestCase):
    
    def test_fedavg_1(self):
        weights1 = np.random.rand(2,2)
        weights2 = np.random.rand(2,2)
        model1 = MessageHelper().weights_to_model_proto([weights1])
        model2 = MessageHelper().weights_to_model_proto([weights2])

        pairs = [
            [(model1, 1)],
            [(model2, 1)]
        ]
    
        weights = FederatedAverage().Aggregate(pairs)
    
        assert np.allclose(weights[0], (weights1 + weights2) / 2)
    