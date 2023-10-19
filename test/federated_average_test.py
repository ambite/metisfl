import numpy as np

from metisfl.controller.aggregation.federated_average import FederatedAverage
from metisfl.learner.message import MessageHelper

    
def test_fedavg_1():
    weights1 = np.ones((2,2))
    weights2 = np.ones((2,2))
    model1 = MessageHelper().weights_to_model_proto([weights1])
    model2 = MessageHelper().weights_to_model_proto([weights2])

    pairs = [
        (model1, 1),
        (model2, 1)
    ]

    model = FederatedAverage().aggregate(pairs)
    weights = MessageHelper().model_proto_to_weights(model)

    for i in range(len(weights)):
        assert np.allclose(weights[i], weights1 + weights2)

def test_fedavg_2():
    weights1 = np.random.rand(20,20,20)
    weights2 = np.random.rand(20,20,20)
    
    model1 = MessageHelper().weights_to_model_proto([weights1])
    model2 = MessageHelper().weights_to_model_proto([weights2])
    
    
    pairs = [
        (model1, 1),
        (model2, 1)
    ]
    
    model = FederatedAverage().aggregate(pairs)
    weights = MessageHelper().model_proto_to_weights(model)
    
    for i in range(len(weights)):
        assert np.allclose(weights[i], (weights1 + weights2))
    
def test_fedavg_3():
    weights1 = np.random.rand(20,20,20)
    weights2 = np.random.rand(20,20,20)
    
    model1 = MessageHelper().weights_to_model_proto([weights1])
    model2 = MessageHelper().weights_to_model_proto([weights2])
    sf1 = 10
    sf2 = 1
    
    pairs = [
        (model1, sf1),
        (model2, sf2)
    ]
    
    model = FederatedAverage().aggregate(pairs)
    weights = MessageHelper().model_proto_to_weights(model)
    
    for i in range(len(weights)):
        assert np.allclose(weights[i], (weights1*sf1 + weights2*sf2))