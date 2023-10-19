import numpy as np

from metisfl.encryption.homomorphic import HomomorphicEncryption
from metisfl.helpers.ckks import generate_keys
from metisfl.learner.message import MessageHelper

batch_size = 8192
scaling_factor_bits = 40
cc = "/tmp/cc.txt"
pk = "/tmp/pk.txt"
prk = "/tmp/prk.txt"
generate_keys(batch_size, scaling_factor_bits, cc, pk, prk)
scheme = HomomorphicEncryption(batch_size, scaling_factor_bits, cc, pk, prk)

def test_no_encryption_1():
    weights = np.random.rand(2,2)
    helper = MessageHelper()
    model = helper.weights_to_model_proto(weights)
    weights_out = helper.model_proto_to_weights(model)
    for w1, w2 in zip(weights, weights_out):
        assert np.allclose(w1, w2)
        
def test_no_encryption_2():
    weights = np.random.rand(10,10,10,10)
    helper = MessageHelper()
    model = helper.weights_to_model_proto(weights)
    weights_out = helper.model_proto_to_weights(model)
    for w1, w2 in zip(weights, weights_out):
        assert np.allclose(w1, w2)
