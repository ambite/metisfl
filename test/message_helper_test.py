import os
import numpy as np

from metisfl.learner.message import MessageHelper
from metisfl.encryption.homomorphic import HomomorphicEncryption
from metisfl.helpers.ckks import generate_keys


class TestMessageHelper:

    def setUp(self) -> None:
        super().setUp()
        batch_size = 8192
        scaling_factor_bits = 40
        cc = "/tmp/cc.txt"
        pk = "/tmp/pk.txt"
        prk = "/tmp/prk.txt"
        generate_keys(batch_size, scaling_factor_bits, cc, pk, prk)
        
        self.scheme = HomomorphicEncryption(batch_size, scaling_factor_bits, cc, pk, prk)
        self.test = np.random.rand(2, 2)

    def test_no_encryption(self):
        helper = MessageHelper()
        model = helper.weights_to_model_proto([self.test])
        weights = helper.model_proto_to_weights(model)
        assert np.allclose(self.test, weights[0])

    def test_encryption(self):
        helper = MessageHelper(self.scheme)
        model = helper.weights_to_model_proto([self.test])
        weights = helper.model_proto_to_weights(model)
        assert np.allclose(self.test, weights[0])

    def test_no_encryption_multiple(self):
        weights = np.random.rand(10,10,10,10)
        helper = MessageHelper()
        model = helper.weights_to_model_proto(weights)
        weights_out = helper.model_proto_to_weights(model)
        for w1, w2 in zip(weights, weights_out):
            assert np.allclose(w1, w2)

    def test_encryption_multiple(self):
        weights = np.random.rand(10,10,10,10)
        helper = MessageHelper(self.scheme)
        model = helper.weights_to_model_proto(weights)
        weights_out = helper.model_proto_to_weights(model)
        for w1, w2 in zip(weights, weights_out):
            assert np.allclose(w1, w2, atol=1e-3)
