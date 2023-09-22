import abc

from typing import Any, Dict, Tuple

from metisfl.model.model_dataset import ModelDataset
from metisfl.model.metis_model import MetisModel
from metisfl.model.types import LearningTaskStats, ModelWeightsDescriptor
from metisfl.proto import metis_pb2


class ModelOps(object):

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.cleanup()
    
    def get_model(self) -> MetisModel:
        assert self._metis_model is not None, "Model is not initialized."
        return self._metis_model

    @abc.abstractmethod
    def train(self,
              train_dataset: ModelDataset,
              learning_task_pb: metis_pb2.LearningTask,
              hyperparameters_pb: metis_pb2.Hyperparameters,
              validation_dataset: ModelDataset = None,
              test_dataset: ModelDataset = None,
              verbose=False) -> Tuple[ModelWeightsDescriptor, LearningTaskStats]:
        pass
    
    @abc.abstractmethod
    def evaluate(self,
                 eval_dataset: ModelDataset,
                 batch_size=100,
                 metrics=None,
                 verbose=False) -> Dict:
        pass
    
    @abc.abstractmethod
    def infer(self,
              infer_dataset: ModelDataset,
              batch_size=100) -> Any:
        pass

    @abc.abstractmethod
    def cleanup(self):
        pass
