import gc
import torch

from typing import Any, Dict, Tuple

from metisfl.model.model_dataset import ModelDataset
from metisfl.model.model_ops import ModelOps
from metisfl.model.utils import get_num_of_epochs
from metisfl.model.torch.helper import construct_dataset_pipeline
from metisfl.model.torch.torch_model import MetisModelTorch
from metisfl.model.types import LearningTaskStats, ModelWeightsDescriptor
from metisfl.proto import metis_pb2
from metisfl.common.formatting import DataTypeFormatter
from metisfl.common.logger import MetisLogger
from metisfl.proto.proto_messages_factory import MetisProtoMessages

class TorchModelOps(ModelOps):
    
    def __init__(self, model_dir: str):
        self._metis_model = MetisModelTorch.load(model_dir)

    def train(self,
                    train_dataset: ModelDataset,
                    learning_task_pb: metis_pb2.LearningTask,
                    hyperparameters_pb: metis_pb2.Hyperparameters) \
        -> Tuple[ModelWeightsDescriptor, LearningTaskStats]:
        
        if not train_dataset:
            MetisLogger.fatal("Provided `dataset` for training is None.")
        MetisLogger.info("Starting model training.")

        total_steps = learning_task_pb.num_local_updates
        batch_size = hyperparameters_pb.batch_size
        dataset_size = train_dataset.get_size()
        epochs_num = get_num_of_epochs(total_steps, dataset_size, batch_size)
        dataset = construct_dataset_pipeline(train_dataset) # TODO: this is inconsistent with tf counterpart
        
        self._metis_model._backend_model.train() # set model to training mode
        train_res = self._metis_model.fit(dataset, epochs=epochs_num)
        
        # TODO(@stripeli): Need to add the metrics for computing the execution time
        model_weights_descriptor = self.get_model_weights()
        learning_task_stats = LearningTaskStats(
            train_stats=train_res,
            completed_epochs=epochs_num,
            global_iteration=learning_task_pb.global_iteration)
        MetisLogger.info("Model training is complete.")
        return model_weights_descriptor, learning_task_stats

    def evaluate(self, eval_dataset: ModelDataset) -> Dict:
        if not eval_dataset:
            MetisLogger.fatal("Provided `dataset` for evaluation is None.")
        MetisLogger.info("Starting model evaluation.")
        dataset = construct_dataset_pipeline(eval_dataset)
        self._metis_model._backend_model.eval() # set model to evaluation mode
        eval_res = self._metis_model.evaluate(dataset)            
        MetisLogger.info("Model evaluation is complete.")
        metric_values = DataTypeFormatter.stringify_dict(eval_res, stringify_nan=True)
        return MetisProtoMessages.construct_model_evaluation_pb(metric_values)
    
    def infer(self) -> Any:
        # Set model to evaluation state.
        self._metis_model._backend_model.eval()

    def cleanup(self):
        del self._metis_model
        torch.cuda.empty_cache()
        gc.collect()
