
import time
from typing import Dict, List, Tuple

from metisfl.common.dtypes import ControllerConfig
from metisfl.common.utils import random_id_generator
from metisfl.proto import controller_pb2, model_pb2
from metisfl.controller.aggregation import Aggregator
from metisfl.controller.core import LearnerManager
from metisfl.controller.scaling import (batches_scaling, dataset_scaling,
                                    participants_scaling)
from metisfl.controller.selection import ScheduledCardinality
from metisfl.controller.store import ModelStore


class ModelManager:

    aggregator: Aggregator = None

    model: model_pb2.Model = None
    model_store: ModelStore = None
    controller_config: ControllerConfig = None
    metadata: controller_pb2.ModelMetadata = {}
    selector: ScheduledCardinality = None
    learner_manager: LearnerManager = None
    is_initialized: bool = False

    def __init__(
        self,
        controller_config: ControllerConfig,
        learner_manager: LearnerManager,
        model_store: ModelStore,
    ) -> None:
        """Initializes the model manager.

        Parameters
        ----------
        learner_manager : LearnerManager
            The learner manager to be used.
        controller_config : ControllerConfig
            The global training configuration.
        model_store_config : ModelStoreConfig
            The model store configuration.
        """
        self.learner_manager = learner_manager
        self.controller_config = controller_config
        self.model_store = model_store

    def set_initial_model(self, model: model_pb2.Model) -> None:
        """Sets the initial model for the controller.

        Parameters
        ----------
        model : model_pb2.Model
            The initial model.
        """

        if self.is_initialized:
            raise Exception("Model already initialized")

        self.model = model
        self.is_initialized = True

    def insert_model(self, learner_id: str, model: model_pb2.Model) -> None:
        """Inserts a model into the model manager.

        Parameters
        ----------
        learner_id : str
            The learner id.
        model : model_pb2.Model
            The model to be inserted.
        """
        self.model_store.insert(
            [(learner_id, model)],
        )

    def update_model(
        self,
        to_schedule: List[str],
        learner_ids: List[str],
    ) -> None:
        """Updates the model performing an aggregation step.

        Parameters
        ----------
        to_schedule : List[str]
            The learners to be scheduled.
        learner_ids : List[str]
            The learners ids.
        """
        selected_ids = self.selector.select(to_schedule, learner_ids)
        scaling_factors = self.compute_scaling_factor(selected_ids)
        stride_length = self.get_stride_length(len(learner_ids))

        update_id = self.init_metadata()
        aggregation_start_time = time.time()
        
        to_select_block = []
        
        for learner_id in learner_ids:
            
            lineage_length = self.get_lineage_length(learner_id)
            to_select_block.append((learner_id, lineage_length))
            block_size = len(to_select_block)
            
            if block_size == stride_length or learner_id == learner_ids[-1]:
                
                selected_models = self.select_models(update_id, to_select_block)
                
                to_aggregate_block = self.get_aggregation_pairs(
                    selected_models, scaling_factors
                )
                
                self.model = self.aggregate(update_id, to_aggregate_block)
                self.model_store.re
                # TODO: C++ had a "RecordBlockSize" methon; not sure if needed
        
        self.record_aggregation_time(update_id, aggregation_start_time)
        self.aggregator.reset()        
                
    def erase_models(self, learner_ids: List[str]) -> None:
        """Erases the models of the learners.

        Parameters
        ----------
        learner_ids : List[str]
            The learners ids.
        """
        self.model_store.erase(learner_ids)

    def get_model(self) -> model_pb2.Model:
        """Gets the model.

        Returns
        -------
        model_pb2.Model
            The model.
        """
        return self.model

    def init_metadata(self) -> str:
        """Initializes the metadata."""

        update_id = str(random_id_generator())
        self.metadata[update_id] = controller_pb2.ModelMetadata()

        return

    def get_stride_length(self, num_learners: int) -> int:
        """Returns the stride length.

        Parameters
        ----------
        num_learners : int
            The number of learners.

        Returns
        -------
        int
            The stride length.
        """

        stride_length = num_learners

        if self.controller_config.aggregation_rule == "FedStride":
            fed_stride_length = self.controller_config.stride_length
            if fed_stride_length > 0:
                stride_length = fed_stride_length

        return stride_length

    def get_lineage_length(self, learner_id: str) -> int:
        """Returns the lineage length for the given learner id.

        Parameters
        ----------
        learner_id : str
            The learner id.

        Returns
        -------
        int
            The lineage length for the given learner id.
        """
        lineage_length = self.model_store.get_lineage_length(learner_id)
        required_lineage_length = self.aggregator.required_lineage_length()
        return min(lineage_length, required_lineage_length)
        

    def compute_scaling_factor(
        self,
        learner_ids: List[str],
    ) -> Dict[str, float]:
        """Computes the scaling factor for the given learners.

        Parameters
        ----------
        learner_ids : List[str]
            The learner ids.

        Returns
        -------
        Dict[str, float]
            The scaling factor for each learner.
        """

        scaling_factor = self.controller_config.scaling_factor

        if scaling_factor == "NumCompletedBatches":
            num_completed_batches = self.learner_manager.get_num_completed_batches(
                learner_ids
            )
            return batches_scaling(num_completed_batches)
        elif scaling_factor == "NumTrainingExamples":
            num_training_examples = self.learner_manager.get_num_training_examples(
                learner_ids
            )
            return dataset_scaling(num_training_examples)
        elif scaling_factor == "NumParticipants":
            return participants_scaling(learner_ids)
        else:
            raise Exception("Invalid scaling factor")

    def get_aggregation_pairs(
        self,
        selected_models: Dict[str, List[model_pb2.Model]],
        scaling_factors: Dict[str, float],
    ) -> List[List[Tuple[model_pb2.Model, float]]]:
        """Returns the aggregation pairs. """
        
        to_aggregate_block: List[List[Tuple[model_pb2.Model, float]]] = []
        tmp: List[Tuple[model_pb2.Model, float]] = []
        
        for learner_id, models in selected_models.items():
            scaling_factor = scaling_factors[learner_id]
            
            for model in models:
                tmp.append((model, scaling_factor))
            
            to_aggregate_block.append(tmp)
            tmp = []
        
        return to_aggregate_block

    def aggregate(
        self,
        update_id: str,
        to_aggregate_block: List[List[Tuple[model_pb2.Model, float]]],
    ) -> model_pb2.Model:
        """Aggregates the models and keeps track of the aggregation time.

        Parameters
        ----------
        update_id : str
            The update id.
        to_aggregate_block : List[List[Tuple[model_pb2.Model, float]]]
            The models to be aggregated.
            
        Returns
        -------
        model_pb2.Model
            The aggregated model.
        """
        start_time = time()
        model = self.aggregator.aggregate(to_aggregate_block)
        end_time = time()
        self.metadata[update_id].aggregation_block_duration_ms.append(
            end_time - start_time
        )
        return model

    def record_block_size(self, update_id: str, block_size: int) -> None:
        """Records the block size.

        Parameters
        ----------
        update_id : str
            The update id.
        block_size : int
            The block size.
        """
        pass

    def record_aggregation_time(self, update_id: str, start_time: float) -> None:
        """Records the aggregation time.

        Parameters
        ----------
        update_id : str
            The update id.
        start_time : float
            The start time.
        """
        end_time = time()
        self.metadata[update_id].aggregation_duration_ms = end_time - start_time


    def select_models(
        self,
        update_id: str,
        to_select_block: List[Tuple[str, int]],
    ) -> Dict[str, List[model_pb2.Model]]:
        """Selects the models.

        Parameters
        ----------
        update_id : str
            The update id.
        to_select_block : List[Tuple[str, int]]
            The models to be selected.

        Returns
        -------
        Dict[str, List[model_pb2.Model]]
            The selected models.
        """
        start_time = time()
        selected_models = self.model_store.select(to_select_block)
        end_time = time()
        self.metadata[update_id].selection_duration_ms.append(
            end_time - start_time
        )

        return selected_models