
from abc import ABC, abstractmethod
from typing import List, Tuple, Union

from metisfl.proto import model_pb2


class Aggregator(ABC):
    name: str = None

    @abstractmethod
    def aggregate(
        self, 
        pairs: Union[ 
            List[List[Tuple[model_pb2.Model, float]]],
            List[Tuple[model_pb2.Model, float]]
        ]    
        ) -> model_pb2.Model:
        """Aggregates the models.

        Parameters
        ----------
        pairs: Union[List[List[Tuple[Model, float]]], List[Tuple[Model, float]]]
            The models to aggregate. 
            
            If the input is a list of list of tuples, then the first list is the list of learners,
            and the second list is the list of (model, scaling_factor) tuples for each learner.
            If the input is a list of tuples, then the list is the list of (model, scaling_factor) tuples for each learner.
            
        Returns
        -------
        Model
            The aggregated model.
        """
        pass

    @abstractmethod
    def required_lineage_length(self) -> int:
        """Returns the required lineage length.

        Returns
        -------
        int
            The required lineage length.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Resets the aggregator."""
        pass
