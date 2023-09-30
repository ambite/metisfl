#!/usr/bin/python3
import copy
from typing import List, Tuple

import numpy as np

from metisfl.common.dtypes import MODEL_WEIGHTS_DTYPE
from metisfl.controller.aggregation.aggregation import Aggregator
from metisfl.controller.aggregation.tensor_ops import serialize_tensor
from metisfl.proto.model_pb2 import Model


class FederatedAverage(Aggregator):
    """Federated Average Aggregation Algorithm."""

    def aggregate(self, pairs: List[List[Tuple[Model, float]]]) -> Model:
        
        model = Model()

        sample_model = pairs[0][0][0]
        
        if sample_model.encrypted:
            raise RuntimeError("Cannot aggregate encrypted tensors using Federated Average.")
        
        for tensor in sample_model.tensors:
            model.tensors.append(copy.deepcopy(tensor))

        total_tensors = len(model.tensors)

        for var_idx in range(total_tensors):
            var_num_values = model.tensors[var_idx].length

            aggregated_tensor = self.aggregate_at_index(pairs, var_idx, var_num_values)
            model.tensors[var_idx].value = serialize_tensor(aggregated_tensor)

        return model

    def aggregate_at_index(
        self,
        pairs: List[List[Tuple[Model, float]]],
        var_idx: int,
        var_num_values: int
    ) -> List[float]:
        
        aggregated_tensor = np.zeros(var_num_values)

        for pair in pairs:
            local_model = pair[0][0]
            scaling_factor = pair[0][1]
            local_tensor = local_model.tensors[var_idx]
            
            t2_r = np.frombuffer(local_tensor.value, dtype=MODEL_WEIGHTS_DTYPE) * scaling_factor
            aggregated_tensor += t2_r
                  
        return aggregated_tensor
    
    def required_lineage_length(self) -> int:
        return 1
    
    def reset(self) -> None:
        return super().reset()
    
    
