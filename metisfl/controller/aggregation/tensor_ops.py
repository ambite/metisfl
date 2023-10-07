"""Helper functions for serializing and deserializing tensors."""

import struct
from typing import List

from google.protobuf.text_format import Parse

from metisfl.proto.model_pb2 import Tensor


def deserialize_tensor(tensor: Tensor) -> List[float]:
    tensor_bytes = tensor.value
    tensor_elements_num = tensor.length
    deserialized_tensor = struct.unpack(f'{tensor_elements_num}d', tensor_bytes)
    return list(deserialized_tensor)

def serialize_tensor(v: List[float]) -> bytes:
    serialized_tensor = struct.pack(f'{len(v)}d', *v)
    return serialized_tensor

def print_serialized_tensor(data: bytes, num_values: int) -> None:
    loaded_values = struct.unpack(f'{num_values}d', data)
    print(', '.join(map(str, loaded_values)))

def parse_text_or_die(input_str: str, message_type):
    result = message_type()
    Parse(input_str, result)
    return result
