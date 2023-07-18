
from dataclasses import dataclass
from metisfl.proto import metis_pb2

@dataclass
class EnvConfig(object):
    federation_rounds: int
    execution_time_cutoff_mins: int
        
@dataclass
class ServerEntity(object):
    hostname: str
    port: int
    enable_ssl: bool
    public_certificate_file: str
    private_key_file: str
        
    def to_proto(self):
        return metis_pb2.ServerEntity(**self.__dict__)
    
@dataclass
class GlobalModelSpecs(object):
    aggregation_rule: str
    scaling_factor: int
    stride_length: int
    he_batch_size: int
    he_scaling_factor_bits: int
    he_crypto_context_file: str
       
    def to_proto(self):
        return metis_pb2.GlobalModelSpecs(**self.__dict__)
    
@dataclass
class CommunicationSpecs(object):
    protocol: str
    semi_sync_lambda: int
    semi_sync_recompute_num_updates: bool
    
    def to_proto(self):
        return metis_pb2.CommunicationSpecs(**self.__dict__)
    
@dataclass
class ModelStoreConfig(object):
    model_store: str
    eviction_policy: str
    model_store_hostname: str
    model_store_port: int
    
    def to_proto(self):
        return metis_pb2.ModelStoreConfig(**self.__dict__)
    
@dataclass
class ModelHyperparams(object):
    batch_size: int
    epochs: int
    
    def to_proto(self):
        return metis_pb2.ModelHyperparams(**self.__dict__)
