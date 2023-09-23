import os
import yaml

import metisfl.config as config

from metisfl.proto import metis_pb2, model_pb2
from metisfl.encryption.encryption_scheme import EncryptionScheme
from metisfl.proto import metis_pb2
from metisfl.proto.proto_messages_factory import MetisProtoMessages


METIS_WORKING_DIR = ".metisfl"
DRIVER_DIR_NAME = "driver"
CONTROLLER_DIR_NAME = "controller"
LEARNER_DIR_NAME = "learner_{}"
MODEL_SAVE_DIR_NAME = "model_definition"
SSL_PATH = "resources/ssl"
GEN_CERTS_SCRIPT_NAME = "generate.sh"
GEN_CERTS_SCRIPT_NAME_DIR = os.path.join(SSL_PATH, GEN_CERTS_SCRIPT_NAME)
SERVER_CERT_NAME = "server-cert.pem"
SERVER_KEY_NAME = "server-key.pem"
SERVER_CERT_DIR = os.path.join(SSL_PATH, SERVER_CERT_NAME)
SERVER_KEY_DIR = os.path.join(SSL_PATH, SERVER_KEY_NAME)

CRYPTO_RESOURCE_DIR = "resources/crypto"
CRYPTO_CONTEXT_FILE = "cryptocontext.txt"
CRYPTO_KEY_PUBLIC_FILE = "public.txt"
CRYPTO_KEY_PRIVATE_FILE = "private.txt"

DEFAULT_CONTROLLER_HOSTNAME = "[::]"
DEFAULT_CONTROLLER_PORT = 50051
DEFAULT_LEARNER_HOST = "[::]"
DEFAULT_LEARNER_PORT = 50052

TRAIN = "train"
VALIDATION = "validation"
TEST = "test"
TASK_KEYS = [TRAIN, VALIDATION, TEST]

KERAS_NN_ENGINE = "keras_nn"
PYTORCH_NN_ENGINE = "pytorch_nn"
NN_ENGINES = [KERAS_NN_ENGINE, PYTORCH_NN_ENGINE]

DATASET_RECIPE_FILENAMES = {
    "train": "model_train_dataset_ops.pkl",
    "validation": "model_validation_dataset_ops.pkl",
    "test": "model_test_dataset_ops.pkl"
}

LEARNING_TASK = "learning"
EVALUATION_TASK = "evaluation"
INFERENCE_TASK = "inference"
TASKS = [LEARNING_TASK, EVALUATION_TASK, INFERENCE_TASK]

LEARNER_ID_FILE = "learner_id.txt"
AUTH_TOKEN_FILE = "auth_token.txt"

CANCEL_RUNNING_ON_SHUTDOWN = {
    LEARNING_TASK: True,
    EVALUATION_TASK: False,
    INFERENCE_TASK: False
}


class FederationEnvironment(object):

    def __init__(self, fed_env_config_fp=None, fed_env_config_dict=None):
        if fed_env_config_fp:
            fstream = open(fed_env_config_fp).read()
            self._fed_env = yaml.load(fstream, Loader=yaml.SafeLoader)
        elif fed_env_config_dict:
            self._fed_env = fed_env_config_dict
        self.controller = RemoteHost(self._fed_env.get("Controller"))
        self.learners = [RemoteHost(learner) for learner in self._fed_env.get("Learners")]
        self._encryption_config_pb = self._setup_encryption_config()

    # Communication protocol - REQUIRED.
    def get_communication_protocol(self):
        return self._fed_env.get("CommunicationProtocol")

    @property
    def communication_protocol(self):
        return self.get_communication_protocol().get("Name")

    @property
    def semi_sync_lambda(self):
        return self.get_communication_protocol().get("SemiSyncLambda")

    @property
    def semi_sync_recompute(self):
        return self.get_communication_protocol().get("SemiSyncRecompute")

    # Termination signals - REQUIRED.
    def get_termination_signals(self):
        return self._fed_env.get("TerminationSignals")

    @property
    def federation_rounds(self):
        return self.get_termination_signals().get("FederationRounds")

    @property
    def execution_time_cutoff_mins(self):
        return self.get_termination_signals().get("ExecutionCutoffTimeMins")

    @property
    def evaluation_metric(self):
        return self.get_termination_signals().get("EvaluationMetric")

    @property
    def metric_cutoff_score(self):
        return self.get_termination_signals().get("EvaluationMetricCutoffScore")

    # Model store configurations - OPTIONAL.
    def get_model_store_config(self):
        return self._fed_env.get("ModelStoreConfig")

    @property
    def model_store(self):
        return self.get_model_store_config().get("ModelStore")

    @property
    def model_store_hostname(self):
        return self.get_model_store_config().get("ModelStoreHostname")

    @property
    def model_store_port(self):
        return self.get_model_store_config().get("ModelStorePort")

    @property
    def eviction_policy(self):
        return self.get_model_store_config().get("EvictionPolicy")

    @property
    def lineage_length(self):
        return self.get_model_store_config().get("LineageLength")

    # Encryption scheme configurations - OPTIONAL. 
    def get_encryption_scheme(self):
        return self._fed_env.get("EncryptionScheme")

    @property
    def encryption_scheme_name(self):
        return self.get_encryption_scheme().get("Name")

    @property
    def he_batch_size(self):
        return self.get_encryption_scheme().get("BatchSize")

    @property
    def he_scaling_factor_bits(self):
        return self.get_encryption_scheme().get("ScalingFactorBits")

    def _setup_encryption_config(self) -> metis_pb2.EncryptionConfig:        
        if self.get_encryption_scheme() and \
                self.encryption_scheme_name.upper() == "CKKS":
            ckks_scheme_pb = \
                MetisProtoMessages.construct_ckks_scheme_pb(
                    self.he_batch_size, self.he_scaling_factor_bits)
            fcc, fpb, fsk = config.get_crypto_resources()
            he_scheme_config_pb = \
                MetisProtoMessages.construct_he_scheme_config_pb(
                as_files=True,
                crypto_context=fcc,
                public_key=fpb,
                private_key=fsk)
            # FIXME(@stripeli): If homomorphic encryption files are 
            #  passed over the wire then fabric raises EOFError. This
            #  is due to the size of the transmitted text, which is 
            #  greater than the allowed stream.
            # he_scheme_config_pb = \
            #     MetisProtoMessages.construct_he_scheme_config_pb(
            #     as_files=False)
            he_scheme_pb = \
                MetisProtoMessages.construct_he_scheme_pb(
                    he_scheme_config_pb=he_scheme_config_pb,
                    scheme_pb=ckks_scheme_pb)
            encryption_config_pb = \
                MetisProtoMessages.construct_encryption_config_pb(
                    he_scheme_pb=he_scheme_pb)
            # Initialize crypto parameters of encryption scheme.
            encryption = EncryptionScheme(init_crypto_params=True) \
                    .from_proto(encryption_config_pb)
            encryption_scheme_pb = encryption.to_proto()
            return encryption_scheme_pb
        else:
            return metis_pb2.EncryptionConfig()

    def get_encryption_config_pb(self, for_aggregation=False) -> metis_pb2.EncryptionConfig:
        encryption_config_pb = metis_pb2.EncryptionConfig()        
        if self._encryption_config_pb.HasField("he_scheme"):
            # We perform a deep copy of the encryption configuration 
            # because we might need to edit the private and we do not 
            # want to destroy the original keys.
            encryption_config_pb.he_scheme.CopyFrom(self._encryption_config_pb.he_scheme)
            if for_aggregation:
                encryption_config_pb.he_scheme.he_scheme_config.private_key = ""
        return encryption_config_pb

    # Global model configurations - REQUIRED.
    def get_global_model_config(self):
        return self._fed_env.get("GlobalModelConfig")

    @property
    def aggregation_rule(self):
        return self.get_global_model_config().get("AggregationRule")

    @property
    def participation_ratio(self):
        return self.get_global_model_config().get("PariticipationRatio")

    @property
    def scaling_factor(self):
        return self.get_global_model_config().get("ScalingFactor")

    @property
    def stride_length(self):
        return self.get_global_model_config().get("StrideLength")

    # Local model configurations - REQUIRED.
    def get_local_model_config(self):
        return self._fed_env.get("LocalModelConfig")

    @property
    def train_batch_size(self):
        return self.get_local_model_config().get("BatchSize")
    
    @property
    def local_epochs(self):
        return self.get_local_model_config().get("LocalEpochs")        

    @property
    def validation_percentage(self):
        return self.get_local_model_config().get("ValidationPercentage")

    def get_local_model_config_pb(self) -> metis_pb2.ControllerParams.ModelHyperparams:
        return MetisProtoMessages.construct_controller_modelhyperparams_pb(
            batch_size=self.train_batch_size,
            epochs=self.local_epochs,
            optimizer_pb=self._get_optimizer_pb(),
            percent_validation=self.validation_percentage)

    def _get_optimizer_pb(self) -> model_pb2.OptimizerConfig:
        optimizer_name = self.get_local_model_config().get("Optimizer")
        params = self.get_local_model_config().get("OptimizerParams", {})
        params = { param: str(val) for param, val in params.items() }
        return model_pb2.OptimizerConfig(name=optimizer_name, params=params)

    def get_global_model_config_pb(self) -> metis_pb2.GlobalModelSpecs:
        aggregation_rule_pb = MetisProtoMessages.construct_aggregation_rule_pb(
            rule_name=self.aggregation_rule,
            scaling_factor=self.scaling_factor,
            stride_length=self.stride_length,            
            encryption_config_pb=self.get_encryption_config_pb(for_aggregation=True))
        return MetisProtoMessages.construct_global_model_specs(
            aggregation_rule_pb=aggregation_rule_pb,
            learners_participation_ratio=self.participation_ratio)

    def get_communication_protocol_pb(self) -> metis_pb2.CommunicationSpecs:    
        protocol = self.communication_protocol        
        # If we have the Semi-Synchronous protocol, then we need to know the
        # SemiSync λ-value and whether the number of steps performed at every
        # round for every learner should be recomputed at every synchronization 
        # step or remain remain constant throughout execution based on the steps 
        # that were configured during the cold start round.
        # See also the algorithm here: https://dl.acm.org/doi/full/10.1145/3524885
        semi_sync_lambda = self.semi_sync_lambda
        semi_sync_recompute = self.semi_sync_recompute
        return MetisProtoMessages.construct_communication_specs_pb(
            protocol=protocol,
            semi_sync_lambda=semi_sync_lambda,
            semi_sync_recompute_num_updates=semi_sync_recompute)

    def get_model_store_config_pb(self) -> metis_pb2.ModelStoreConfig:        
        model_store_config_pb = \
            MetisProtoMessages.construct_model_store_config_pb(
                name=self.model_store,
                eviction_policy=self.eviction_policy,
                lineage_length=self.lineage_length,
                store_hostname=self.model_store_hostname,
                store_port=self.model_store_port)
        return model_store_config_pb

class RemoteHost(object):
    def __init__(self, config_map):
        self._config_map = config_map

    @property
    def identifier(self):
        return "{}:{}".format(self.grpc_hostname, self.grpc_port)

    @property
    def project_home(self):
        return self._config_map.get("ProjectHome")

    # Connection Configurations - REQUIRED.
    def get_connection_config(self):
        return self._config_map.get("ConnectionConfig")

    @property
    def hostname(self):
        return self.get_connection_config().get("Hostname")

    @property
    def cuda_devices(self):
        return self._config_map.get("CudaDevices")

    @property
    def port(self):
        return self.get_connection_config().get("Port")

    @property
    def username(self):
        return self.get_connection_config().get("Username")

    @property
    def password(self):
        return self.get_connection_config().get("Password")

    @property
    def key_filename(self):
        return self.get_connection_config().get("KeyFilename")

    @property
    def passphrase(self):
        return self.get_connection_config().get("Passphrase")

    @property
    def on_login_command(self):
        return self._config_map.get("OnLoginCommand", "")

    # GRPC Configurations - REQUIRED.
    def get_grpc_server(self):
        return self._config_map.get("GRPCServer")

    @property
    def grpc_hostname(self):
        return self.get_grpc_server().get("Hostname")

    @property
    def grpc_port(self):
        return self.get_grpc_server().get("Port")

    # SSL Configurations.
    def get_ssl_config(self):
        return self._config_map.get("SSLConfig")

    @property
    def ssl_private_key(self):        
        return self.get_ssl_config().get("PrivateKey") \
            if self.get_ssl_config() else None

    @property
    def ssl_public_certificate(self):
        return self.get_ssl_config().get("PublicCertificate") \
            if self.get_ssl_config() else None

    def get_fabric_connection_config(self):
        # Look for parameters values here:
        # https://docs.paramiko.org/en/latest/api/client.html#paramiko.client.SSHClient.connect
        # 'allow_agent' show be disabled if working with username/password.
        connect_kwargs = {
            "password": self.password,
            "allow_agent": False if self.password else True,
            "look_for_keys": True if self.key_filename else False,
        }
        if self.key_filename:
            connect_kwargs["key_filename"] = self.key_filename
            connect_kwargs["passphrase"] = self.passphrase

        conn_config = {
            "host": self.hostname,
            "port": self.port,
            "user": self.username,
            "connect_kwargs": connect_kwargs
        }
        return conn_config

    def get_server_entity_pb(self, gen_connection_entity=False) -> metis_pb2.ServerEntity:
        """Generate a ServerEntity proto object for the given parameters.

        Args:
            gen_connection_entity (bool, optional): Sets the private key to None. Defaults to False.

        Returns:
            metis_pb2.ServerEntity: The generated ServerEntity proto object.
        """        
        certificate = self.ssl_public_certificate
        private_key = None if gen_connection_entity else self.ssl_private_key
        ssl_enable = True if certificate else False
        ssl_config_files_pb = MetisProtoMessages.construct_ssl_config_files_pb(
            public_certificate_file=certificate,
            private_key_file=private_key)
        ssl_config_pb = MetisProtoMessages.construct_ssl_config_pb(
            ssl_enable, ssl_config_files_pb)
        server_entity_pb = metis_pb2.ServerEntity(
            hostname=self.grpc_hostname,
            port=self.grpc_port,
            ssl_config=ssl_config_pb)
        return server_entity_pb
