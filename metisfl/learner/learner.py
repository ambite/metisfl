from metisfl.learner.dataset_handler import LearnerDataset
from metisfl.model.metis_model import MetisModel
from metisfl.proto import metis_pb2

class Learner(object):

    def __init__(self,
                 learner_dataset: LearnerDataset,
                 metis_model: MetisModel,
                 encryption_config_pb: metis_pb2.EncryptionConfig,
                 controller_server_entity_pb: metis_pb2.ServerEntity,
                 learner_server_entity_pb: metis_pb2.ServerEntity):
        self.learner_dataset = learner_dataset
        self.metis_model = metis_model
        self.encryption_config_pb = encryption_config_pb
        self.controller_server_entity_pb = controller_server_entity_pb
        self.learner_server_entity_pb = learner_server_entity_pb
