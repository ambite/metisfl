import argparse

import metisfl.learner.utils as learner_utils

from metisfl.learner.dataset_handler import LearnerDataset
from metisfl.learner.learner_executor import LearnerExecutor
from metisfl.learner.learner_server import LearnerServer
from metisfl.learner.learner_task import LearnerTask
from metisfl.model.utils import get_model_ops_fn


def init_learner(args):
    learner_server_entity_pb, controller_server_entity_pb = \
        learner_utils.create_servers(args)
    encryption_config_pb = learner_utils.parse_encryption_config_hex(
        args.encryption_config_protobuff_serialized_hexadecimal)
    model_ops_fn = get_model_ops_fn(args.neural_engine)

    learner_dataset = LearnerDataset(
        train_dataset_fp=args.train_dataset,
        validation_dataset_fp=args.validation_dataset,
        test_dataset_fp=args.test_dataset,
        train_dataset_recipe_pkl=args.train_dataset_recipe,
        validation_dataset_recipe_pkl=args.validation_dataset_recipe,
        test_dataset_recipe_pkl=args.test_dataset_recipe)
    learner_task = LearnerTask(
        learner_server_entity_pb=learner_server_entity_pb,
        learner_dataset=learner_dataset,
        model_dir=args.model_dir,
        model_ops_fn=model_ops_fn,
        encryption_config_pb=encryption_config_pb)
    learner_executor = LearnerExecutor(learner_task=learner_task)
    learner_server = LearnerServer(
        controller_server_entity_pb=controller_server_entity_pb,
        dataset_metadata=learner_dataset.get_dataset_metadata(),
        learner_executor=learner_executor,
        learner_server_entity_pb=learner_server_entity_pb,
        server_workers=5)
    learner_server.init_server()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # @stripeli the first 3-4 args are the most verbose I've ever seen :)
    # and some of them shorthands do not make sense, e.g. -e for neural engine :)
    parser.add_argument("-l", "--learner_server_entity_protobuff_serialized_hexadecimal", type=str,
                        default="",
                        help="Learner server entity.")
    parser.add_argument("-c", "--controller_server_entity_protobuff_serialized_hexadecimal", type=str,
                        default="",
                        help="Controller server entity.")
    parser.add_argument("-f", "--encryption_config_protobuff_serialized_hexadecimal", type=str,
                        default="",
                        help="A serialized Encryption Config protobuf message.")
    parser.add_argument("-e", "--neural_engine", type=str,
                        default="keras",
                        help="neural network training library")
    parser.add_argument("-m", "--model_dir", type=str,
                        default="",
                        help="model definition directory")
    parser.add_argument("-t", "--train_dataset", type=str,
                        default="",
                        help="train dataset filepath")
    parser.add_argument("-v", "--validation_dataset", type=str,
                        default="",
                        help="validation dataset filepath")
    parser.add_argument("-s", "--test_dataset", type=str,
                        default="",
                        help="test dataset filepath")
    parser.add_argument("-u", "--train_dataset_recipe", type=str,
                        default="",
                        help="train dataset recipe")
    parser.add_argument("-w", "--validation_dataset_recipe", type=str,
                        default="",
                        help="validation dataset recipe")
    parser.add_argument("-z", "--test_dataset_recipe", type=str,
                        default="",
                        help="test dataset recipe")
    args = parser.parse_args()
    init_learner(args)
