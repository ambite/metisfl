
import signal

from metisfl.learner.learner import Learner
from metisfl.learner.learner_server import LearnerServer
from metisfl.learner.learner_executor import LearnerExecutor
from metisfl.learner.learner_task import LearnerTask
from metisfl.common.logger import MetisLogger
from metisfl.model.utils import get_model_ops_fn

def register_handlers(learner_server: LearnerServer):
    """ Register handlers for SIGTERM and SIGINT to leave the federation.

    Parameters∏
    ----------
    client : GRPCClient
        The GRPCClient object.
    server : LearnerServer
        The LearnerServer object.
    """

    def handler(signum, frame):
        MetisLogger.info("Received SIGTERM, leaving federation...")
        learner_server.ShutDown()
        exit(0)

    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)


def app(learner: Learner):
    """
    
    Entry point for the MetisFL Learner application.

    Parameters
    ----------
    learner : Learner object

    """
    model_ops_fn = get_model_ops_fn(
        learner.metis_model.get_neural_engine)    
    learner_task = LearnerTask(
        learner_server_entity_pb=learner.learner_server_entity_pb,
        learner_dataset=learner.learner_dataset,
        model_dir=learner.model_dir,
        model_ops_fn=model_ops_fn,
        encryption_config_pb=learner.encryption_config_pb)
    learner_executor = LearnerExecutor(learner_task)
    learner_server = LearnerServer(
        learner_executor=learner_executor,                
        controller_server_entity_pb=learner.controller_server_entity_pb,
        learner_server_entity_pb=learner.learner_server_entity_pb,
        learner_dataset=learner.learner_dataset)
    
    # Register shutdown handlers.
    register_handlers(learner_server)

    # Start the Learner server; blocking call
    learner_server.init_server()
