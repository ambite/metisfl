""" Entrypoint for the controller. """

import signal

from metisfl.common.dtypes import (ControllerConfig, ModelStoreConfig,
                                   ServerParams)
from metisfl.controller.aggregation.aggregation import Aggregator
from metisfl.controller.aggregation.federated_average import FederatedAverage
from metisfl.controller.learners import LearnerManager
from metisfl.controller.manager import ControllerManager
from metisfl.controller.model import ModelManager
from metisfl.controller.scheduling import (AsynchronousScheduler, Scheduler,
                                           SynchronousScheduler)
from metisfl.controller.selection import ScheduledCardinality
from metisfl.controller.server import ControllerServer
from metisfl.controller.store import HashMapModelStore, ModelStore


def get_aggregator(controller_config: ControllerConfig) -> Aggregator:
    """Returns the aggregator."""
    
    if controller_config.aggregation_rule == "FedAvg":
        return FederatedAverage()
    else:
        raise ValueError("Invalid aggregator")

def get_model_store(model_store_config: ModelStoreConfig) -> ModelStore:
    """Returns the model store."""    
    
    if model_store_config.model_store == "InMemory":
        return HashMapModelStore(model_store_config.lineage_length)
    else:
        raise ValueError("Invalid model store")
    
def get_scheduler(controller_config: ControllerConfig) -> Scheduler:
    """Returns the scheduler."""
    
    if controller_config.scheduler == "Synchronous" or \
        controller_config.scheduler == "SemiSynchronous":
        return SynchronousScheduler()
    elif controller_config.scheduler == "Asynchronous":
        return AsynchronousScheduler()
    else:
        raise ValueError("Invalid scheduler")        


class Controller:
    
    def __init__(
        self,
        server_params: ServerParams,
        controller_config: ControllerConfig,
        model_store_config: ModelStoreConfig,
    ) -> None:
        """Initializes the Controller.

        Parameters
        ----------
        server_params : ServerParams
            The server parameters.
        controller_config : ControllerConfig
            The controller configuration.
        model_store_config : ModelStoreConfig
            The model store configuration.
        """
               
        # Get dependencies 
        aggregator = get_aggregator(controller_config)
        learner_manager = LearnerManager()
        model_store = get_model_store(model_store_config)
        model_manager = ModelManager(
            aggregator=aggregator,
            controller_config=controller_config,
            learner_manager=learner_manager,
            model_store=model_store,
        )
        selector = ScheduledCardinality()
        scheduler = get_scheduler(controller_config)
        
        # Create the controller manager
        controller_manager = ControllerManager(
            learner_manager=learner_manager,
            model_manager=model_manager,
            selector=selector,
            scheduler=scheduler,
        )
        
        # Create the server
        self.server = ControllerServer(
            server_params=server_params,
            controller_manager=controller_manager,
        )
    
    def start(self) -> None:
        # Start the server
        self.server.start()
        
        # Register the handlers for termination signals
        self.register_handlers()
        
    def register_handlers(self):
        signal.signal(signal.SIGINT, self.server.ShutDown)
        signal.signal(signal.SIGTERM, self.server.ShutDown)

        
        
        
        
        
    
        
        
        
            
        