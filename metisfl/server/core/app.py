

import signal
from metisfl.common.types import (ControllerConfig, ModelStoreConfig,
                                  ServerParams)
from metisfl.server.core import (ControllerManager, ControllerServer,
                                 LearnerManager, ModelManager)
from metisfl.server.scheduling import (AsynchronousScheduler, Scheduler,
                                       SynchronousScheduler)
from metisfl.server.store import HashMapModelStore, ModelStore


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
        return SynchronousScheduler(controller_config.num_learners)
    elif controller_config.scheduler == "Asynchronous":
        return AsynchronousScheduler(controller_config.num_learners)
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
        self.server_params = server_params
        self.controller_config = controller_config
        self.model_store_config = model_store_config
               
        # Get dependencies 
        self.learner_manager = LearnerManager()
        self.model_store = get_model_store(model_store_config)
        self.model_manager = ModelManager(
            controller_config=self.controller_config,
            learner_manager=self.learner_manager,
            model_store=self.model_store,
        )
        
        # Create the controller manager
        self.controller_manager = ControllerManager(
            leaner_manager=self.learner_manager,
            model_manager=self.model_manager,
        )
        
        # Create the server
        self.server = ControllerServer(
            server_params=self.server_params,
            controller=self.controller_manager,
        )
    
    def start(self) -> None:
        # Start the server
        self.server.start()
        
        # Register the handlers for termination signals
        self.register_handlers()
        
    def register_handlers(self):
        signal.signal(signal.SIGINT, self.server.ShutDown)
        signal.signal(signal.SIGTERM, self.server.ShutDown)

        
        
        
        
        
    
        
        
        
            
        