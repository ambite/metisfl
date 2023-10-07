

from typing import Dict, List, Optional
from metisfl.common.client import get_client
from metisfl.common.utils import get_timestamp, random_id_generator
from metisfl.common.dtypes import ClientParams

from metisfl.proto import (controller_pb2, learner_pb2, learner_pb2_grpc,
                           model_pb2, service_common_pb2)


def get_learner_id(
    hostname: str,
    port: int,
) -> str:
    """Gets the learner id from the hostname and port."""
    return f"{hostname}:{port}"


class LearnerManager:

    # learner_id -> {}
    learners: Dict[str, controller_pb2.Learner] = {}
    client_params: Dict[str, ClientParams] = {}
    train_params: learner_pb2.TrainParams = {}
    eval_params: learner_pb2.EvaluationParams = {}

    # task_id -> {}
    tasks: Dict[str, learner_pb2.Task] = {}
    train_results: Dict[str, controller_pb2.TrainResults] = {}
    eval_results: Dict[str, learner_pb2.EvaluationResults] = {}

    # learner_id -> {}
    last_train_results: Dict[str, controller_pb2.TrainResults] = {}

    def add_learner(self, learner: controller_pb2.Learner) -> str:
        """Adds a learner to the controller.

        Parameters
        ----------
        learner : controller_pb2.Learner
            The learner to be added.
        """
        learner_id = get_learner_id(
            hostname=learner.hostname,
            port=learner.port,
        )

        if learner_id in self.learners:
            raise ValueError(f"Learner {learner_id} already exists.")

        self.learners[learner_id] = learner
        rt_bytes = bytes(learner.root_certificate_bytes, "utf-8") if learner.root_certificate_bytes else None
        self.client_params[learner_id] = ClientParams(
            hostname=learner.hostname,
            port=learner.port,
            root_certificate=rt_bytes,
        )

        return learner_id

    def remove_learner(self, learner_id: str) -> None:
        """Removes a learner from the controller.

        Parameters
        ----------
        learner_id : str
            The learner id.
        """
        if learner_id not in self.learners:
            raise ValueError(f"Learner {learner_id} does not exist.")

        self.learners.pop(learner_id)
        self.client_params.pop(learner_id)

    def get_client(self, learner_id: str):
        return get_client(
            stub_class=learner_pb2_grpc.LearnerServiceStub,
            client_params=self.client_params[learner_id],
        )

    def schedule_train(
        self,
        learner_ids: List[str],
        model: model_pb2.Model,
        request_retries: Optional[int] = 1,
        request_timeout: Optional[int] = None,
        block: Optional[bool] = False
    ) -> service_common_pb2.Ack:

        for learner_id in learner_ids:
            
            task = self.get_task(learner_id=learner_id)
            train_params = self.train_params.get(learner_id, None)
            
            with self.get_client(learner_id=learner_id) as client:
                
                stub: learner_pb2_grpc.LearnerServiceStub = client[0]
                schedule = client[1]
                
                def _request(_timeout=None):

                    request = learner_pb2.TrainRequest(
                        task=task,
                        model=model,
                        params=train_params
                    )
                    
                    return stub.Train(request, timeout=_timeout)

                schedule(_request, request_retries, request_timeout, block)

    def schedule_evaluate(
        self,
        learner_ids: List[str],
        model: model_pb2.Model,
        request_retries: Optional[int] = 1,
        request_timeout: Optional[int] = None,
        block: Optional[bool] = False
    ) -> service_common_pb2.Ack:

        for learner_id in learner_ids:
            
            task = self.get_task(learner_id=learner_id)
            eval_params = self.eval_params.get(learner_id, None)
            
            with self.get_client(learner_id=learner_id) as client:

                stub: learner_pb2_grpc.LearnerServiceStub = client[0]
                schedule = client[1]

                def _request(_timeout=None):

                    request = learner_pb2.EvaluateRequest(
                        task=task,
                        model=model,
                        params=eval_params
                    )

                    return stub.Evaluate(request, timeout=_timeout)

                schedule(_request, request_retries, request_timeout, block)

    def get_task(self, learner_id: str) -> learner_pb2.Task:
        """ Gets a task for a learner. """
        
        task = learner_pb2.Task(
            id=random_id_generator(),
            learner_id=learner_id,
            sent_at=get_timestamp(),
        )
        self.tasks[task.id] = task
        return task

    def shutdown_client(self):
        """Shuts down the client."""
        with self.get_client() as client:
            client[2].shutdown()

    def get_num_training_examples(self, learner_ids: List[str]) -> Dict[str, int]:
        """Gets the number of training examples for the learners.

        Parameters
        ----------
        learner_ids : List[str]
            The learner ids.

        Returns
        -------
        Dict[str, int]
            The number of training examples for each learner.
        """
        num_training_examples = {}
        for learner_id in learner_ids:
            train_result = self.last_train_results.get(learner_id)
            if not hasattr(train_result, "metadata") or not hasattr(train_result.metadata, "num_training_examples"):
                num_training_examples[learner_id] = 1 # TODO: fix this
            else:
                num_training_examples[learner_id] = train_result.metadata.get("num_training_examples")
            
        return num_training_examples

    def get_num_completed_batches(self, learner_ids: List[str]) -> Dict[str, int]:
        """Gets the number of completed batches for the learners.

        Parameters
        ----------
        learner_ids : List[str]
            The learner ids.

        Returns
        -------
        Dict[str, int]
            The number of completed batches for each learner.
        """
        num_completed_batches = {}
        for learner_id in learner_ids:
            train_result = self.last_train_results.get(learner_id)
            if not hasattr(train_result, "metadata") or not hasattr(train_result.metadata, "num_completed_batches"):
                num_completed_batches[learner_id] = 1
            else:
                num_completed_batches[learner_id] = train_result.metadata.get("num_completed_batches")
                
        return num_completed_batches

    def get_learner_ids(self) -> List[str]:
        """Gets the learner ids.

        Returns
        -------
        List[str]
            The learner ids.
        """
        return list(self.learners.keys())

    def get_learner_id(self, task_id: str) -> str:
        """Gets the learner id of a task."""
        return self.tasks[task_id].learner_id

    def update_train_result(
        self,
        task: learner_pb2.Task,
        learner_id: str,
        train_results: controller_pb2.TrainResults,
    ) -> None:
        """Updates the train result of a learner.

        Parameters
        ----------
        task : learner_pb2.Task
            The newly completed task.
        learner_id : str
            The learner id.
        train_result : controller_pb2.TrainResults
            The train result of the task.
        """
        
        task_id = task.id
        
        self.train_results[task_id] = train_results
        
        self.last_train_results[learner_id] = train_results

        self.tasks[task_id] = learner_pb2.Task(
            id=task_id,
            learner_id=learner_id,
            sent_at=self.tasks[task_id].sent_at,
            received_at=task.received_at,
            completed_at=task.completed_at,
        )
