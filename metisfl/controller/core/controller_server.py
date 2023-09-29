
import threading
from typing import Any, Union
import grpc
from metisfl.common.formatting import get_timestamp
from metisfl.common.server import Server
from metisfl.common.types import ServerParams
from metisfl.proto import model_pb2, controller_pb2, controller_pb2_grpc, service_common_pb2
from metisfl.controller.core import ControllerManager


class ControllerServer(Server, controller_pb2_grpc.ControllerServiceServicer):

    controller_manager: ControllerManager = None

    def __init__(
        self,
        server_params: ServerParams,
        controller_manager: ControllerManager,
    ) -> None:
        """Initializes the Controller Server.

        Parameters
        ----------
        server_params : ServerParams
            The server parameters.
        controller_manager : ControllerManager
            The controller manager to be attached to the server.
        """
        super().__init__(
            server_params=server_params,
            add_servicer_to_server_fn=controller_pb2_grpc.add_ControllerServiceServicer_to_server,
        )
        self.controller_manager = controller_manager

    def SetInitialModel(
        self,
        model: model_pb2.Model,
        context: Any
    ) -> service_common_pb2.Ack:
        """Sets the initial weights of the model.

        Parameters
        ----------
        request : model_pb2.Model
            The ProtoBuf object containing the model.
        context : Any
            The gRPC context of the request.

        Returns
        -------
        service_common_pb2.Ack
            The response containing the acknoledgement.
        """

        if not self.is_serving(context):
            return service_common_pb2.Ack(status=False)

        status = self.controller_manager.set_initial_model(model=model)

        return service_common_pb2.Ack(
            status=status,
            timestamp=get_timestamp(),
        )

    def JoinFederation(
        self,
        learner: controller_pb2.Learner,
        context: Any
    ) -> Union[controller_pb2.LearnerId, service_common_pb2.Ack]:
        """Joins the federation.

        Parameters
        ----------
        request : controller_pb2.Learner
            The ProtoBuf object containing the learner.
        context : Any
            The gRPC context of the request.

        Returns
        -------
        Union[controller_pb2.LearnerId, service_common_pb2.Ack]
            The response containing the learner id or a failure acknoledgement if the server is not serving.
        """

        if not self.is_serving(context):
            return service_common_pb2.Ack(status=False)

        learner_id = self.controller_manager.add_learner(learner=learner)

        return controller_pb2.LearnerId(
            learner_id=learner_id,
        )

    def LeaveFederation(
        self,
        learner_id: controller_pb2.LearnerId,
        context: Any
    ) -> service_common_pb2.Ack:
        """Leaves the federation.

        Parameters
        ----------
        request : controller_pb2.LearnerId
            The ProtoBuf object containing the learner id.
        context : Any
            The gRPC context of the request.

        Returns
        -------
        service_common_pb2.Ack
            The response containing the acknoledgement.
        """

        if not self.is_serving(context):
            return service_common_pb2.Ack(status=False)

        self.controller_manager.remove_learner(learner_id=learner_id.learner_id)

        return service_common_pb2.Ack(
            status=True
        )

    def StartTraining(
        self,
        _: service_common_pb2.Empty,
        context: Any
    ) -> service_common_pb2.Ack:
        """Starts the training process.

        Parameters
        ----------
        request : service_common_pb2.Empty
            The ProtoBuf object containing the empty request.
        context : Any
            The gRPC context of the request.

        Returns
        -------
        service_common_pb2.Ack
            The response containing the acknoledgement.
        """

        if not self.is_serving(context):
            return service_common_pb2.Ack(status=False)

        self.controller_manager.start_training()

        return service_common_pb2.Ack(
            status=True
        )

    def TrainDone(
        self,
        request: controller_pb2.TrainDoneRequest,
        context: Any
    ) -> service_common_pb2.Ack:
        """Notifies the controller that the training is done.

        Parameters
        ----------
        request : controller_pb2.TrainDoneRequest
            The request containing information about the task and model.
        context : Any
            The gRPC context of the request.

        Returns
        -------
        service_common_pb2.Ack
            The response containing the acknoledgement.
        """

        if not self.is_serving(context):
            return service_common_pb2.Ack(status=False)

        self.controller_manager.train_done(request=request)

        return service_common_pb2.Ack(
            status=True
        )

    def GetLogs(
        self,
        _: service_common_pb2.Empty,
        context: Any
    ) -> Union[controller_pb2.Logs, service_common_pb2.Ack]:
        """Gets the logs of the controller.

        Parameters
        ----------
        request : service_common_pb2.Empty
            The ProtoBuf object containing the empty request.
        context : Any
            The gRPC context of the request.

        Returns
        -------
        Union[controller_pb2.Logs, service_common_pb2.Ack]
            The response containing the logs or a failure acknoledgement if the server is not serving.
        """

        if not self.is_serving(context):
            return service_common_pb2.Ack(status=False)

        logs = self.controller_manager.get_logs()

        return controller_pb2.Logs(
            logs=logs,
        )

    def ShutDown(self, _: service_common_pb2.Empty, __: Any) -> service_common_pb2.Ack:
        return super().ShutDown(_, __)
