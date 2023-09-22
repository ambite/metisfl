from metisfl.controller import Controller
from metisfl.proto.proto_messages_factory import MetisProtoMessages

server_entity_pb = MetisProtoMessages.construct_server_entity_pb(
    hostname="localhost",
    port=2004
)
controller_params = MetisProtoMessages.construct_controller_params_pb(
    server_entity_pb=server_entity_pb,
    global_model_specs_pb=global_model_specs_pb,
    communication_specs_pb=communication_specs_pb,
    model_store_config_pb=model_store_config_pb,
    model_hyperparams_pb=model_hyperparams_pb
)

controller = Controller(controller_params)
controller.start()
