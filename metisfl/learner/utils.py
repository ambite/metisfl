import metisfl.config as config
import metisfl.proto.metis_pb2 as metis_pb2

from metisfl.proto.proto_messages_factory import MetisProtoMessages

def parse_server_entity_hex(hex_str, default_host, default_port):
    if hex_str:
        server_entity_pb = metis_pb2.ServerEntity()
        server_entity_pb_ser = bytes.fromhex(hex_str)
        server_entity_pb.ParseFromString(server_entity_pb_ser)
    else:
        server_entity_pb = MetisProtoMessages.construct_server_entity_pb(
            hostname=default_host, port=default_port)
    return server_entity_pb

def parse_encryption_config_hex(hex_str):
    if hex_str:
        encryption_config_protobuff_ser = bytes.fromhex(hex_str)
        encryption_config_pb = metis_pb2.EncryptionConfig()
        encryption_config_pb.ParseFromString(encryption_config_protobuff_ser)
        return encryption_config_pb
    else:
        return None

def create_servers(args):
    learner_server_entity_pb = parse_server_entity_hex(
        args.learner_server_entity_protobuff_serialized_hexadecimal,
        config.DEFAULT_LEARNER_HOST, config.DEFAULT_LEARNER_PORT)
    controller_server_entity_pb = parse_server_entity_hex(
        args.controller_server_entity_protobuff_serialized_hexadecimal,
        config.DEFAULT_CONTROLLER_HOSTNAME, config.DEFAULT_CONTROLLER_PORT)
    return learner_server_entity_pb, controller_server_entity_pb
