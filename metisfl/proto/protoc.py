""" This module contains functions related to proto compilation. """

import glob
import grpc_tools

from grpc_tools import protoc
from os import path

GRPC_PATH = grpc_tools.__path__[0]

DIR_PATH = path.dirname(path.realpath(__file__))
IN_PATH = path.normpath(path.join(DIR_PATH, "..", ".."))
OUT_PATH = IN_PATH
PROTO_FILES = glob.glob(f"{DIR_PATH}/*.proto")


def compile() -> None:
    """ Compile all protos in the `metisfl/proto` directory. """

    try:
        command = [
            "grpc_tools.protoc",
            f"--proto_path={GRPC_PATH}/_proto",
            f"--proto_path={IN_PATH}",
            f"--python_out={OUT_PATH}",
            f"--grpc_python_out={OUT_PATH}",
        ] + PROTO_FILES    
        protoc.main(command)
    except Exception as error:
        raise RuntimeError(f"An error occured while compiling protos, {error}.")


if __name__ == "__main__":
    compile()
