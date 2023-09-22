import glob
import os
import shutil
import site
import sys

PY_VERSIONS = ["3.8", "3.9", "3.10"]
os.environ["PYTHON_BIN_PATH"] = sys.executable
os.environ["PYTHON_LIB_PATH"] = site.getsitepackages()[0]

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))

BAZEL_CMD = "bazelisk"
BUILD_DIR = "build"

CONTROLER_SO_DST = "metisfl/controller/controller.so"
CONTROLER_SO_SRC = "bazel-bin/metisfl/controller/controller.so"
CONTROLER_SO_TARGET = "//metisfl/controller:controller.so"

FHE_SO_DST = "metisfl/encryption/fhe.so"
FHE_SO_SRC = "bazel-bin/metisfl/encryption/fhe.so"
FHE_SO_TARGET = "//metisfl/encryption:fhe.so"


def copy_helper(src_path, dst):
    """ 
    Copies a file to the given destination. 
    If the destination is a directory, the file is copied into it. 
    """

    if os.path.isdir(dst):
        fname = os.path.basename(src_path)
        dst = os.path.join(dst, fname)

    if os.path.isfile(dst):
        os.remove(dst)

    shutil.copy(src_path, dst)


def run_build(python_verion):

    """ Builds MetisFL wheel package for the passing Python version."""

    # Build .so and proto/grpc classes
    os.system("{} build {}".format(BAZEL_CMD, CONTROLER_SO_TARGET))
    os.system("{} build {}".format(BAZEL_CMD, FHE_SO_TARGET))

    # Compile proto files
    os.system(f"{sys.executable} {ROOT_DIR}/metisfl/proto/protoc.py")

    # Copy .so
    copy_helper(CONTROLER_SO_SRC, CONTROLER_SO_DST)
    copy_helper(FHE_SO_SRC, FHE_SO_DST)


    # Build wheel.
    os.system(
        "{bazel} build //:metisfl-wheel --define python={python}".format(
            bazel=BAZEL_CMD, python=python_verion
        )
    )

    # Copy wheel.
    os.makedirs(BUILD_DIR, exist_ok=True)
    for file in glob.glob("bazel-bin/*.whl"):
        copy_helper(file, BUILD_DIR)


def copy_helper(src_file, dst):
    if os.path.isdir(dst):
        fname = os.path.basename(src_file)
        dst = os.path.join(dst, fname)

    if os.path.isfile(dst):
        os.remove(dst)
    shutil.copy(src_file, dst)


if __name__ == "__main__":

    """ Builds MetisFL wheel package for the current Python version. """

    py_version = ".".join(
        map(str, [sys.version_info.major, sys.version_info.minor]))
    if py_version not in PY_VERSIONS:
        raise ValueError(
            "Python version {} is not supported. Supported versions are {}".format(
                py_version, PY_VERSIONS
            ))
    lib_path = os.environ["PYTHON_LIB_PATH"]
    if not os.path.isdir(lib_path):
        raise ValueError("PYTHON_LIB_PATH {} does not exist".format(lib_path))
    
    run_build(py_version)
