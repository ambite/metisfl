import sys
import os
import shutil
import glob

BAZEL_CMD = "bazelisk"
BUILD_DIR = "build"
CONTROLER_SO_DST = "metisfl/controller/controller.so"
CONTROLER_SO_SRC = "bazel-bin/metisfl/controller/controller.so"
CONTROLER_SO_TARGET = "//metisfl/controller:controller.so"
DOCKER_REPO = "us-west2-docker.pkg.dev/nevron-385600/builders/ubuntu_focal_x86_64_py38"
FHE_SO_DST = "metisfl/encryption/fhe.so"
FHE_SO_SRC = "bazel-bin/metisfl/encryption/fhe.so"
FHE_SO_TARGET = "//metisfl/encryption:fhe.so"
PROT_DST_DIR = "metisfl/proto"
PROTO_GRPC_TARGET = "//metisfl/proto:py_grpc_src"
PROTO_SRC_DIR = "bazel-bin/metisfl/proto/py_grpc_src/metisfl/proto/"
PY_VERSIONS = ["3.8", "3.9", "3.10"]


def run_build(python_verion):
    # Build targets
    os.system("{} build {}".format(BAZEL_CMD, CONTROLER_SO_TARGET))
    os.system("{} build {}".format(BAZEL_CMD, FHE_SO_TARGET))
    os.system("{} build {}".format(BAZEL_CMD, PROTO_GRPC_TARGET))

    # Copy .so
    copy_helper(CONTROLER_SO_SRC, CONTROLER_SO_DST)
    copy_helper(FHE_SO_SRC, FHE_SO_DST)

    # Copy proto/grpc classes
    for file in glob.glob("{}*.py".format(PROTO_SRC_DIR)):
        copy_helper(file, PROT_DST_DIR)

    # Build wheel
    os.system(
        "{bazel} build //:metisfl-wheel --define python={python}".format(
            bazel=BAZEL_CMD, python=python_verion
        )
    )

    # Copy wheel
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
    py_version = ".".join(map(str, sys.version_info[:2]))
    if py_version not in PY_VERSIONS:
        print(
            "Detected Python {} in environment. Need {}".format(py_version, PY_VERSIONS)
        )
        exit(1)
    run_build(py_version)