
""" Builds MetisFL wheel package. """

import os
import sys

from setuptools import setup, find_packages

__version__ = "0.1.0"

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))

# Compile proto files
os.system(f"{sys.executable} {ROOT_DIR}/metisfl/proto/protoc.py")

setup(
    name="metisfl",
    version=__version__,
    description="MetisFL: The developer-friendly federated learning framework",
    author="MetisFL Team",
    author_email="hello@nevron.ai",
    url="https://github.com/nevronai/metisfl",
    classifiers=[
            "Development Status :: 2 - Pre-Alpha",
            "Intended Audience :: Developers",
            "Topic :: Software Development :: Testing",
            "Topic :: Software Development :: Libraries",
            "Programming Language :: Python",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: C++"
    ],
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "Pebble>=5.0.3",
        "PyYAML>=6.0",
        "pandas>=1.3.2",
        "protobuf>=4.23.4",
        "termcolor>=2.3.0",
        "pyfiglet>=0.8.post1",
        "loguru>=0.7.1",
    ],
    python_requires=">=3.8",
)
