# Development Guide

Thank you for you interest in contributing to MetisFL. We welcome all contributions from bug fixes to new features and documentation. For ideas on what to contribute, please check out our [issues](https://github.com/NevronAI/MetisFL/issues) page. This guide will help you get started with contributing to MetisFL. If you are new to open source, please check out this [guide](https://docs.github.com/en/get-started/quickstart/contributing-to-projects).

# Development Environment
First, you need to setup you development environment. Currently, the setup mentioned below has been tested on Linux based systems but should work on any system that supports Python 3.8+. MetisFL uses poetry to manage its dependencies. To install poetry, please follow the instructions [here](https://python-poetry.org/docs/#installation). Once you have installed poetry, you can install the dependencies of MetisFL using the following command:

```
poetry install
```

To activate the virtual environment created by poetry, you can use the following command:

```
poetry shell
```

This is going to activate the virtual environment and you can now develop, run and test your contributions. Before starting your work, please make sure that you run the following command to compile the protobuf files:

```
python metis/proto/protoc.py
```
This is going to compile the protobuf files and generate the necessary python classes and pyi files under the `metis/proto` directory.


After you have finished your work, you can start a PR to merge your changes into the [main](https://github.com/NevronAI/MetisFL/tree/main) branch.
