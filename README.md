# MetisFL: The blazing-fast and developer-friendly federated learning framework.

&nbsp;

MetisFL is a federated learning framework that allows developers to federate their machine learning workflows and train their models across distributed datasets without having to collect the data in a centralized location. Currently, the project is transitioning from a private, experimental version to a public, beta phase. We are actively encouraging developers, researchers and data scientists to experiment with the framework and contribute to the codebase.

MetisFL sprung up from the Information and Science Institute (ISI) in the University of Southern California (USC). It is backed by several years of Ph.D. and Post-Doctoral research and several publications in top-tier machine learning and system conferences. 

# Quickstart (coming soon)

As an introductory example to quickly demonstrate the MetisFL framework in practice, we will run the `Hello World` example of deep learning. To get started, install the MetisFL python library using pip:

```Bash
pip install metisfl
```

and then clone the repository on you local machine:

```Bash
git clone https://github.com/NevronAI/metisfl.git
```

Navigate to the `Tensorflow Quickstart` example under `examples/tensorflow-quickstart` and open up 5 terminals. In the first terminal, run the controller:

```Bash
python controller.py
```

This is going to startup the controller at port 50001 in the local machine. In the next 3 terminals run the learners:

```Bash
python learner.py -l ID
```

where `ID` is the unique identifier of the learner (1, 2, 3). This will startup 3 learners servers at ports 50002, 50003 and 50004 and the learners will connect to the controller and wait for the training task. Finally, in the last terminal run the driver:

```Bash
python driver.py
```

The driver will initiate the federated training, monitor the controller and learners and terminate the experiment once 5 federated rounds have been completed. Congratulations! You are now running your first federated learning experiment using MetisFL!

# Documentation and Examples

When you are ready to dive deeper into the framework, check out the following resources:

- [Documentation](https://docs.nevron.ai/)
- [What is Federated Learning?](https://docs.nevron.ai/introduction/federated-learning)
- [Install & Configure](https://docs.nevron.ai/quickstart/installation)
- [MetisFL with PyTorch](https://docs.nevron.ai/quickstart/torch)
- [MetisFL with Tensorflow](https://docs.nevron.ai/quickstart/tensorflow)
- [MetisFL with SKLearn](https://docs.nevron.ai/quickstart/sklearn)

# Project Structure Overview

The project is organized as follows:

    .
    ├── examples              # Examples and use-cases for MetisFL
    ├── metisfl               # Main source code folder
        ├── common            # Common utilities and helper functions
        ├── controller        # The MetisFL Controller/Aggregator
        ├── driver            # Python library for the MetisFL Driver
        ├── encryption        # Fully Homomorphic Encryption (FHE) library
        ├── helpers           # Helper functions to generate encryption keys for FHE/SSL       
        ├── learner           # Python Learner library
        ├── proto             # Protobuf definitions 
    ├── test                  # Testing folder (under construction)
    ├── CONTRIBUTING.md       # Contribution guidelines
    ├── LICENSE               # License file
    ├── pyproject.toml        # Poetry configuration file
    └── README.md             # This file

# Architecture Overview

The architecture of MetisFL consists of three main components: the **Federation Controller**, the **Federation Learner** and the **Federation Driver**.

<div align="center">
<picture>
  <source media="(prefers-color-scheme: light)" srcset="https://docs.nevron.ai/img/light/MetisFL-Components-Overview-02.webp" width="700px">
  <img alt="MetisFL Components Overview" src="https://docs.nevron.ai/img/dark/MetisFL-Components-Overview-01.webp" width="700px">
</picture>
</div>

## Federation Controller

The Federation Controller is responsible for selecting and delegating training and evaluation tasks to the federation learners, for receiving, aggregating and storing the model updates and for storing the training logs and metadata. For the training tasks, the communication between the Learner and Controller is asynchronous at the protocol level. The Controller sends the training task to the Learner and the Learner sends back a simple acknowledgement of receiving the task. When the Learner finishes the task it sends the results back to the Controller by calling its `TrainDone` endpoint. For the evaluation tasks, the communication is synchronous at the protocol level, i.e., the Controller sends the evaluation task to the Learner and waits for the results using the same channel.

- Aggregation Rules: Federated Average, Federated Recency, Federated Stride, Secure Aggregation
- Schedulers: Synchronous, Semi-Synchronous, Asynchronous
- Model Storage: In-memory, Redis (beta)
- Secure gRPC communication using SSL/TLS

## Federation Learner

The main abstraction of the client is called MetisFL Learner. The MetisFL Learner is responsible for training the model on the local dataset and communicating with the server. The Controller sends the evaluation task to the Learner and waits for the results using the same channel. The abstract class that defines the Learner can be found [here](https://github.com/NevronAI/metisfl/blob/main/metisfl/learner/learner.py). Each MetisFL learner must implement the following methods:

- `set_weights`: This method is called by the Controller to set the initial model weights. The Learner must implement the logic to set the weights of the local model.
- `get_weights`: This method is called by the Controller to get the current model weights. The Learner must implement the logic to return the weights of the local model.
- `train`: This method is called by the Controller to initiate the training task. The input to this method is the current model weights and a dictionary of hyperparameters. The Learner must implement the training logic and return the updated model weights to the Controller.
- `evaluate`: This method is called by the Controller to initiate the evaluation task. The Learner must implement the evaluation logic and return the evaluation metrics to the Controller.

## Federation Driver

The Driver orchestrates the training process by communicating with the Controller and the Learners. First, it initializes the model weights by requesting the initial values from a randomly selected learner and distributing to everyone else. Additionally, it monitors the federation for the termination signal(s) and shuts the learners/controller down once the signal is reached. Finally, it collects the training logs and metadata from the Controller and stores them in a local file.

## Contributing
We currently welcome contributions in the form of bug fixes, feature requests and documentation. If you are interested in contributing to the project, please check out our [contribution guidelines](
  https://github.com/NevronAI/MetisFL/blob/main/CONTRIBUTING.md)
