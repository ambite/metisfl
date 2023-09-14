# 🚀 MetisFL Generative AI Tutorial


## ⚙️ Prerequisites

Before running this example, please make sure you have installed the MetisFL package

```bash
pip install metisfl
```

The default installation of MetisFL does not include any backend. This example uses Pytorch as a backend as well as torchvision to load the CIFAR10 dataset. Both can be installed using pip.

```bash
pip install torch torchvision
```

## 📝 Overview & Configuration

```python
# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

# Decide which device we want to run on
device = torch.device("cuda:0" if (
    torch.cuda.is_available() and ngpu > 0) else "cpu")

# Root directory for dataset
dataroot = "celeba"

# Number of workers for dataloader
workers = 2

# Batch size during training
batch_size = 128

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 64

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Number of training epochs
num_epochs = 5

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparameter for Adam optimizers
beta1 = 0.5

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.
```

## 💾 Dataset

## 🧠 Model

## 👨‍💻 MetisFL Learner

The main abstraction of the client is called MetisFL Learner. The MetisFL Learner is responsible for training the model on the local dataset and communicating with the server. Following the [class](https://github.com/NevronAI/metisfl/blob/main/metisfl/learner/learner.py) that must be implemented by the learner, we first start by the `get_weights` and `set_weights` methods. These methods are used by the Controller to get and set the model parameters. The `get_weights` method returns a list of numpy arrays and the `set_weights` method takes a list of numpy arrays as input.

```python
def get_weights(self):
    return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

def set_weights(self, parameters):
    params = zip(self.model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params})
    self.model.load_state_dict(state_dict, strict=True)
```

Then, we implement the `train` and `evaluate` methods. Both of them take the model weights and a dictionary of configuration parameters as input. The `train` method returns the updated model weights, a dictionary of metrics and a dictionary of metadata. The `evaluate` method returns a dictionary of metrics.

## 🎛️ MetisFL Controller

The Controller is responsible for send training and evaluation tasks to the learners and for aggregating the model parameters. The entrypoint for the Controller is `Controller` class found [here](https://github.com/NevronAI/metisfl/blob/127ad7147133d25188fc07018f2d031d6ad1b622/metisfl/controller/controller_instance.py#L10). The `Controller` class is initialized with the parameters of the Learners and the global training configuration.

```python
controller_params = ServerParams(
    hostname="localhost",
    port=50051,
)

controller_config = ControllerConfig(
    aggregation_rule="FedAvg",
    scheduler="Synchronous",
    scaling_factor="NumParticipants",
)

model_store_config = ModelStoreConfig(
    model_store="InMemory",
    lineage_length=0
)
```

The ServerParams define the hostname and port of the Controller and the paths to the root certificate, server certificate and private key. Certificates are optional and if not given then SSL is not active. The ControllerConfig defines the aggregation rule, scheduler and model scaling factor.

For the full set of options in the ControllerConfig please have a look [here](https://github.com/NevronAI/metisfl/blob/127ad7147133d25188fc07018f2d031d6ad1b622/metisfl/common/types.py#L99). Finally, this example uses an "InMemory" model store with no eviction (`lineage_length=0`). A positive value for `lineage_length` means that the Controller will start dropping models from the model store after the given number of models, starting from the oldest.

## 🚦 MetisFL Driver

The MetisFL Driver is the main entry point to the MetisFL application. It will initialize the model weights by requesting the model weights from a random learner and then distributing the weights to all learners and the controller. Additionally, it monitor the federation and will stop the training process when the termination condition is met.

```python
# Setup the environment.
termination_signals = TerminationSingals(
    federation_rounds=5)
learners = [get_learner_server_params(i) for i in range(max_learners)]
is_async = controller_config.scheduler == 'Asynchronous'

# Start the driver session.
session = DriverSession(
    controller=controller_params,
    learners=learners,
    termination_signals=termination_signals,
    is_async=is_async,
)

# Run
logs = session.run()
```

To see and experiment with the different termination conditions, please have a look at the TerminationsSignals class [here](https://github.com/NevronAI/metisfl/blob/127ad7147133d25188fc07018f2d031d6ad1b622/metisfl/common/types.py#L18).

## 🎬 Running the example

To run the example, you need to open one terminal for the Controller, one terminal for each Learner and one terminal for the Driver. First, start the Controller.

```bash
python controller.py
```

Then, start the Learners.

```bash
python learner.py -l X
```

where `X` is the numerical id of the Learner (1,2,3). Note that both the learner and driver scripts have been configured to use 3 learners by default. If you want to experiment with a different number of learners, you need to change the `max_learners` variable in both scripts. Also, please make sure to start the controller before the Learners otherwise the Learners will not be able to connect to the Controller.

Finally, start the Driver.

```bash
python driver.py
```

The Driver will start the training process and each terminal will show the progress. The experiment will run for 5 federation rounds and then stop. The logs will be saved in the `results.json` file in the current directory.

## 🚀 Next steps

Congratulations 👏 you have successfully run your first MetisFL federated learning experiment using Pytorch! And you should see an output similar to the image on the top of this page. You may notice that the performance of the model is not that good. You can try to improve it by experimenting both the the federated learning parameters (e.g., the number of learners, federation rounds, aggregation rule) as well as with the typical machine learning parameters (e.g., learning rate, batch size, number of epochs, model architecture).

Please share your results with us or ask any questions that you might have on our [Slack channel](https://nevronai.slack.com/archives/C05E9HCG0DB). We would love to hear from you!
