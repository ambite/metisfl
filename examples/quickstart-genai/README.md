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

<div align="center">
  <img src="https://mmlab.ie.cuhk.edu.hk/projects/CelebA/intro.png" width="800px" />
</div>

The dataset used in this example is the well-known Celeba dataset which contains more than 200k images of celebrities. The dataset is available [here](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). 
The dataset is partitioned into 3 chunks and each chunk is stored in a separate directory. The partitioning is done using the `iid_partition_dir` function from the `metisfl.utils` module. A subtle thing to notice here is that in each chunk dir there should be a sub-directory which contains the actual images. This is because the `ImageFolder` class from the `torchvision.datasets` module expects the images to be partitioned into a sub-directory depending on their class. In our case, we have only one class we can just create a sub-directory `img` in each chunk dir and move all the images there.  

```python
def get_dataloaders(num_learners: int) -> List[torch.utils.data.DataLoader]:
    """Partitions the files under the cfg.dataroot into num_learners chunks."""
    chunk_dirs = glob.glob(cfg.dataroot + '/chunk*')
    
    if len(chunk_dirs) > 0:   
        assert len(chunk_dirs) == num_learners, "Number of learners must match the number of chunks"
    else:
        print("Partitioning data into {} chunks".format(num_learners))
        iid_partition_dir(cfg.dataroot, 'jpg', num_learners, "img")
        chunk_dirs = glob.glob(cfg.dataroot + '/chunk*')    
    
    dataloaders = []
    for chunk_dir in chunk_dirs:        
        dataset = dset.ImageFolder(root=chunk_dir,
                                transform=transforms.Compose([
                                    transforms.Resize(cfg.image_size),
                                    transforms.CenterCrop(cfg.image_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize(
                                        (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ]))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg.batch_size,
                                                shuffle=True, num_workers=cfg.workers)
        dataloaders.append(dataloader)
        
    return dataloaders

def get_dataloader(learner_index: int, num_learners: int) -> torch.utils.data.DataLoader:
    """Returns the dataloader for the learner with index learner_index."""
    dataloaders = get_dataloaders(num_learners)
    return dataloaders[learner_index]
```

## 🧠 Model

```python
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(cfg.nz, cfg.ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(cfg.ngf * 8),
            nn.ReLU(True),
            # state size. ``(ngf*8) x 4 x 4``
            nn.ConvTranspose2d(cfg.ngf * 8, cfg.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(cfg.ngf * 4),
            nn.ReLU(True),
            # state size. ``(ngf*4) x 8 x 8``
            nn.ConvTranspose2d(cfg.ngf * 4, cfg.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(cfg.ngf * 2),
            nn.ReLU(True),
            # state size. ``(ngf*2) x 16 x 16``
            nn.ConvTranspose2d(cfg.ngf * 2, cfg.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(cfg.ngf),
            nn.ReLU(True),
            # state size. ``(ngf) x 32 x 32``
            nn.ConvTranspose2d(cfg.ngf, cfg.nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. ``(nc) x 64 x 64``
        )

    def forward(self, input):
        return self.main(input)
```

```python
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is ``(nc) x 64 x 64``
            nn.Conv2d(cfg.nc, cfg.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf) x 32 x 32``
            nn.Conv2d(cfg.ndf, cfg.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(cfg.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*2) x 16 x 16``
            nn.Conv2d(cfg.ndf * 2, cfg.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(cfg.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*4) x 8 x 8``
            nn.Conv2d(cfg.ndf * 4, cfg.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(cfg.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*8) x 4 x 4``
            nn.Conv2d(cfg.ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
```

## 👨‍💻 MetisFL Learner

The main abstraction of the client is called MetisFL Learner. The MetisFL Learner is responsible for training the model on the local dataset and communicating with the server. Following the [class](https://github.com/NevronAI/metisfl/blob/main/metisfl/learner/learner.py) that must be implemented by the learner, we first start by the `get_weights` and `set_weights` methods. These methods are used by the Controller to get and set the model parameters. The `get_weights` method returns a list of numpy arrays and the `set_weights` method takes a list of numpy arrays as input.

```python
def set_weights_helper(model, parameters):
    params = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict(
        {k: torch.from_numpy(v.copy()) for k, v in params})
    model.load_state_dict(state_dict, strict=True)

def get_weights(self):
    weightsG = [val.cpu().numpy() for _, val in self.netG.state_dict().items()]
    weightsD = [val.cpu().numpy() for _, val in self.netD.state_dict().items()]
    return weightsG + weightsD

def set_weights(self, parameters):
    weightsG = parameters[:self.index]
    weightsD = parameters[self.index:]
    set_weights_helper(self.netG, weightsG)
    set_weights_helper(self.netD, weightsD)
```

```python
def train(self, parameters, config):
    self.fed_round += 1
    self.set_weights(parameters)
    num_epochs = config["num_epochs"] if "num_epochs" in config else 2
    G_losses, D_losses, imgs = train(self.netG, self.netD, self.dataloader, num_epochs)
    fname = "output_train_fr_{}_lid_{}.png".format(self.fed_round, self.learner_id)
    vutils.save_image(imgs, fname, normalize=True)
    weights = self.get_weights()
    metrics = {"G_losses": G_losses, "D_losses": D_losses}
    return weights, metrics, {"num_training_examples": len(self.dataloader.dataset)}
    

def evaluate(self, parameters, config):
    with torch.no_grad():
        self.set_weights(parameters)
        fake = self.netG(cfg.fixed_noise).detach().cpu()
        fake = vutils.make_grid(fake, padding=2, normalize=True)
        fname = "output_eval_fr_{}_lid_{}.png".format(self.fed_round, self.learner_id)
        vutils.save_image(fake, fname, normalize=True)
        return {}
```


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
