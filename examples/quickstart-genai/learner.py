import argparse
from collections import OrderedDict

import config as cfg
import torch
import torchvision.utils as vutils
from controller import controller_params
from data import get_dataloader
from model import get_models
from torch.utils.data import DataLoader
from train import train

from metisfl.common.types import ClientParams, ServerParams
from metisfl.learner import Learner, app


def set_weights_helper(model, parameters):
    params = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict(
        {k: torch.from_numpy(v.copy()) for k, v in params})
    model.load_state_dict(state_dict, strict=True)

class TFLearner(Learner):

    """A simple TensorFlow Learner."""

    def __init__(self, dataloader: DataLoader, learner_id: int):
        super().__init__()
        self.dataloader = dataloader
        self.netG, self.netD = get_models()
        self.index = len(self.netG.state_dict())
        self.learner_id = learner_id
        self.fed_round = 0

    def get_weights(self):
        weightsG = [val.cpu().numpy() for _, val in self.netG.state_dict().items()]
        weightsD = [val.cpu().numpy() for _, val in self.netD.state_dict().items()]
        return weightsG + weightsD

    def set_weights(self, parameters):
        weightsG = parameters[:self.index]
        weightsD = parameters[self.index:]
        set_weights_helper(self.netG, weightsG)
        set_weights_helper(self.netD, weightsD)

    def train(self, parameters, config):
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
            return self.get_weights(), {"fake": fake}

def get_learner_server_params(learner_index, max_learners):
    """A helper function to get the server parameters for a learner. """

    ports = list(range(50002, 50002 + max_learners))

    return ServerParams(
        hostname="localhost",
        port=ports[learner_index],
    )


if __name__ == "__main__":
    """The main function. It loads the data, creates a learner, and starts the learner server."""

    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--learner")
    parser.add_argument("-m", "--max-learners", type=int, default=3)

    args = parser.parse_args()
    index = int(args.learner) - 1
    max_learners = args.max_learners

    dataloader = get_dataloader(learner_index=index, num_learners=max_learners)

    # Setup the Learner and the server parameters based on the given index
    learner = TFLearner(dataloader, index)
    server_params = get_learner_server_params(index, max_learners)

    # Setup the client parameters based on the controller parameters
    client_params = ClientParams(
        hostname=controller_params.hostname,
        port=controller_params.port,
        root_certificate=controller_params.root_certificate,
    )
    
    # Start the app
    app(
        learner=learner,
        server_params=server_params,
        client_params=client_params,
    )
