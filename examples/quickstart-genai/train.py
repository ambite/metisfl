from typing import List, Tuple

import config as cfg
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from torch.utils.data import DataLoader

# Initialize the ``BCELoss`` function
criterion = nn.BCELoss()


def train(
    netG: nn.Module,
    netD: nn.Module,
    dataloader: DataLoader,
    num_epochs: int = 5
) -> Tuple[List[float], List[float]]:
    
    optimizerD = optim.Adam(netD.parameters(), lr=cfg.lr,
                            betas=(cfg.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=cfg.lr,
                            betas=(cfg.beta1, 0.999))

    img_list = []
    G_losses = []
    iters = 0
    D_losses = []

    print("Starting Training Loop...")
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):

            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            netD.zero_grad()
            real_cpu = data[0].to(cfg.device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), cfg.real_label,
                               dtype=torch.float, device=cfg.device)
            output = netD(real_cpu).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            noise = torch.randn(b_size, cfg.nz, 1, 1, device=cfg.device)
            fake = netG(noise)
            label.fill_(cfg.fake_label)
            output = netD(fake.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            # (2) Update G network: maximize log(D(G(z)))
            netG.zero_grad()
            label.fill_(cfg.real_label)
            output = netD(fake).view(-1)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, num_epochs, i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = netG(cfg.fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(
                    fake, padding=2, normalize=True))

            iters += 1

    return G_losses, D_losses