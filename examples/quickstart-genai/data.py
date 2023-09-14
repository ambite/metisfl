from typing import List
import torch
import torch.nn.parallel
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import glob 
import config as cfg
from metisfl.common.utils import iid_partition_dir


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