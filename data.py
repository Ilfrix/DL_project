import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import lightning as L

class MyDataModule(L.LightningDataModule):
  def __init__(self):
      super().__init__()
      self.BATCH_SIZE = 100
      self.data_dir = "data_dir"
      self.transform = transforms.Compose(
          [
              transforms.ToTensor(),
              transforms.Normalize((0.1307,), (0.3081,)),
          ]
      )
      self.train_data = None
      self.test_data = None
      self.val_data = None
      self.dims = (1, 28, 28)
      self.num_classes = 10

  def prepare_data(self):
    torchvision.datasets.CIFAR10(self.data_dir, train=True, download=True)
    torchvision.datasets.CIFAR10(self.data_dir, train=False, download=True)


  def setup(self, stage=None):
    if stage == 'fit' or stage is None:
      data_full = torchvision.datasets.CIFAR10(self.data_dir, train=True, transform=self.transform)
      self.train_data , self.val_data = random_split(data_full, [45000, 5000])

    if stage == "test" or stage is None:
      self.test_data = torchvision.datasets.CIFAR10(self.data_dir, train=False, transform=self.transform)


  def train_dataloader(self):
    return DataLoader(self.train_data, batch_size=self.BATCH_SIZE)

  def test_dataloader(self):
    return DataLoader(self.test_data, batch_size=self.BATCH_SIZE)

  def val_dataloader(self):
    return DataLoader(self.val_data, batch_size=self.BATCH_SIZE)
