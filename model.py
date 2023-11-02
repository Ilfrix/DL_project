import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import torch.optim as optim
from torchmetrics.functional import accuracy
import wandb
  
class LitModel(L.LightningModule):
  def __init__(self, channels, width, height, num_classes, hidden_size=64, learning_rate=1e-3):
    super().__init__()

    self.channels = channels
    self.width = width
    self.height = height
    self.num_classes = num_classes
    self.hidden_size = hidden_size
    self.learning_rate = learning_rate

    self.model = nn.Sequential(
        nn.Conv2d(3, 6, 5),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(6, 16, 5),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Flatten(),
        nn.Linear(16 * 5 * 5, 120),
        nn.ReLU(),
        nn.Linear(120, 84),
        nn.ReLU(),
        nn.Linear(84, 10)
    )

  def forward(self, x):
    x = self.model(x)
    return F.log_softmax(x, dim=1)

  def training_step(self, batch):
    x, y = batch
    logits = self(x)
    loss = F.cross_entropy(logits, y)
    return loss

  def validation_step(self, batch, batch_idx):
    x, y = batch
    logits = self(x)
    loss = F.nll_loss(logits, y)
    preds = torch.argmax(logits, dim=1)
    acc = accuracy(preds, y, task="multiclass", num_classes=self.num_classes)
    
    wandb.log({"accuracy": acc, "loss": loss})
    self.log("val_loss", loss, prog_bar=True)
    self.log("val_acc", acc, prog_bar=True)

  def configure_optimizers(self):
    optimizer = optim.Adam(self.parameters(), lr = self.learning_rate)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9, verbose=True)
    return (
      {'optimizer': optimizer, 'lr_scheduler': {'scheduler':scheduler, 'lr': 'val_loss'}}
    )