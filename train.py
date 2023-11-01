from .data import MyDataModule
# import model
from .model import LitModel
import lightning as L

def train():
    dm = MyDataModule()
    # Init model from datamodule's attributes
    model = LitModel(*dm.dims, dm.num_classes, hidden_size=256)
    trainer = L.Trainer(
        max_epochs=10,
        accelerator="auto",
        devices=1,
    )
    trainer.fit(model, dm)