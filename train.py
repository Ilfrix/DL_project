from .data import MyDataModule
# import model
from .model import LitModel
import lightning as L

def train() -> None:
    dm = MyDataModule()
    # Init model from datamodule's attributes
    model = LitModel(*dm.dims, dm.num_classes, hidden_size=256)
    trainer = L.Trainer(
        max_epochs=50,
        accelerator="auto",
        devices=1,
    )
    trainer.fit(model, dm)