from .data import MyDataModule
# import model
from .model import LitModel
import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor

def train() -> None:
    dm = MyDataModule()
    # Init model from datamodule's attributes
    model = LitModel(*dm.dims, dm.num_classes, hidden_size=256)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = L.Trainer(
        max_epochs=10,
        accelerator="auto",
        devices=1,
        callbacks=[lr_monitor],
    )
    trainer.fit(model, dm)