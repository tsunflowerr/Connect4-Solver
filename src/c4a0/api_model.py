import copy
from sys import platform
from typing import Optional
import torch
import pytorch_lightning as pl
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from pytorch_lightning.trainer.trainer import Trainer

app = FastAPI()

class BestModelCheckpointConfig(BaseModel):
    monitor: str = "val_loss"
    mode: str = "min"

def get_torch_device() -> torch.device:
    """Tries to use cuda or mps, if available, otherwise falls back to cpu."""
    if torch.cuda.is_available():
        return torch.device("cuda")

    if platform == "darwin":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif not torch.backends.mps.is_built():
            raise RuntimeError(
                "MPS unavailable because the current torch install was not built with MPS enabled."
            )
        else:
            raise RuntimeError(
                "MPS unavailable because the current MacOS version is not 12.3+ and/or you do not "
                "have an MPS-enabled device on this machine."
            )

    return torch.device("cpu")

class BestModelCheckpoint(pl.callbacks.Callback):
    def __init__(self, monitor: str = "val_loss", mode: str = "min") -> None:
        self.monitor = monitor
        self.mode = mode
        self.best_model: Optional[pl.LightningModule] = None
        self.best_score = float("inf") if mode == "min" else float("-inf")

    def on_validation_end(self, trainer: Trainer, pl_module: pl.LightningModule) -> None:
        current_score = trainer.callback_metrics.get(self.monitor)
        if current_score is None:
            return

        if isinstance(current_score, torch.Tensor):
            current_score = current_score.item()

        if self.mode == "min":
            if current_score < self.best_score:
                self.best_score = current_score
                self.best_model = copy.deepcopy(pl_module)
        elif self.mode == "max":
            if current_score > self.best_score:
                self.best_score = current_score
                self.best_model = copy.deepcopy(pl_module)

    def get_best_model(self) -> pl.LightningModule:
        assert self.best_model is not None, "no model checkpoint called"
        return self.best_model

@app.get("/device")
def get_device():
    try:
        device = get_torch_device()
        return {"device": str(device)}
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/create_best_model_checkpoint")
def create_best_model_checkpoint(config: BestModelCheckpointConfig):
    try:
        callback = BestModelCheckpoint(monitor=config.monitor, mode=config.mode)
        return {"message": f"BestModelCheckpoint created with monitor: {config.monitor}, mode: {config.mode}"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/get_best_model")
def get_best_model():

    callback = BestModelCheckpoint()  
    try:
        best_model = callback.get_best_model()
        return {"message": "Best model retrieved", "best_model": str(best_model)}
    except AssertionError:
        raise HTTPException(status_code=404, detail="No best model found")
