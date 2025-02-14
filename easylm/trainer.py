import os
import torch
from tqdm import tqdm
from typing import Union
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset

from easylm.model.bert import BertModel

class Trainer:
    def __init__(self, config: object) -> None:
    
        self.model = config.model
        self.epochs = config.epochs
        self.device = config.device

        self.gradient_accumulation_steps = config.gradient_accumulation_steps
        self.gradient_clipping = config.gradient_clipping
        self.precision = config.precision
        self.validation_steps = config.validation_steps
        self.learning_rate = config.learning_rate
        self.optimizer = config.optimizer
        self.batch_size = config.batch_size
        self.logging_steps = config.logging_steps
        self.validation_steps = config.validation_steps
        self.save_steps = config.save_steps
        self.scaler = GradScaler()
        self.enable_amp = True if torch.cuda.is_available() else False
        # Initialize optimizer if needed
        if self.optimizer is None:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), 
                lr=self.learning_rate if self.learning_rate is not None else 5e-5)
        else:
            self.optimizer = self.optimizer(self.model.parameters(), lr=self.learning_rate)

        self.train_data = self.dataloader(config.train_data)
        self.val_data = self.dataloader(config.val_data)
        self.test_data = self.dataloader(config.test_data)
        
        self.logs = {
            "epoch": 0,
            "global_step": 0,
            "train_loss": [],
            "val_loss": []
        }
        self.model.to(self.device)


    def train(self):
        self.model.train()
        global_step = 1
        for epoch in range(1, self.epochs + 1):
            for batch in tqdm(self.train_data , desc="Training"):
                #
                with autocast(
                    device_type=self.device, 
                    dtype=self.precision, 
                    enabled=self.enable_amp
                    ):
                    loss = self.train_step(self.model, batch, self.device)
                
                self.scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clipping)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                if global_step % self.logging_steps == 0 and global_step != 0:
                    self.logs["train_loss"].append(loss.item())
                    self.logs["global_step"] = global_step
                    self.logs["epoch"] = epoch
                    print(
                        f"Epoch: {epoch}, Global Step: {global_step}, "
                        f"Train Loss: {loss.item():.4f}, "
                        f"Learning Rate: {self.learning_rate:.4f}"
                    )
                
                if (global_step % self.validation_steps == 0 
                    and global_step != 0 
                    and self.val_data is not None):
                    self.evaluate()

                if global_step % self.save_steps == 0 and global_step != 0:
                    os.makedirs("checkpoints", exist_ok=True)
                    if os.path.exists("checkpoints/pytorch_moddel.bin"):
                        os.remove("checkpoints/pytorch_moddel.bin")
                    torch.save(self.model.state_dict(), f"checkpoints/pytorch_moddel.bin")

                global_step += 1


    @staticmethod        
    def train_step(model: torch.nn.Module, batch, device:torch.device):
        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)
        targets = targets.view(-1)
        logits = model(inputs)
        logits = logits.view(-1, logits.size(-1))
        if isinstance(model, BertModel):
            logits = logits[1:]
        loss = F.cross_entropy(logits, targets)
        return loss
    

    @staticmethod
    def validation_step(model: torch.nn.Module, batch, device:torch.device):
        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)
        targets = targets.view(-1)
        logits = model(inputs)
        if isinstance(model, BertModel):
            logits = logits[1:]
        logits = logits.view(-1, logits.size(-1))
        loss = F.cross_entropy(logits, targets)
        return loss


    def evaluate(self):
        self.model.eval()
        for batch in tqdm(self.val_data , desc="Validating"):
            with torch.no_grad():
                with autocast(device_type=self.device, dtype=self.precision, enabled=self.enable_amp):
                    loss = self.validation_step(self.model, batch, self.device)
                    self.logs["val_loss"].append(loss.item())


    def dataloader(self, dataset: Union[Dataset, DataLoader]) -> DataLoader:
        if isinstance(dataset, Dataset):
            return DataLoader(dataset, batch_size=self.batch_size)
        return dataset


__all__ = ["Trainer"]