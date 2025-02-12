# import torch
# from tqdm import tqdm
# from typing import Union
# import torch.nn.functional as F
# from torch.amp import autocast, GradScaler

# class Trainer:
#     def __init__(self, config: object) -> None:
    
#         self.model = config.model
#         self.epochs = config.epochs
#         self.device = config.device
#         self.train_dataloader = config.train_dataloader
#         self.val_dataloader = config.val_dataloader
#         self.test_dataloader = config.test_dataloader
#         self.gradient_accumulation_steps = config.gradient_accumulation_steps
#         self.gradient_clipping = config.gradient_clipping
#         self.precision = config.precision
#         self.validation_steps = config.validation_steps
#         self.learning_rate = config.learning_rate
#         self.scaler = GradScaler()
#         self.model.to(self.device)
#         self.enable_amp = True if torch.cuda.is_available() else False
#         self.optimizer = (
#             torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate) 
#             if config.optimizer is None 
#             else config.optimizer(self.model.parameters(), lr=self.learning_rate)
#         )
        
#         self.logs = {
#             "epoch": None,
#             "global_step": None,
#             "train_loss": [],
#             "val_loss": []
#         }

#     def train(self):
#         self.model.train()
#         global_step = 0
#         for epoch in range(self.epochs):
#             for batch in tqdm(self.train_dataloader , desc="Training"):
#                 with autocast(device_type=self.device, dtype=self.precision, enabled=self.enable_amp):
#                     loss = self.train_step(self.model, batch)
#                 self.scaler.scale(loss).backward()
#                 torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clipping)
#                 self.scaler.step(self.optimizer)
#                 self.scaler.update()
#                 self.optimizer.zero_grad()

#                 if global_step % self.validation_steps == 0:
#                     self.logs["train_loss"].append(loss.item())
#                     self.logs["global_step"] = global_step
#                     self.logs["epoch"] = epoch
#                     if self.val_dataloader is not None:
#                         self.evaluate()
#                     self.model.train()
#                 global_step += 1


#     @staticmethod        
#     def train_step(model: torch.nn.Module, batch):
#         inputs, targets = batch
#         targets = targets.view(-1)
#         output = model(inputs)
#         output = output.view(-1, output.size(-1))
#         loss = F.cross_entropy(output, targets)
#         return loss
    

#     @staticmethod
#     def validation_step(model: torch.nn.Module, batch):
#         inputs, targets = batch
#         targets = targets.view(-1)
#         output = model(inputs)
#         output = output.view(-1, output.size(-1))
#         loss = F.cross_entropy(output, targets)
#         return loss


#     def evaluate(self):
#         self.model.eval()
#         for batch in tqdm(self.val_dataloader , desc="Validating"):
#             with torch.no_grad():
#                 with autocast(device_type=self.device, dtype=self.precision, enabled=self.enable_amp):
#                     loss = self.validation_step(self.model, batch)
#                     self.logs["val_loss"].append(loss.item())



# __all__ = ["Trainer"]


import torch
from tqdm import tqdm
from typing import Union
import torch.nn.functional as F
from torch.amp import autocast, GradScaler

class Trainer:
    def __init__(self, config: object) -> None:
    
        self.model = config.model
        self.epochs = config.epochs
        self.device = config.device
        self.train_dataloader = config.train_dataloader
        self.val_dataloader = config.val_dataloader
        self.test_dataloader = config.test_dataloader
        self.gradient_accumulation_steps = config.gradient_accumulation_steps
        self.gradient_clipping = config.gradient_clipping
        self.precision = config.precision
        self.validation_steps = config.validation_steps
        self.learning_rate = config.learning_rate
        self.optimizer = config.optimizer
        self.enable_amp = torch.cuda.is_available()
        self.scaler = GradScaler(enabled=self.enable_amp)
        self.model.to(self.device)

        # Initialize optimizer if needed
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), 
                lr=self.learning_rate if self.learning_rate is not None else 1e-3)
        else:
            self.optimizer = self.optimizer(self.model.parameters(), lr=self.learning_rate)
        
        self.logs = {
            "epoch": None,
            "global_step": None,
            "train_loss": [],
            "val_loss": []
        }

    def train(self):
        self.model.train()
        global_step = 0
        # Loop over epochs
        for epoch in range(self.epochs):
            for i, batch in enumerate(tqdm(self.train_dataloader, desc=f"Training Epoch {epoch+1}")):
                # Use autocast only if AMP is enabled
                with autocast(device_type=self.device, dtype=self.precision, enabled=self.enable_amp):
                    loss = self.train_step(self.model, batch)
                # Divide loss by gradient_accumulation_steps to average the gradients
                loss = loss / self.gradient_accumulation_steps
                self.scaler.scale(loss).backward()
                
                # Perform optimizer step when accumulation is complete
                if (i + 1) % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clipping)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    global_step += 1

                    # Log and evaluate periodically
                    if global_step % self.validation_steps == 0:
                        self.logs["train_loss"].append(loss.item())
                        self.logs["global_step"] = global_step
                        self.logs["epoch"] = epoch
                        if self.val_dataloader is not None:
                            self.evaluate()
                            print(
                                f"Epoch: {self.logs['epoch']} | Global Step: {self.logs['global_step']} | "
                                f"Train Loss: {self.logs['train_loss'][-1]:.4f} | Val Loss: {self.logs['val_loss'][-1]:.4f}"
                            )
                        self.model.train()

    @staticmethod        
    def train_step(model: torch.nn.Module, batch):
        inputs, targets = batch
        targets = targets.view(-1)
        output = model(inputs)
        output = output.view(-1, output.size(-1))
        loss = F.cross_entropy(output, targets, reduction='mean')
        if torch.isnan(loss):
            print("Warning: Loss is NaN. Check your inputs and model output.")
        return loss
    
    @staticmethod
    def validation_step(model: torch.nn.Module, batch):
        inputs, targets = batch
        targets = targets.view(-1)
        output = model(inputs)
        output = output.view(-1, output.size(-1))
        loss = F.cross_entropy(output, targets)
        return loss

    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validating"):
                with autocast(device_type=self.device, dtype=self.precision, enabled=self.enable_amp):
                    loss = self.validation_step(self.model, batch)
                    self.logs["val_loss"].append(loss.item())
        self.model.train()


__all__ = ["Trainer"]