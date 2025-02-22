import os
import yaml
import torch
from tqdm import tqdm
from typing import Union
import torch.nn.functional as F
from torch.utils.data import Dataset
from dataclasses import asdict
from easylm.model.bert import BertModel
from torch.amp import autocast, GradScaler
#from easylm.data.dataloader import DataLoader
from torch.utils.data import DataLoader
from easylm.config import TrainingConfig
from easylm.model.gpt import GPTModel
from easylm.model.llama import LlamaModel
from easylm.tokenizer import Tokenizer


class Trainer:
    def __init__(
            self, 
            model: torch.nn.Module, 
            config: TrainingConfig,
            tokenizer: Tokenizer,
            optimizer: Union[torch.optim.Optimizer, None] = None, 
            model_name: str = "my_pretrained_model"
    ) -> None:
        self.config = config
        self.model = model
        self.device = config.device
        self.optimizer = optimizer
        self.tokenizer = tokenizer
        self.model_name = model_name
        if os.path.exists(self.model_name):
            raise ValueError(f"Model name '{self.model_name}' already exists in the current directory.")
        os.makedirs(self.model_name, exist_ok=True)
        #===============================================

        self.epochs = config.epochs
        self.gradient_accumulation_steps = config.gradient_accumulation_steps
        self.gradient_clipping = config.gradient_clipping
        self.validation_steps = config.validation_steps
        self.learning_rate = config.learning_rate
        self.weight_decay = config.weight_decay
        self.lr_epsilon = config.lr_epsilon
        self.batch_size = config.batch_size
        self.logging_steps = config.logging_steps
        self.validation_steps = config.validation_steps
        self.save_steps = config.save_steps
        self.num_workers = config.num_workers
        self.seed = config.seed
        self.shuffle_train_data = config.shuffle_train_data
        self.pin_memory = config.pin_memory
        self.scaler = GradScaler(device=self.device)
        self.enable_amp = True if torch.cuda.is_available() else False

        torch.manual_seed(self.seed) if self.seed is not None else None
        torch.cuda.manual_seed(self.seed) if self.seed is not None else None
        assert self.gradient_accumulation_steps > 0, "Gradient accumulation steps must be greater than 0"

        #=======================================Precision==================================================
        if config.precision == "fp16":
            self.precision = torch.float16
        elif config.precision == "bf16":
            self.precision = torch.bfloat16
        elif config.precision == "fp32":
            self.precision = torch.float32
        else:
            raise ValueError(f"Invalid precision: {config.precision}, must be one of ['fp16', 'bf16', 'fp32']")
        
        if self.precision == torch.bfloat16 and torch.cuda.is_bf16_supported() is False:
            raise ValueError("Your device does not support bf16")
        #==================================================================================================
        

        # Initialize optimizer if needed
        if self.optimizer is None:
            self.optimizer = torch.optim.AdamW(
                params=self.model.parameters(), 
                weight_decay=self.weight_decay,
                lr=self.learning_rate,
                eps=self.lr_epsilon
                )
        else:
            self.optimizer = self.optimizer(
                params=self.model.parameters(), 
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                eps=self.lr_epsilon
                )

        self.train_data = self.dataloader(config.train_data)
        self.val_data = self.dataloader(config.val_data)
        self.test_data = self.dataloader(config.test_data)

        
        self.logs = {
            "epoch": 0,
            "global_step": 0,
            "train_loss": [],
            "val_loss": [],
            "best_loss": float("inf"),
        }
        self.model.to(self.device)
        self.save_config()


    def train(self):
        self.model.train()
        global_step = 1
        accumulated_loss = 0.0  # Track accumulated loss
        
        with tqdm(total=self.epochs * len(self.train_data), desc="Training") as pbar:
            for epoch in range(1, self.epochs + 1):
                for batch in self.train_data:
                    with autocast(
                        device_type=self.device, 
                        dtype=self.precision, 
                        enabled=self.enable_amp
                        ):
                        loss = self.train_step(self.model, batch, self.device)
                        loss = loss / self.gradient_accumulation_steps
                    
                    accumulated_loss += loss.item()
                    self.scaler.scale(loss).backward()

                    # Perform optimization step only when accumulated steps reach the limit
                    if (global_step % self.gradient_accumulation_steps == 0) or (global_step == 1):
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clipping)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad()

                        # Log loss and update tqdm bar
                        avg_loss = accumulated_loss / self.gradient_accumulation_steps
                        accumulated_loss = 0.0  # Reset loss accumulator
                        
                        if self.logging_steps is not None and global_step % self.logging_steps == 0:
                            self.logs["train_loss"].append(avg_loss)
                            self.logs["global_step"] = global_step
                            self.logs["epoch"] = epoch
                            pbar.set_postfix(
                                step=global_step, 
                                loss=f"{avg_loss:.4f}", 
                                best_loss=f"{self.logs['best_loss']:.4f}"
                                )

                        # Save model
                        if self.save_steps is not None and global_step % self.save_steps == 0:
                            self.save()

                    global_step += 1

                    # Validation
                    if global_step % self.validation_steps == 0 and self.val_data is not None:
                        self.evaluate()
                        self.model.train()
                  
                pbar.update(1)

    @staticmethod        
    def train_step(
        model: torch.nn.Module, 
        batch, 
        device: Union[torch.device, str] = torch.device
        ) ->torch.FloatTensor:
        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs, targets)
        loss = outputs.loss
        return loss
    

    @staticmethod
    def validation_step(
        model: torch.nn.Module, 
        batch, 
        device: Union[torch.device, str] = torch.device
        ) ->torch.FloatTensor:
        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs)
        return outputs.loss


    def evaluate(self) -> None:
        self.model.eval()
        for batch in tqdm(self.val_data , desc="Validating"):
            with torch.no_grad():
                with autocast(
                    device_type=self.device, 
                    dtype=self.precision, 
                    enabled=self.enable_amp
                    ):
                    loss = self.validation_step(self.model, batch, self.device)
                    self.logs["val_loss"].append(loss.item())


    def dataloader(self, dataset: Union[Dataset, DataLoader]) -> DataLoader:
        if isinstance(dataset, Dataset):
            return DataLoader(
                dataset=dataset, 
                batch_size=self.batch_size, 
                num_workers=self.num_workers, 
                shuffle=self.shuffle_train_data,
                pin_memory=self.pin_memory
            )
        return dataset

    def save(self):
        current_loss = self.logs["train_loss"][-1]  # Get latest loss
        if current_loss < self.logs["best_loss"]:
            self.logs["best_loss"] = current_loss  # Update best loss
            torch.save(self.model.state_dict(), f"{self.model_name}/pytorch_model.pt")  


    def save_config(self):
        # Save config in a yaml file
        trainer_config_dict = {}
        for key, value in asdict(self.config).items():
            if value is not None and key != "train_data" and key != "val_data" and key != "test_data":
                trainer_config_dict[key] = value
            elif value is None:
                trainer_config_dict[key] = "None"
        
        with open(f"{self.model_name}/trainer_config.yaml", "w") as f:
            yaml.dump(trainer_config_dict, f)

        model_config_dict = {}
        for key, value in asdict(self.model.config).items():
            if value is not None:
                model_config_dict[key] = value
            elif value is None:
                model_config_dict[key] = "None"

        with open(f"{self.model_name}/model_config.yaml", "w") as f:
            yaml.dump(model_config_dict, f)

        self.tokenizer.save(path=self.model_name)