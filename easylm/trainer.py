from dataclasses import asdict
import os
import torch
from tqdm import tqdm
from typing import Union
import torch.nn.functional as F
from torch.utils.data import Dataset
import yaml
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
            pretrained_path: str = "pretrained_model"
    ) -> None:
        self.config = config
        self.model = model
        self.device = config.device
        self.optimizer = optimizer
        self.tokenizer = tokenizer
        self.pretrained_path = pretrained_path
        os.makedirs(self.pretrained_path, exist_ok=True)
        #===============================================

        self.epochs = config.epochs
        self.gradient_accumulation_steps = config.gradient_accumulation_steps
        self.gradient_clipping = config.gradient_clipping
        self.validation_steps = config.validation_steps
        self.learning_rate = config.learning_rate
        self.batch_size = config.batch_size
        self.logging_steps = config.logging_steps
        self.validation_steps = config.validation_steps
        self.save_steps = config.save_steps
        self.num_workers = config.num_workers
        self.shuffle_train_data = config.shuffle_train_data
        self.pin_memory = config.pin_memory
        self.scaler = GradScaler(device=self.device)
        self.enable_amp = True if torch.cuda.is_available() else False

        #===============================Precision==========================================================
        if config.precision == "fp16":
            self.precision = torch.float16
        elif config.precision == "bfp16":
            self.precision = torch.bfloat16
        elif config.precision == "fp32":
            self.precision = torch.float32
        else:
            raise ValueError(f"Invalid precision: {config.precision}, must be one of ['fp16', 'bfp16', 'fp32']")
        
        if self.precision == torch.bfloat16 and torch.cuda.is_bf16_supported() is False:
            raise ValueError("Your device does not support bf16")
        #==================================================================================================
        

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
        self.save_config()


    def train(self):
        self.model.train()
        global_step = 1
        for epoch in range(1, self.epochs + 1):
            for batch in tqdm(self.train_data, desc="Training"):
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

                if self.logging_steps is not None and global_step % self.logging_steps == 0 and global_step != 0:
                    self.logs["train_loss"].append(loss.item())
                    self.logs["global_step"] = global_step
                    self.logs["epoch"] = epoch
                    print(
                        f"Epoch: {epoch}, Global Step: {global_step}, "
                        f"Train Loss: {loss.item():.4f}, "
                    )
                
                if (global_step % self.validation_steps == 0 
                    and global_step != 0 
                    and self.val_data is not None
                    ):
                    self.evaluate()
                    self.model.train()

                if global_step % self.save_steps == 0 and global_step != 0:
                    self.save()

                global_step += 1


    @staticmethod        
    def train_step(
        model: torch.nn.Module, 
        batch, 
        device: Union[torch.device, str] = torch.device):
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
    def validation_step(
        model: torch.nn.Module, 
        batch, 
        device: Union[torch.device, str] = torch.device):
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
            return DataLoader(
                dataset=dataset, 
                batch_size=self.batch_size, 
                num_workers=self.num_workers, 
                shuffle=self.shuffle_train_data,
                pin_memory=self.pin_memory
            )
        return dataset

    def save(self):
        torch.save(self.model.state_dict(), f"{self.pretrained_path}/pytorch_model.bin")


    def predict(self, text: Union[str, None], max_seq_len: Union[int, None] = None) -> str:
        if self.do_predict is True:
            raise ValueError(f"text cannot be None when do_predict is True")
        inputs = self.tokenizer.encode(text)
        if isinstance(self.model, LlamaModel, GPTModel):
            assert max_seq_len is not None, "max_seq_len cannot be None when using LlamaModel or GPTModel"
            result = self.model.generate(inputs, self.tokenizer.eos_token_id, max_seq_len)
        elif isinstance(self.model, BertModel):
            result = self.model.fill_mask(inputs)
        return self.tokenizer.decode(result)    


    def save_config(self):
        # Save config in a yaml file
        trainer_config_dict = {}
        for key, value in asdict(self.config).items():
            if value is not None and key != "train_data" and key != "val_data" and key != "test_data":
                trainer_config_dict[key] = value
            elif value is None:
                trainer_config_dict[key] = "None"
        
        with open(f"{self.pretrained_path}/trainer_config.yaml", "w") as f:
            yaml.dump(trainer_config_dict, f)

        model_config_dict = {}
        for key, value in asdict(self.model.config).items():
            if value is not None:
                model_config_dict[key] = value
            elif value is None:
                model_config_dict[key] = "None"

        with open(f"{self.pretrained_path}/model_config.yaml", "w") as f:
            yaml.dump(model_config_dict, f)

        self.tokenizer.save(path=self.pretrained_path)