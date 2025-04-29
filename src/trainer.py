import os
import shutil
import yaml
import torch
import wandb
import warnings
from tqdm import tqdm
from typing import Any, Union
from dataclasses import asdict
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from src.config import TrainingConfig
from src.tokenizer import Tokenizer


class Trainer:
    def __init__(
        self, 
        model: torch.nn.Module, 
        config: TrainingConfig,
        tokenizer: Tokenizer,
        optimizer: Union[torch.optim.Optimizer, None] = None, 
        model_name: str = "my_pretrained_model",
        collate_fn: Union[None, callable] = None,
        ) -> None:
        self.config = config
        self.model = model
        self.device = config.device
        self.optimizer = optimizer
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.collate_fn = collate_fn
        # if os.path.exists(self.model_name):
        #     shutil.rmtree(self.model_name)
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
        self.num_checkpoints = config.num_checkpoints
        self.shuffle_train_data = config.shuffle_data
        self.pin_memory = config.pin_memory
        self.scaler = GradScaler(device=self.device)
        self.enable_amp = True if self.device == "cuda" else False
        self.report_to_wandb = config.report_to_wandb
        self.wandb_project = config.wandb_project
        self.distributed_backend = config.distributed_backend

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

        
        self.logs = {"epoch": 0, "global_step": 0, "train_loss": [],"val_loss": [],"best_loss": float("inf")}
        self.model.to(self.device)

        if self.report_to_wandb:
            wandb.init(project=self.wandb_project, name=self.model_name)
            wandb.watch(self.model, log="all", log_graph=False, log_freq=self.logging_steps)

        self.save_config()


        #==================================================================================================
        if self.distributed_backend == "ddp":
            self.model = self._trigger_ddp()
        elif self.distributed_backend == "dp":
            self.model = self._trigger_dp()
        elif self.distributed_backend is None:
            pass
        else:
            raise ValueError(f"Invalid distributed backend: {self.distributed_backend}, must be one of ['ddp', 'dp', None]")

        self.train_data = self._created_dataloader(config.train_data)
        self.val_data = self._created_dataloader(config.val_data)
        self.test_data = self._created_dataloader(config.test_data)
    
    def _trigger_ddp(self) -> torch.nn.Module:
        """
        This method will be trigger for DistributedDataParallel (DDP) training.
        """
        self.rank = int(os.environ["RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])
        self.local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(self.local_rank)
        self.device = torch.device(self.device, self.local_rank)
        torch.distributed.init_process_group(
            backend="nccl",
            init_method="env://"
        )
        self.model = torch.nn.parallel.DistributedDataParallel(
            self.model, 
            device_ids=[self.local_rank], 
            output_device=self.local_rank,
            find_unused_parameters=True
            )
        return self.model
    

    def _trigger_dp(self) -> torch.nn.Module:
        """
        This method will be trigger for DataParallel (DP) training.
        """
        device_count = torch.cuda.device_count()
        self.model = torch.nn.DataParallel(self.model, device_ids=list(range(device_count)))
        return self.model

        

    def train(self) -> None:
        self.model.train()
        global_step = 1 if self.logs["global_step"] == 0 else self.logs["global_step"]
        accumulated_loss = 0.0  # Track accumulated loss
        start_epoch = self.logs["epoch"]
        device_type = self.device.type if isinstance(self.device, torch.device) else self.device

        with tqdm(total=self.epochs, desc=f"Training | Step {global_step}") as pbar:
            pbar.update(self.logs["epoch"]) if self.logs["epoch"] > 0 else None
            for epoch in range(start_epoch, self.epochs + 1):
                self.logs["epoch"] = epoch
                for batch in self.train_data:
                    pbar.set_description(f"Training | Step {global_step}")
                    with autocast(
                        device_type=device_type, 
                        dtype=self.precision, 
                        enabled=self.enable_amp
                        ):
                        loss = self.step(self.model, batch, self.device)
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
                                train_loss=f"{avg_loss:.4f}", 
                                best_train_loss=f"{self.logs['best_loss']:.4f}",
                                val_loss=f"{self.logs['val_loss'][-1]:.4f}" if self.val_data is not None else 'N/A',
                            )

                            if self.report_to_wandb:
                                wandb.log(
                                    data={
                                        "train/train_loss": avg_loss,
                                        "train/best_train_loss": self.logs["best_loss"],
                                        "validation/val_loss": self.logs["val_loss"][-1] if self.val_data is not None else None,
                                        "train/global_step": global_step,
                                        "train/epoch": epoch,
                                        },
                                    step=global_step
                                )

                        # Save model
                        if self.save_steps is not None and global_step % self.save_steps == 0:
                            self.save()

                    global_step += 1

                    # Validation
                    if global_step % self.validation_steps == 0 and self.val_data is not None:
                        pbar.set_description(f"Validating")
                        self.evaluate()
                        self.model.train()
                        pbar.set_description(f"Training | Step {global_step}")
                  
                pbar.update(1)

        if self.report_to_wandb:
            wandb.finish()
        if self.distributed_backend == "ddp":
            torch.distributed.destroy_process_group()

    
    def from_checkpoint(self, checkpoint_path: str) -> None:
        if os.path.exists(checkpoint_path):
            model_file = os.path.join(checkpoint_path, "pytorch_model.pt")
            state_dict = torch.load(model_file)

            # Handle DDP/DP prefix if necessary
            if self.distributed_backend == "ddp" or self.distributed_backend == "dp":
                state_dict = {f"module.{k}": v for k, v in state_dict.items()}

            self.model.load_state_dict(state_dict)  # Load the model state dict
            self.optimizer.load_state_dict(torch.load(os.path.join(checkpoint_path, "optimizer.pt")))
            self.scaler.load_state_dict(torch.load(os.path.join(checkpoint_path, "scaler.pt")))
            self.logs = torch.load(os.path.join(checkpoint_path, "logs.pt"))

            print(f"Model loaded from {checkpoint_path}")
        else:
            raise ValueError(f"Checkpoint path {checkpoint_path} does not exist")


    @staticmethod        
    def step(model: torch.nn.Module, batch, device: Union[torch.device, str]) ->torch.FloatTensor:
        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs, targets)
        loss = outputs.loss
        return loss


    def evaluate(self) -> None:
        self.model.eval()
        for batch in self.val_data:
            with torch.no_grad():
                with autocast(
                    device_type=self.device, 
                    dtype=self.precision, 
                    enabled=self.enable_amp
                    ):
                    loss = self.step(self.model, batch, self.device)
                    self.logs["val_loss"].append(loss.item())


    def _created_dataloader(self, dataset: Union[Any, DataLoader]) -> DataLoader:
        if dataset is None:
            return None
        
        if isinstance(dataset, DataLoader):
            return dataset

        sampler = None
        shuffle = self.shuffle_train_data

        if self.distributed_backend == "ddp":
            if not torch.distributed.is_initialized():
                raise RuntimeError("Distributed training is not initialized.")
            
            if not hasattr(dataset, "__len__"):
                warnings.warn(
                    f"Dataset {dataset.__class__.__name__} does not have __len__ attribute. Skipping `DistributedSampler`."
                )
            else:
                sampler = torch.utils.data.distributed.DistributedSampler(dataset)
                shuffle = False  # Shuffling is handled by the sampler


        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=shuffle,
            sampler=sampler,
            collate_fn=self.collate_fn,
            drop_last=True,
        )
        

    def save(self) -> None:
        if self.distributed_backend == "ddp" or self.distributed_backend == "dp":
            model_state_dict = self.model.module.state_dict()
        else:
            model_state_dict = self.model.state_dict()

        current_loss = self.logs["train_loss"][-1]
        if current_loss < self.logs["best_loss"]:
            self.logs["best_loss"] = current_loss
            torch.save(model_state_dict, f"{self.model_name}/pytorch_model.pt")  

            # Save all snapshots in the checkpoint directory
            ckpt_dir = os.path.join(self.model_name, f"checkpoint-{self.logs['global_step']}")
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(model_state_dict, os.path.join(ckpt_dir, "pytorch_model.pt"))
            torch.save(self.optimizer.state_dict(), os.path.join(ckpt_dir, "optimizer.pt"))
            torch.save(self.scaler.state_dict(), os.path.join(ckpt_dir, "scaler.pt"))
            torch.save(self.logs, os.path.join(ckpt_dir, "logs.pt"))

            # Keep n latest checkpoints
            checkpoints = sorted(
            [d for d in os.listdir(self.model_name) if d.startswith("checkpoint-") and "-" in d],
            key=lambda x: int(x.split("-")[1]),
            reverse=True
            )
            for i, checkpoint in enumerate(checkpoints):
                if i >= self.num_checkpoints:
                    shutil.rmtree(os.path.join(self.model_name, checkpoint))


    def save_config(self) -> None:
        # Save config in a yaml file
        trainer_config_dict = {}
        for key, value in asdict(self.config).items():
            if value is not None and key != "train_data" and key != "val_data" and key != "test_data":
                trainer_config_dict[key] = value
            elif key == "model":
                trainer_config_dict[key] = str(self.model.__class__.__name__)
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
        if self.report_to_wandb:
            trainer_config_dict["model"] = model_config_dict
            model_config_dict["architecture"] = str(self.model.__class__.__name__)
            wandb.config.update(trainer_config_dict)
        self.tokenizer.save(path=self.model_name)