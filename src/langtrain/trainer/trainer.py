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
from langtrain.tokenizer.base import Tokenizer
from langtrain.config.config import TrainingConfig
from langtrain.utils import seed_everything
from langtrain.trainer.trainer_utils import callback_fn



class Trainer:
    """
    Trainer Class
    The `Trainer` class is designed to facilitate the training, evaluation, and management of deep learning models. 
    It provides support for distributed training, mixed precision, gradient accumulation, and integration with 
    logging tools like Weights & Biases (wandb). The class is highly configurable and supports various training 
    scenarios, including single-device, multi-device, and distributed setups.
    Attributes:
        model (torch.nn.Module): The model to be trained.
        config (TrainingConfig): Configuration object containing training parameters.
        tokenizer (Tokenizer): Tokenizer used for preprocessing text data.
        collate_fn (callable, optional): Custom collate function for DataLoader.
        model_name (str): Name of the model, used for saving and loading.        
    """

    def __init__(
        self, 
        model: torch.nn.Module, 
        config: TrainingConfig,
        tokenizer: Tokenizer,
        optimizer: Union[torch.optim.Optimizer, None] = None, 
        model_name: str = "my_pretrained_model",
        collate_fn: Union[None, callable] = None) -> None:

        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.collate_fn = collate_fn

        self.device = config.device if isinstance(config.device, torch.device) else torch.device(str(config.device))
        self.scaler = GradScaler(device=self.device)
        self.enable_amp = True if self.device.type == "cuda" else False

        if self.config.overwrite_output_dir and os.path.exists(self.model_name):
            shutil.rmtree(self.model_name)
        os.makedirs(self.model_name, exist_ok=True) 

        if self.config.seed is not None:
            seed_everything(self.config.seed)
            
        assert self.config.gradient_accumulation_steps > 0, "Gradient accumulation steps must be greater than 0"

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
        
        # Initialize optimizer if needed
        if self.optimizer is None:
            self.optimizer = torch.optim.AdamW(
                params=self.model.parameters(), 
                weight_decay=self.config.weight_decay,
                lr=self.config.learning_rate,
                eps=self.config.lr_epsilon
                )
        else:
            self.optimizer = self.optimizer(
                params=self.model.parameters(), 
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                eps=self.config.lr_epsilon
                )

        self.logs = {"epoch": 0, "global_step": 0, "train_loss": None,"val_loss": None,"best_loss": float("inf")}
        self.model.to(self.device)

        if self.config.report_to_wandb:
            wandb.init(project=self.config.wandb_project, name=self.model_name)
            wandb.watch(self.model, log="all", log_graph=False, log_freq=self.config.logging_steps)

        self.save_config()

        if self.config.distributed_backend == "ddp":
            self.model = self._trigger_ddp()
        elif self.config.distributed_backend == "dp":
            self.model = self._trigger_dp()
        elif self.config.distributed_backend is None:
            pass
        else:
            raise ValueError(f"Invalid distributed backend: {self.config.distributed_backend}, must be one of ['ddp', 'dp', None]")

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
        self.device = torch.device(self.device.type, self.local_rank)
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
        if self.config.distributed_backend == "ddp":
            torch.distributed.barrier()

        self.model.train()
        global_step = 0 if self.logs["global_step"] == 0 else self.logs["global_step"]
        accumulated_loss = 0.0  # Track accumulated loss
        start_epoch = self.logs["epoch"] if self.logs["epoch"] > 0 else 0
        patience_counter = 0
        try:
            with tqdm(total=self.config.epochs, desc=f"Training | Step {global_step}") as pbar:
                pbar.update(self.logs["epoch"]) if self.logs["epoch"] > 0 else None
                for epoch in range(start_epoch, self.config.epochs):
                    self.logs["epoch"] = epoch
                    for batch in self.train_data:
                        pbar.set_description(f"Training | Step {global_step}")
                        with autocast(
                            device_type=self.device.type, 
                            dtype=self.precision, 
                            enabled=self.enable_amp
                            ):
                            loss = self.step(self.model, batch, self.device)
                            loss = loss / self.config.gradient_accumulation_steps
                        
                        accumulated_loss += loss.item()
                        self.scaler.scale(loss).backward()

                        if global_step % self.config.validation_steps == 0 and self.val_data is not None and global_step != 0:
                            pbar.set_description(f"Validating")
                            self.evaluate()
                            self.model.train()
                            pbar.set_description(f"Training | Step {global_step}")

                        if global_step % self.config.gradient_accumulation_steps == 0 and global_step != 0:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clipping)
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                            self.optimizer.zero_grad()
                            avg_loss = accumulated_loss / self.config.gradient_accumulation_steps
                            accumulated_loss = 0.0  # Reset accumulated loss

                            if self.config.logging_steps and global_step % self.config.logging_steps == 0 and global_step != 0:
                                self.logs["train_loss"] = avg_loss
                                self.logs["global_step"] = global_step
                                self.logs["epoch"] = epoch
                                pbar.set_postfix(
                                    # step=global_step, 
                                    train_loss=f"{avg_loss:.4f}", 
                                    best_train_loss=f"{self.logs['best_loss']:.4f}",
                                    val_loss=f"{self.logs['val_loss']:.4f}" if self.val_data != None and self.logs['val_loss'] else 'N/A',
                                )
                                if self.config.report_to_wandb:
                                    wandb.log(
                                        data={
                                            "train/train_loss": avg_loss,
                                            "train/best_train_loss": self.logs["best_loss"],
                                            "validation/val_loss": self.logs["val_loss"],
                                            "train/global_step": global_step,
                                            "train/epoch": epoch,
                                            }
                                        )
                                if self.config.early_stopping:
                                    should_stop, patience_counter = callback_fn(
                                        curr_value=avg_loss, 
                                        best_value=self.logs["best_loss"], 
                                        patience_counter=patience_counter, 
                                        patience=self.config.patience
                                        )         
                                    if should_stop:
                                        print(f"Early stopping at epoch {epoch} | step {global_step}")
                                        break

                        if self.config.save_steps and global_step % self.config.save_steps == 0 and global_step != 0:
                            self.save()

                        global_step += 1
                    pbar.update(1)

        except KeyboardInterrupt:
            print("Training interrupted by user.")
        except Exception as e:
            raise e
        finally:
            if self.config.report_to_wandb:
                wandb.finish()
            if self.config.distributed_backend == "ddp":
                torch.distributed.destroy_process_group()

    
    def from_checkpoint(self, checkpoint_path: str) -> None:
        if os.path.exists(checkpoint_path):
            model_file = os.path.join(checkpoint_path, "pytorch_model.pt")
            state_dict = torch.load(model_file)
            # Handle DDP/DP prefix if necessary
            if self.config.distributed_backend == "ddp" or self.config.distributed_backend == "dp":
                state_dict = {f"module.{k}": v for k, v in state_dict.items()}

            self.model.load_state_dict(state_dict)  # Load the model state dict
            self.optimizer.load_state_dict(torch.load(os.path.join(checkpoint_path, "optimizer.pt")))
            self.scaler.load_state_dict(torch.load(os.path.join(checkpoint_path, "scaler.pt")))
            self.logs = torch.load(os.path.join(checkpoint_path, "logs.pt"))
            print(f"Model loaded from {checkpoint_path}")
            self.train()
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
        avg_loss = 0.0
        data_length = 0
        for batch in self.val_data:
            with torch.no_grad():
                with autocast(
                    device_type=self.device.type, 
                    dtype=self.precision, 
                    enabled=self.enable_amp
                    ):
                    loss = self.step(self.model, batch, self.device)
                    avg_loss += loss.item()
                    data_length += 1
        avg_loss /= data_length
        self.logs["val_loss"] = avg_loss



    def _created_dataloader(self, dataset: Union[Any, DataLoader]) -> DataLoader:
        if dataset is None:
            return None
        
        if isinstance(dataset, DataLoader):
            return dataset

        sampler = None
        shuffle = self.config.shuffle_data

        if self.config.distributed_backend == "ddp":
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
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            shuffle=shuffle,
            sampler=sampler,
            collate_fn=self.collate_fn,
            drop_last=True,
        )
        

    def save(self) -> None:
        if self.config.distributed_backend == "ddp" or self.config.distributed_backend == "dp":
            model_state_dict = self.model.module.state_dict()
        else:
            model_state_dict = self.model.state_dict()

        current_loss = self.logs["train_loss"]
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
                if i >= self.config.num_checkpoints:
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
        if self.config.report_to_wandb:
            trainer_config_dict["model"] = model_config_dict
            model_config_dict["architecture"] = str(self.model.__class__.__name__)
            wandb.config.update(trainer_config_dict)
        self.tokenizer.save(path=self.model_name)