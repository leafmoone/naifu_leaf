import os
import time
import logging
import torch
import lightning as pl
from tqdm import tqdm
from torch.utils.data import DataLoader as TorchDataLoader
from common.logging import logger
from common.utils import *
from common.logging import logger
from omegaconf import OmegaConf
from pathlib import Path
from lightning.fabric.strategies import DeepSpeedStrategy
import safetensors.torch
os.environ['DEEPSPEED_DEBUG'] = '1'

os.environ['NCCL_DEBUG'] = 'WARN'
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'OFF' 

logging.getLogger("lightning.fabric").setLevel(logging.WARNING)
logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)

logging.basicConfig(level=logging.WARNING)

# 计算本地梯度范数:无用
def local_grad_norm(model):
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().norm(2)
            total += param_norm.item() ** 2
    return total ** 0.5

class Trainer:
    def __init__(self, fabric: pl.Fabric, config: OmegaConf):
        """
        Initialize the trainer with the given fabric and configuration.
        Args:
            fabric (pl.Fabric): The PyTorch Lightning Fabric instance.
            config (OmegaConf): The configuration object.
        """
        self.fabric = fabric
        logger.info(f"self.fabric:{self.fabric}")
        model_cls = get_class(config.target)
        model, dataset, dataloader, optimizer, scheduler = model_cls(fabric, config)

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dataset = dataset  # 这个 dataset 是 RatioDataset 实例
        self.dataloader = dataloader
        self.global_step = int(config.get("global_step", 0))
        self.current_epoch = int(config.get("current_epoch", 0))
        
        strategy_path = self.model.config.lightning.get("strategy", "")
        self.is_deepspeed = isinstance(self.fabric.strategy, DeepSpeedStrategy)
        logger.info(f" self.is_deepspeed:{self.is_deepspeed}")
        logger.info(f"self.fabric:{self.fabric}")

    def prepare_logger(self):
        """Prepare the logger and log hyperparameters if the logger is not CSVLogger."""
        fabric = self.fabric
        if fabric.logger and fabric.logger.__class__.__name__ != "CSVLogger":
            config = OmegaConf.to_container(self.model.config, resolve=True)
            fabric.logger.log_hyperparams(config)

    def on_post_training_batch(self, is_last=False):
        """Perform actions after each training batch."""
        if self.fabric.logger and not is_last:
            self.log_lr_values()
        self.perform_sampling(is_last=is_last)
        self.save_model(is_last=is_last)
        self.eval_model(is_last=is_last)

    def log_lr_values(self):
        optimizer_name = self.model.config.optimizer.name
        last_lr = [group.get("lr", 0) for group in self.optimizer.param_groups]
        ocls = self.optimizer.__class__.__name__
        for i, lr in enumerate(last_lr):
            self.fabric.log(f"lr/{ocls}-{i}", lr, step=self.global_step)
        is_da = optimizer_name.startswith("DAdapt")
        is_prodigy = optimizer_name.startswith("prodigyopt")
        if not (is_da or is_prodigy):
            return
        last_d_lr = [(g["d"] * g["lr"]) for g in self.optimizer.param_groups]
        for i, lr in enumerate(last_d_lr):
            self.fabric.log(f"d*lr/{ocls}-{i}", lr, step=self.global_step)

    def eval_model(self, is_last: bool = False):
        config = self.model.config
        cfg = config.trainer
        eval_st = cfg.get("eval_steps", -1)
        eval_fq = cfg.get("eval_epochs", -1)
        is_eval_step = eval_st > 0 and self.global_step % eval_st == 0
        is_eval_epoch = eval_fq > 0 and self.current_epoch % eval_fq == 0
        should_eval = (is_last and is_eval_epoch) or is_eval_step
        has_eval_method = hasattr(self.model, "eval_model")
        if not should_eval or not has_eval_method:
            return
        if "schedulefree" in self.optimizer.__class__.__name__.lower():
            self.optimizer.eval()
        self.model.eval_model(logger=self.fabric.logger, current_epoch=self.current_epoch, global_step=self.global_step)
        torch.cuda.empty_cache()
        if "schedulefree" in self.optimizer.__class__.__name__.lower():
            self.optimizer.train()

    def save_model(self, is_last: bool = False):
        cfg = self.model.config.trainer
        ckpt_dir = cfg.checkpoint_dir
        ckpt_st = cfg.checkpoint_steps
        ckpt_fq = cfg.checkpoint_freq
        save_weights_only = cfg.get("save_weights_only", False)
        is_ckpt_step = ckpt_st > 0 and self.global_step % ckpt_st == 0
        is_ckpt_epoch = ckpt_fq > 0 and self.current_epoch % ckpt_fq == 0
        postfix = f"e{self.current_epoch}_s{self.global_step}"
        # Check whether it's time to save the checkpoint
        if not ((is_last and is_ckpt_epoch) or is_ckpt_step):
            return
        logger.info("Saving model checkpoint")
        metadata = {"global_step": str(self.global_step), "current_epoch": str(self.current_epoch)}
        path = os.path.join(ckpt_dir, f"checkpoint-{postfix}")
        self.model.save_checkpoint(path, metadata)
        # If we are saving model weights only
        if save_weights_only:
            logger.info("Saving model weights only")
            return
        logger.info(f"Saving train state to {path}")
        # Otherwise, save the full checkpoint with model, optimizer, and metadata
        strategy = self.fabric.strategy
        if hasattr(strategy, "_deepspeed_engine"):
            logger.info("Saving DeepSpeed checkpoint")
            strategy._deepspeed_engine.save_checkpoint(path, client_state=metadata)
        elif hasattr(strategy, "_fsdp_kwargs"):
            logger.info("Saving FSDP checkpoint")
            self.fabric.save(path, {"model": self.model, "optimizer": self.optimizer, "metadata": metadata})
        else:
            logger.info("Saving DDP/single-device checkpoint")
            self.fabric.save(path, {"state_dict": self.model.state_dict(), "optimizer": self.optimizer.state_dict(), "metadata": metadata})

    def perform_sampling(self, is_last: bool = False):
        config = self.model.config
        enabled_sampling = config.sampling.enabled and hasattr(self.model, "generate_samples")
        sampling_cfg = config.sampling
        sampling_steps = sampling_cfg.every_n_steps
        sample_by_step = sampling_steps > 0 and self.global_step % sampling_steps == 0
        sampling_epochs = sampling_cfg.every_n_epochs
        sample_by_epoch = sampling_epochs > 0 and self.current_epoch % sampling_epochs == 0
        sample_on_start = config.sampling.get("sample_on_start", False) and not getattr(self, "sampler_initialized", False)
        if not enabled_sampling or len(sampling_cfg.prompts) == 0:
            return
        if sampling_cfg.get("save_dir", None):
            os.makedirs(sampling_cfg.save_dir, exist_ok=True)
        if (is_last and sample_by_epoch) or sample_by_step or sample_on_start:
            setattr(self, "sampler_initialized", True)
            if "schedulefree" in self.optimizer.__class__.__name__.lower():
                self.optimizer.eval()
            torch.cuda.empty_cache()
            rng_state = torch.get_rng_state()
            cuda_rng_state = torch.cuda.get_rng_state()
            self.model.generate_samples(logger=self.fabric.logger, current_epoch=self.current_epoch, global_step=self.global_step)
            torch.cuda.empty_cache()
            torch.set_rng_state(rng_state)
            torch.cuda.set_rng_state(cuda_rng_state)
            if "schedulefree" in self.optimizer.__class__.__name__.lower():
                self.optimizer.train()

    def train_loop(self):
        config = self.model.config
        fabric = self.fabric
        cfg = config.trainer
    
        logger.info(f"Config trainer: {cfg}")
        logger.info("into train_loop")
    
        grad_accum_steps = cfg.accumulate_grad_batches
    
        os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    
        should_stop = False
        if cfg.max_epochs > 0 and self.current_epoch >= cfg.max_epochs:
            should_stop = True
    
        self.prepare_logger()
    
        # === 收集每个 dataset 的 chunk store ===
        datasets = self.dataset.datasets
        chunk_stores = [getattr(ds, "store", None) for ds in datasets]
    
        def is_chunked(store):
            return store is not None and hasattr(store, "load_next_chunk")
    
        def has_next_chunk(store):
            return (
                is_chunked(store)
                and store.current_chunk_idx < len(store.chunks) - 1
            )
    
        progress = ProgressBar(
            total=len(self.dataloader),
            disable=not fabric.is_global_zero,
        )
    
        # ===epoch loop===
        while not should_stop:
            if "schedulefree" in self.optimizer.__class__.__name__.lower():
                self.optimizer.train()
    
            # ====== chunk 状态机 ======
            first_chunk = True
    
            while True:
                # ---- 是否需要切 chunk ----
                if not first_chunk:
                    if not any(has_next_chunk(store) for store in chunk_stores):
                        break  # 所有 dataset 都没 chunk 了
    
                    logger.info(f"[Epoch {self.current_epoch}] Switching chunk(s)...")
    
                    # 只推进「还能推进的 dataset」
                    for ds, store in zip(datasets, chunk_stores):
                        if has_next_chunk(store):
                            store.load_next_chunk()
                            ds.init_batches()
    
                    # 所有 dataset 切完 chunk 后，再统一 reset
                    self.dataset.start_epoch()
    
                    # 重建 dataloader
                    _raw_loader = TorchDataLoader(
                        self.dataset,
                        batch_size=1,
                        num_workers=self.dataloader.num_workers,
                        collate_fn=lambda x: x[0],
                        pin_memory=True,
                    )
                    self.dataloader = fabric.setup_dataloaders(_raw_loader)
                    progress.set_total(len(self.dataloader))

                    logger.info(
                        "[ChunkState] " +
                        ", ".join(
                            f"{ds.name}:{store.current_chunk_idx + 1}/{len(store.chunks)}"
                            for ds, store in zip(datasets, chunk_stores)
                            if store is not None
                        )
                    )

    
                first_chunk = False
    
                # ====== 训练当前 chunk 状态 ======
                epoch_desc = f"Epoch {self.current_epoch}"
                progress.update(epoch_desc, 0)
                chunk_desc = " | ".join(
                    f"{ds.name}:{store.current_chunk_idx + 1}/{len(store.chunks)}"
                    for ds, store in zip(datasets, chunk_stores)
                    if store is not None
                )
                

    
                for batch_idx, batch in enumerate(self.dataloader):
                    source = batch.get("source_dataset", "unknown")
    
                    raw_loss = self.model(batch)
                    bs = batch["pixels"].shape[0]
                    scaled_loss = (raw_loss / bs) * 4
    
                    fabric.backward(scaled_loss)
    
                    grad_norm = None
                    if fabric.is_global_zero and self.is_deepspeed:
                        grad_norm = fabric.strategy._deepspeed_engine.get_global_grad_norm()
    
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
    
                    if self.scheduler is not None:
                        self.scheduler.step(self.global_step)
    
                    stat = f"loss={raw_loss.item():.4f}"
                    if grad_norm is not None:
                        stat += f", grad_norm={grad_norm:.4f}"
    
                    progress.update(
                        f"{epoch_desc} |{chunk_desc}| {stat} | {source}",
                        batch_idx + 1,
                    )

                    metrics = {
                        "train/loss": raw_loss.item(),
                        f"train/loss_{source}": raw_loss.item(),
                    }
                    if grad_norm is not None:
                        metrics["train/grad_norm"] = grad_norm
                        metrics[f"train/grad_norm_{source}"] = grad_norm
    
                    if fabric.logger:
                        fabric.log_dict(metrics, step=self.global_step)
    
                    self.global_step += 1
                    self.on_post_training_batch()
                    # progress.set_total(len(self.dataloader))
    
            # ====== epoch 结束 ======
            progress.close()
            logger.info(f"Epoch {self.current_epoch} finished")
    
            self.current_epoch += 1
            if cfg.max_epochs > 0 and self.current_epoch >= cfg.max_epochs:
                should_stop = True
    
            self.on_post_training_batch(is_last=True)
    

