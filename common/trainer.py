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
        grad_clip_val = cfg.gradient_clip_val

        local_step = 0
        os.makedirs(cfg.checkpoint_dir, exist_ok=True)
        latest_ckpt = get_latest_checkpoint(cfg.checkpoint_dir)

        should_stop = False
        if cfg.max_epochs > 0 and self.current_epoch >= cfg.max_epochs:
            should_stop = True

        self.prepare_logger()

        chunk_store = getattr(self.dataset.datasets[0], 'store', None)
        # chunk_store = getattr(self.dataset, 'store', None)
        is_chunked_dataset = chunk_store is not None and hasattr(chunk_store, "load_next_chunk")
        total_chunks = len(chunk_store.chunks) if is_chunked_dataset else 1

        if is_chunked_dataset:
            fabric.print(f"Chunked training enabled. Found {total_chunks} chunks.")
        else:
            fabric.print("Standard training mode. Dataset is not chunked or not accessible.")
            logger.warning(f"is_chunked_dataset:{is_chunked_dataset},chunk_store:{chunk_store},self.dataset:{self.dataset}.self.dataset.datasets:{self.dataset.datasets[0].store}")
            is_chunked_dataset=True
            

        progress_bar_total = len(self.dataloader)
        progress = ProgressBar(
            total=progress_bar_total,
            disable=not fabric.is_global_zero,
        )
        assert len(self.dataloader) > 0, "Dataloader is empty"
        
        start_epoch = self.current_epoch

        while not should_stop:
            if "schedulefree" in self.optimizer.__class__.__name__.lower():
                self.optimizer.train()
            for chunk_idx in range(total_chunks):
                need_load = True
                if self.current_epoch == start_epoch and chunk_idx == 0:
                    need_load = False
                
                if is_chunked_dataset and need_load:
                    logger.info(f"\n[Epoch {self.current_epoch}] Switching to Chunk {chunk_idx+1}/{total_chunks}...")
                    t0 = time.time()
                    chunk_store.load_next_chunk() 
                    t1 = time.time()
                    logger.info(f"Chunk loaded in {t1-t0:.2f}s. Images in memory: {len(chunk_store)}")
                    logger.info("Re-initializing dataset batches for the new chunk...")
                    self.dataset.init_batches() 
                    logger.info("Re-initialization complete.")

                    _workers = self.dataloader.num_workers
                    _raw_loader = TorchDataLoader(
                        self.dataset, 
                        batch_size=1,
                        num_workers=_workers,
                        collate_fn=lambda x: x[0], 
                        pin_memory=True,
                    )
                    self.dataloader = self.fabric.setup_dataloaders(_raw_loader)
                    progress_bar_total = len(self.dataloader)
                    progress.total = progress_bar_total
        
                desc = f"Epoch {self.current_epoch}"
                if is_chunked_dataset:
                    desc += f" [C{chunk_idx+1}/{total_chunks}]"
                
                progress.update(desc, 0)

                torch.cuda.empty_cache()
                strategy = self.fabric.strategy
                for batch_idx, batch in enumerate(self.dataloader):
                    is_accumulating = (batch_idx + 1) % grad_accum_steps != 0
                    logger.info(f"source :{batch.get('source_dataset', 'unknown')}, idx:{batch_idx}")
                    source = batch.get('source_dataset', 'unknown')
                    raw_loss = self.model(batch)  # 用于日志，不进行bs缩放
                    bs = batch["pixels"].shape[0]
                    scaled_loss = (raw_loss / bs)*4  # 用于 backward，进行bs缩放来避免不同分辨率样本的梯度不公平

                 # 记录每个 batch 的大小和图像尺寸
                    logger.info(f"Batch {batch_idx+1}/{len(self.dataloader)} | Batch size: {bs} | Image size: {batch['pixels'].shape[1:]}")

                    # 计算本地梯度范数
                    self.fabric.backward(scaled_loss)
                    grad_norm = None
                    logger.warning(f"self.fabric.is_global_zero:{self.fabric.is_global_zero},self.is_deepspeed:{self.is_deepspeed}")
                    if self.fabric.is_global_zero and self.is_deepspeed:
                        # 获取DeepSpeed引擎的全局梯度范数
                        grad_norm = strategy._deepspeed_engine.get_global_grad_norm()
                        

                    
                    # 如果是DeepSpeed并且grad_norm不为None，记录梯度范数
                    if grad_norm is not None and self.fabric.logger:
                        self.fabric.log("train/grad_norm", grad_norm, step=self.global_step)



  


                    stat_str = f"raw_loss: {raw_loss.item():.3f}, scaled_loss: {scaled_loss.item():.6f}"
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)

                    if self.scheduler is not None:
                        self.scheduler.step(self.global_step)

                    if grad_norm is not None:
                        stat_str += f", grad_norm: {grad_norm:.6f}"
                
                    desc = f"Epoch {self.current_epoch} | Batch {batch_idx+1}/{len(self.dataloader)} | {stat_str}"

                    # 记录训练指标
                    progress.update(f"{desc} | {source}", batch_idx + 1)
                    metrics = { "train/loss": raw_loss.item(), f"train/loss_{source}": raw_loss.item(), }
                    if grad_norm is not None:
                        metrics["train/grad_norm"] = grad_norm
                        metrics[f"train/grad_norm_{source}"] = grad_norm
                    if fabric.logger:
                        fabric.log_dict(metrics=metrics, step=self.global_step)
                    self.global_step += 1
                    self.on_post_training_batch()

            progress.close()  # 这里关闭进度条
            logger.info(f"Epoch {self.current_epoch} 完成")
            self.current_epoch += 1
            if cfg.max_epochs > 0 and self.current_epoch >= cfg.max_epochs:
                should_stop = True
            self.on_post_training_batch(is_last=True)


