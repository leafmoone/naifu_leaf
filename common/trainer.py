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
from data.bucket import DatasetExhausted

os.environ['DEEPSPEED_DEBUG'] = '1'

os.environ['NCCL_DEBUG'] = 'WARN'
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'OFF' 

logging.getLogger("lightning.fabric").setLevel(logging.WARNING)
logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)

logging.basicConfig(level=logging.WARNING)



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
        # if hasattr(strategy, "_deepspeed_engine"):
        #     logger.info("Saving DeepSpeed checkpoint")
        #     strategy._deepspeed_engine.save_checkpoint(path, client_state=metadata)
        if hasattr(strategy, "_deepspeed_engine"):
            engine = strategy._deepspeed_engine
    
            logger.info("Saving DeepSpeed checkpoint via engine.save_checkpoint")
            engine.save_checkpoint(
                path,
                client_state=metadata
            )
            return

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
    
        logger.info("=" * 80)
        logger.info("Entering train_loop (Scheduler-based)")
        logger.info(f"max_epochs={cfg.max_epochs}, max_steps={cfg.max_steps}")
        logger.info("=" * 80)
    
        os.makedirs(cfg.checkpoint_dir, exist_ok=True)
        self.prepare_logger()
    
        progress = ProgressBar(
            total=1000000,
            disable=not fabric.is_global_zero,
        )
    
        # ============================
        # Dataset / Chunk info
        # ============================
        datasets = self.dataset.datasets
        chunk_stores = [getattr(ds, "store", None) for ds in datasets]
    
        def has_next_chunk(store):
            return (
                store is not None
                and store.current_chunk_idx < len(store.chunks) - 1
            )
    
        logger.info(
            f"[TrainLoop] start at epoch={self.current_epoch}, "
            f"global_step={self.global_step}"
        )
    
        if fabric.is_global_zero and self.is_deepspeed:
            engine = fabric.strategy._deepspeed_engine
            # logger.info(f"[DS] zero_stage = {engine.zero_optimization_stage()}")
            # logger.info(f"[DS] gradient_clipping(method) = {engine.gradient_clipping}")
            logger.info(
                f"[DS] ds_config.gradient_clipping = "
                f"{engine.config.get('gradient_clipping', None)}"
            )
    
        # ============================
        # Epoch loop
        # ============================
        num_sources = len(datasets)
        alive = self.dataset.alive

        while True:
            if cfg.max_steps > 0 and self.global_step >= cfg.max_steps:
                logger.info("[TrainLoop] max_steps reached, stopping")
                break
    
            if cfg.max_epochs > 0 and self.current_epoch >= cfg.max_epochs:
                logger.info("[TrainLoop] max_epochs reached, stopping")
                break
    
            dataloader_iter = iter(self.dataloader)
    
            while True:
                try:
                    batch = next(dataloader_iter)

    
                except DatasetExhausted as e:
                    src = e.source_idx
                    ds = datasets[src]
                    store = chunk_stores[src]
    
                    logger.warning(
                        f"[ChunkExhausted] source={getattr(ds,'name',src)} "
                        f"chunk={store.current_chunk_idx+1}/{len(store.chunks)}"
                    )
    
                    if has_next_chunk(store):
                        # logger.info(
                        #     f"[ChunkSwitch] source={getattr(ds,'name',src)} "
                        #     f"→ load next chunk {store.current_chunk_idx+2}/{len(store.chunks)}"
                        # )
    
                        store.load_next_chunk()
                        ds.init_batches()
    
                        # 重建 dataloader & iterator
                        raw_loader = TorchDataLoader(
                            self.dataset,
                            batch_size=None,
                            num_workers=self.dataloader.num_workers,
                            pin_memory=False,
                        )
                        self.dataloader = fabric.setup_dataloaders(raw_loader)
                        dataloader_iter = iter(self.dataloader)
    
                        continue  # 继续训练
    
                    else:
                        self.dataset.alive[src] = False
                        self.dataset.mark_dead(src)

                        # self.dataset.alive[src] = False

                        logger.info(
                            f"[SourceFinished] source={ds.name} fully exhausted,alive:{alive}"
                        )
                        if not any(self.dataset.alive):
                            logger.info("[EpochFinished] all sources exhausted")
                            self.current_epoch += 1
                            
                            # 检查是否还要继续
                            if cfg.max_epochs > 0 and self.current_epoch >= cfg.max_epochs:
                                logger.info("[TrainLoop] max_epochs reached, stopping")
                                logger.info(
                                    f"[TrainLoop] finished at epoch={self.current_epoch}, "
                                    f"global_step={self.global_step}"
                                )
                                self.on_post_training_batch(is_last=True)


                                
                                break
                            
                            # reset 所有 dataset / chunk
                            for i, ds in enumerate(datasets):
                                store = chunk_stores[i]
                                if store is not None:
                                    logger.info(
                                        f"[EpochReset] reset source={ds.name} "
                                        f"chunk {store.current_chunk_idx+1} → 1"
                                    )
                                    store.current_chunk_idx = 0
                                    ds.init_batches()
                            
                            # reset scheduler alive 状态


                            
                            self.on_post_training_batch(is_last=True)



                            
                            self.dataset.reset_alive()
                            
                            # 重建 dataloader
                            raw_loader = TorchDataLoader(
                                self.dataset,
                                batch_size=None,
                                num_workers=self.dataloader.num_workers,
                                pin_memory=False,
                            )
                            self.dataloader = fabric.setup_dataloaders(raw_loader)
                            
                            logger.info(
                                f"[EpochStart] epoch={self.current_epoch} started"
                            )



                            
                            break
                
                        raw_loader = TorchDataLoader(
                            self.dataset,
                            batch_size=None,
                            num_workers=self.dataloader.num_workers,
                            pin_memory=False,
                        )
                        self.dataloader = fabric.setup_dataloaders(raw_loader)
                        dataloader_iter = iter(self.dataloader)
                        continue
    
                except StopIteration:
                    logger.info("[Epoch] dataloader exhausted")
                    break
    
                source = batch["source_dataset"]
                source_idx = batch.get("_source_idx", "n/a")
                chunk_i = batch.get("_chunk_idx")
                chunk_n = batch.get("_chunk_total")
    
                # logger.info(
                #     f"[Step {self.global_step}] "
                #     f"batch from source={source} (idx={source_idx}) "
                #     f"Train step | source={source} | chunk={chunk_i}/{chunk_n}"
                # )
    
                raw_loss = self.model(batch)
                bs = batch["pixels"].shape[0]
                scaled_loss = (raw_loss / bs)*4
                logger.info(f"batch:{batch["pixels"].shape},bs:{bs}")
                fabric.backward(scaled_loss)
    
                grad_norm = None
                if fabric.is_global_zero and self.is_deepspeed:
                    grad_norm = (
                        fabric.strategy
                        ._deepspeed_engine
                        .get_global_grad_norm()
                    )
                    # logger.warning(
                    #     "grad_norm = fabric.strategy._deepspeed_engine.get_global_grad_norm()"
                    # )
    
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
    
                if self.scheduler is not None:
                    self.scheduler.step(self.global_step)
    
                stat = f"raw_loss={raw_loss.item():.6f}"
                if grad_norm is not None:
                    stat += f", grad_norm={grad_norm:.6f}"
                # else:
                #     logger.warning(
                #         f"fabric.is_global_zero:{fabric.is_global_zero} "
                #         f"and self.is_deepspeed:{self.is_deepspeed}"
                #     )
    
                desc = (
                    f"Epoch {self.current_epoch} | "
                    f"step {self.global_step} | "
                    f"{stat} | source: {source} | chunk: {chunk_i}/{chunk_n}"
                )
    
                progress.update(desc, self.global_step + 1)
    
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
    

    
        logger.info(
            f"[TrainLoop] finished at epoch={self.current_epoch}, "
            f"global_step={self.global_step}"
        )
        self.on_post_training_batch(is_last=True)

