import safetensors
import torch
import os
import lightning as pl
from omegaconf import OmegaConf
from common.utils import get_class, get_latest_checkpoint, load_torch_file
from common.logging import logger
from modules.sdxl_model import StableDiffusionModel
from modules.scheduler_utils import apply_snr_weight
from lightning.pytorch.utilities.model_summary import ModelSummary
from torch.utils.data import DataLoader as TorchDataLoader
from data.bucket import MultiSourceDataset
def setup(fabric: pl.Fabric, config: OmegaConf):

    logger.info("=" * 80)
    logger.info("Entering setup()")
    logger.info(f"Fabric strategy type: {type(fabric.strategy)}")
    logger.info(f"Fabric strategy repr: {fabric.strategy}")
    logger.info(f"Global rank: {fabric.global_rank}")
    logger.info(f"World size: {fabric.world_size}")
    logger.info("=" * 80)

    logger.info("Building SupervisedFineTune model...")
    model = SupervisedFineTune(
        model_path=config.trainer.model_path,
        config=config,
        device=fabric.device
    )

    logger.info(f"Model class: {model.__class__.__name__}")
    logger.info(f"Inner model class: {model.model.__class__.__name__}")

    ds_cfg = config.dataset
    dataset_name = ds_cfg.get("name", "data.AspectRatioDataset")
    logger.info(f"Dataset name: {dataset_name}")

    if dataset_name.endswith("MultiSourceDataset"):
        logger.info("Using MultiSourceDataset")
        datasets = []
        repeats_source=ds_cfg.get("repeats_source", None)
        shuffle=ds_cfg.get("shuffle", True)

        for i, sub_cfg in enumerate(ds_cfg.datasets):
            cls = get_class(sub_cfg["class"])
            logger.info(f"  Sub-dataset[{i}]: {cls.__name__}")
            ds = cls(
                batch_size=sub_cfg.batch_size,
                img_path=sub_cfg.img_path,
                rank=fabric.global_rank,
                dtype=torch.float32,
                **{k: v for k, v in sub_cfg.items()
                   if k not in ["class", "batch_size", "img_path", "name"]}
            )
            datasets.append(ds)

        dataset = MultiSourceDataset(
            datasets=datasets,
            repeats_source=repeats_source,
            shuffle=shuffle
        )

        dataloader = TorchDataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=lambda x: x[0],
            pin_memory=True,
        )
    else:
        logger.info("Using single dataset")
        dataset_class = get_class(dataset_name)
        dataset = dataset_class(
            batch_size=config.trainer.batch_size,
            rank=fabric.global_rank,
            dtype=torch.float32,
            **ds_cfg,
        )
        dataloader = dataset.init_dataloader()

    logger.info(f"Dataset length: {len(dataset)}")

    # =========================
    # 3. Optimizer
    # =========================
    logger.info("Building optimizer...")
    params_to_optim = [{"params": model.model.parameters()}]

    if config.advanced.get("train_text_encoder_1"):
        lr = config.advanced.get(
            "text_encoder_1_lr",
            config.optimizer.params.lr
        )
        logger.info(f"Train text encoder 1, lr={lr}")
        params_to_optim.append({
            "params": model.conditioner.embedders[0].parameters(),
            "lr": lr
        })

    if config.advanced.get("train_text_encoder_2"):
        lr = config.advanced.get(
            "text_encoder_2_lr",
            config.optimizer.params.lr
        )
        logger.info(f"Train text encoder 2, lr={lr}")
        params_to_optim.append({
            "params": model.conditioner.embedders[1].parameters(),
            "lr": lr
        })

    optimizer = get_class(config.optimizer.name)(
        params_to_optim,
        **config.optimizer.params
    )

    logger.info(f"Optimizer class: {optimizer.__class__.__name__}")

    scheduler = None
    if config.get("scheduler"):
        scheduler = get_class(config.scheduler.name)(
            optimizer,
            **config.scheduler.params
        )
        logger.info(f"Scheduler class: {scheduler.__class__.__name__}")

    # # =========================
    # # 4. Resume（只加载权重）
    # # =========================
    # # if config.trainer.get("resume"):
    # #     latest_ckpt = get_latest_checkpoint(config.trainer.checkpoint_dir)
    # #     logger.info(f"Resume enabled. Latest ckpt: {latest_ckpt}")
    # #     if latest_ckpt:
    # #         sd = load_torch_file(ckpt=latest_ckpt, extract=False)
    # #         model.load_state_dict(sd.get("state_dict", sd))
    # #         meta = sd.get("metadata", {})
    # #         config.global_step = int(meta.get("global_step", 0))
    # #         config.current_epoch = int(meta.get("current_epoch", 0))
    # #         logger.info(
    # #             f"Resumed at epoch={config.current_epoch}, step={config.global_step}"
    # #         )

    # # =========================
    # # 5. Fabric 接管（🔥 最关键）
    # # =========================
    # logger.info("Calling fabric.setup(model, optimizer)...")

    
    # model, optimizer = fabric.setup(model, optimizer)
    # logger.info("fabric.setup(model, optimizer) DONE")
    # logger.info("fabric.setup(model, optimizer) 111111111111111111111111111111111111111111111")

    # dataloader = fabric.setup_dataloaders(dataloader)
    # if hasattr(model, "generate_samples"):
    #     try:
    #         model.mark_forward_method("generate_samples")
    #         logger.info("[OK] generate_samples marked as forward method")
    #     except Exception as e:
    #         logger.warning(f"Failed to mark generate_samples: {e}")
    # logger.info("fabric.setup_dataloaders DONE")

    # # =========================
    # # 6. 分布式类型确认
    # # =========================
    # if hasattr(fabric.strategy, "_deepspeed_engine"):
    #     logger.info(">>> Distributed backend: DeepSpeed (Lightning Fabric)")
    #     engine = fabric.strategy._deepspeed_engine
    #     logger.info(f"DeepSpeed engine type: {type(engine)}")
    #     logger.info(f"ZeRO stage: {engine.zero_optimization_stage()}")
    #     model._distributed_type = "deepspeed"
        
    #     model.get_module = lambda: model
    #     model._deepspeed_engine = fabric.strategy._deepspeed_engine

    # elif hasattr(fabric.strategy, "_fsdp_kwargs"):
    #     logger.info(">>> Distributed backend: FSDP (Lightning Fabric)")
    #     model._distributed_type = "fsdp"

    # else:
    #     logger.info(">>> Distributed backend: DDP / Single GPU")
    #     model._distributed_type = "ddp"

    # model._fabric = fabric

    # logger.info("setup() finished successfully")
    # logger.info("=" * 80)
    # model._fabric_wrapped = fabric

    # return model, dataset, dataloader, optimizer, scheduler

    if config.trainer.get("resume"):
        latest_ckpt = get_latest_checkpoint(config.trainer.checkpoint_dir)
        remainder = {}
        if latest_ckpt:
            logger.info(f"Loading weights from {latest_ckpt}")
            remainder = sd = load_torch_file(ckpt=latest_ckpt, extract=False)
            if latest_ckpt.endswith(".safetensors"):
                remainder = safetensors.safe_open(latest_ckpt, "pt").metadata()
            model.load_state_dict(sd.get("state_dict", sd))
            config.global_step = remainder.get("global_step", 0)
            config.current_epoch = remainder.get("current_epoch", 0)
        
    model.first_stage_model.to(torch.float32)
    if fabric.is_global_zero and os.name != "nt":
        print(f"\n{ModelSummary(model, max_depth=1)}\n")
        
    if hasattr(fabric.strategy, "_deepspeed_engine"):
        model, optimizer = fabric.setup(model, optimizer)
        model.get_module = lambda: model
        model._deepspeed_engine = fabric.strategy._deepspeed_engine
        logger.info(">>> Distributed backend: DeepSpeed (Lightning Fabric)")
        # engine = fabric.strategy._deepspeed_engine
        # logger.info(f"DeepSpeed engine type: {type(engine)}")
        # logger.info(f"ZeRO stage: {engine.zero_optimization_stage()}")
        # model._distributed_type = "deepspeed"
    elif hasattr(fabric.strategy, "_fsdp_kwargs"):
        model, optimizer = fabric.setup(model, optimizer)
        model.get_module = lambda: model
        model._fsdp_engine = fabric.strategy
    else:
        model.model, optimizer = fabric.setup(model.model, optimizer)
        if config.advanced.get("train_text_encoder_1") or config.advanced.get("train_text_encoder_2"):
            model.conditioner = fabric.setup(model.conditioner)

    if hasattr(model, "mark_forward_method"):
        model.mark_forward_method('generate_samples')

    dataloader = fabric.setup_dataloaders(dataloader)
    model._fabric_wrapped = fabric
    return model, dataset, dataloader, optimizer, scheduler





def get_sigmas(sch, timesteps, n_dim=4, dtype=torch.float32, device="cuda:0"):
    sigmas = sch.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = sch.timesteps.to(device)
    timesteps = timesteps.to(device)

    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma

class SupervisedFineTune(StableDiffusionModel):    
    def forward(self, batch):
        
        advanced = self.config.get("advanced", {})
        if not batch["is_latent"]:
            self.first_stage_model.to(self.target_device)
            latents = self.encode_first_stage(batch["pixels"].to(self.first_stage_model.dtype))
            if torch.any(torch.isnan(latents)):
                logger.info("NaN found in latents, replacing with zeros")
                latents = torch.where(torch.isnan(latents), torch.zeros_like(latents), latents)
        else:
            self.first_stage_model.cpu()
            latents = self._normliaze(batch["pixels"])

        cond = self.encode_batch(batch)
        model_dtype = next(self.model.parameters()).dtype
        cond = {k: v.to(model_dtype) for k, v in cond.items()}

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents, dtype=model_dtype)
        if advanced.get("offset_noise"):
            offset = torch.randn(latents.shape[0], latents.shape[1], 1, 1, device=latents.device)
            noise = torch.randn_like(latents) + float(advanced.get("offset_noise_val")) * offset

        bsz = latents.shape[0]

        # Sample a random timestep for each image
        timestep_start = advanced.get("timestep_start", 0)
        timestep_end = advanced.get("timestep_end", 1000)
        timestep_sampler_type = advanced.get("timestep_sampler_type", "uniform")

        # Sample a random timestep for each image
        if timestep_sampler_type == "logit_normal":  
            mu = advanced.get("timestep_sampler_mean", 0)
            sigma = advanced.get("timestep_sampler_std", 1)
            t = torch.sigmoid(mu + sigma * torch.randn(size=(bsz,), device=latents.device))
            timesteps = t * (timestep_end - timestep_start) + timestep_start  # scale to [min_timestep, max_timestep)
            timesteps = timesteps.long()
        else:
            # default impl
            timesteps = torch.randint(
                low=timestep_start, 
                high=timestep_end,
                size=(bsz,),
                dtype=torch.int64,
                device=latents.device,
            )
            timesteps = timesteps.long()
 
        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # Predict the noise residual
        noisy_latents = noisy_latents.to(model_dtype)
        noise_pred = self.model(noisy_latents, timesteps, cond)

        # Get the target for loss depending on the prediction type
        is_v = advanced.get("v_parameterization", False)
        if is_v:
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            target = noise
        
        min_snr_gamma = advanced.get("min_snr", False)            
        if min_snr_gamma:
            # do not mean over batch dimension for snr weight or scale v-pred loss
            # loss = torch.nn.functional.mse_loss(noise_pred.float(), target.float(), reduction="none")
            loss = torch.nn.functional.huber_loss(noise_pred.float(), target.float(), reduction="none", delta=1.0)
            # loss = torch.nn.functional.smooth_l1_loss(noise_pred.float(), target.float(), reduction="none", beta=1.0)
            loss = loss.mean([1, 2, 3])

            if min_snr_gamma:
                loss = apply_snr_weight(loss, timesteps, self.noise_scheduler, advanced.min_snr_val, is_v)
                
            loss = loss.mean()  # mean over batch dimension
        else:
            loss = torch.nn.functional.mse_loss(noise_pred.float(), target.float(), reduction="mean")

        if torch.isnan(loss).any() or torch.isinf(loss).any():
            raise FloatingPointError("Error infinite or NaN loss detected")

        return loss
