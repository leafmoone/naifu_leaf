import functools
import math
import numpy as np
import random
import torch
from collections import defaultdict

from pathlib import Path
from torch.utils.data import Dataset, get_worker_info
from data.image_storage import DirectoryImageStore, Entry, LatentStore,ChunkedDirectoryImageStore
from torchvision.transforms import Resize, InterpolationMode
from common.logging import logger
from common.utils import get_class

image_suffix = set([".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif", ".webp"])


def is_latent_folder(path: Path):
    # iterate over all files in the folder and find if any of them is a latent
    for p in path.iterdir():
        if p.is_dir():
            continue
        if p.suffix == ".h5":
            return True

def worker_init_fn(worker_id):
    worker_info = get_worker_info()
    dataset: RatioDataset = worker_info.dataset  # type: ignore
    # random.seed(worker_info.seed)  # type: ignore
    dataset.init_batches()


class RatioDataset(Dataset):
    def __init__(
        self,
        batch_size: int,
        img_path: Path | str | list,
        ucg: int = 0,
        rank: int = 0,
        dtype=torch.float16,
        seed: int = 42,
        use_central_crop=True,
        load_into_memory: bool = False,
        name=None,#用于日志记录
        **kwargs,
    ):
        kwargs['load_into_memory'] = load_into_memory
        self.rng = np.random.default_rng(seed)
        self.batch_size = batch_size
        self.num_workers = kwargs.get("num_workers", 24)
        self.use_central_crop = use_central_crop
        self.name = name or Path(img_path).name

        root_path = Path(img_path)
        assert root_path.exists(), f"Path {root_path} does not exist."


        if kwargs.get("store_cls"):
            store_class = get_class(kwargs["store_cls"])
        
        elif is_latent_folder(root_path):
            store_class = LatentStore
            
        elif load_into_memory:
            store_class = ChunkedDirectoryImageStore
            logger.info(f" Strategy: Using ChunkedDirectoryImageStore for {root_path}")
            
        else:
            store_class = DirectoryImageStore

        self.store = store_class(
            root_path,
            rank=rank,
            ucg=ucg,
            dtype=dtype,
            **kwargs,
        )
    
    def generate_buckets(self):
        raise NotImplementedError

    def assign_buckets(self):
        raise NotImplementedError

    def init_batches(self):
        self.assign_buckets()
        self.assign_batches()

    def init_dataloader(self, **kwargs):
        dataloader = torch.utils.data.DataLoader(
            self,
            sampler=None,
            batch_size=None,
            num_workers=self.num_workers,
            worker_init_fn=worker_init_fn,
            shuffle=True,
            pin_memory=True,
            **kwargs,
        )
        return dataloader

    def __len__(self):
        return len(self.batch_idxs)
    
    @staticmethod
    @functools.cache
    def fit_dimensions(target_ratio, min_h, min_w):
        min_area = min_h * min_w
        h = max(min_h, math.ceil(math.sqrt(min_area * target_ratio)))
        w = max(min_w, math.ceil(h / target_ratio))

        if w < min_w:
            w = min_w
            h = max(min_h, math.ceil(w * target_ratio))

        while h * w < min_area:
            increment = 8
            if target_ratio >= 1:
                h += increment
            else:
                w += increment

            w = max(min_w, math.ceil(h / target_ratio))
            h = max(min_h, math.ceil(w * target_ratio))
        return int(h), int(w)

    def assign_batches(self):
        self.batch_idxs = []
        for bucket in self.bucket_content:
            if not bucket or len(bucket) == 0:
                continue
            reminder = len(bucket) % self.batch_size
            bucket = np.array(bucket)
            self.rng.shuffle(bucket)
            if not reminder:
                self.batch_idxs.extend(bucket.reshape(-1, self.batch_size))
            else:
                self.batch_idxs.extend(bucket[:-reminder].reshape(-1, self.batch_size))
                self.batch_idxs.append(bucket[-reminder:])

        self.rng.shuffle(self.batch_idxs)

    def __getitem__(self, idx):
        img_idxs = self.batch_idxs[idx]
        return self.store.get_batch(img_idxs)
    
    
class AspectRatioDataset(RatioDataset):
    """Original implementation of AspectRatioDataset, equal to other frameworks"""
    def __init__(
        self, 
        batch_size: int, 
        img_path: Path | str | list, 
        ucg: int = 0, rank: int = 0, 
        dtype=torch.float16, 
        target_area: int = 1024 * 1024, 
        min_size: int = 512, 
        max_size: int = 2048, 
        divisible: int = 64, 
        seed: int = 42, 
        **kwargs
    ):
        super().__init__(batch_size, img_path, ucg, rank, dtype, seed, **kwargs)
        self.target_area = target_area
        self.max_size, self.min_size, self.divisible = max_size, min_size, divisible
        self.store.crop = self.crop

        self.generate_buckets()
        self.init_batches()
    
    def crop(self, entry: Entry, i: int) -> Entry:
        assert self.to_ratio is not None, "to_ratio is not initialized"
        H, W = entry.pixel.shape[-2:]
        base_ratio = H / W
        target_ratio = self.to_ratio[i]
        h, w = self.ratio_to_bucket[target_ratio]
        if not entry.is_latent:
            resize_h, resize_w = self.fit_dimensions(base_ratio, h, w)
            interp = InterpolationMode.BILINEAR if resize_h < H else InterpolationMode.BICUBIC
            entry.pixel = Resize(
                size=(resize_h, resize_w), 
                interpolation=interp, 
                antialias=None
            )(entry.pixel)
        else:
            h, w = h // 8, w // 8

        H, W = entry.pixel.shape[-2:]
        if self.use_central_crop:
            dh, dw = (H - h) // 2, (W - w) // 2
        else:
            assert H >= h and W >= w, f"{H}<{h} or {W}<{w}"
            dh, dw = random.randint(0, H - h), random.randint(0, W - w)

        entry.pixel = entry.pixel[:, dh : dh + h, dw : dw + w]
        return entry, dh, dw

    def generate_buckets(self):
        assert (
            self.target_area % 4096 == 0
        ), "target area (h * w) must be divisible by 64"
        width = np.arange(self.min_size, self.max_size + 1, self.divisible)
        height = np.minimum(
            self.max_size,
            ((self.target_area // width) // self.divisible) * self.divisible,
        )
        valid_mask = height >= self.min_size

        resos = set(zip(width[valid_mask], height[valid_mask]))
        resos.update(zip(height[valid_mask], width[valid_mask]))
        resos.add(((int(np.sqrt(self.target_area)) // self.divisible) * self.divisible,) * 2)
        self.buckets_sizes = np.array(sorted(resos))
        self.bucket_ratios = self.buckets_sizes[:, 0] / self.buckets_sizes[:, 1]
        self.ratio_to_bucket = {ratio: hw for ratio, hw in zip(self.bucket_ratios, self.buckets_sizes)}

    def assign_buckets(self):
        img_res = np.array(self.store.raw_res)
        img_ratios = img_res[:, 0] / img_res[:, 1]
        self.bucket_content = [[] for _ in range(len(self.buckets_sizes))]
        self.to_ratio = {}

        # Assign images to buckets
        for idx, img_ratio in enumerate(img_ratios):
            diff = np.abs(self.bucket_ratios - img_ratio)
            bucket_idx = np.argmin(diff)
            self.bucket_content[bucket_idx].append(idx)
            self.to_ratio[idx] = self.bucket_ratios[bucket_idx]


class AdaptiveSizeDataset(RatioDataset):
    """AdaptiveRatioDataset, a modified version of AspectRatioDataset which avoid resize from smaller images"""
    def __init__(
        self, 
        batch_size: int, 
        img_path: Path | str | list, 
        ucg: int = 0, rank: int = 0, 
        dtype=torch.float16, 
        target_area: int = 1024 * 1024, 
        divisible: int = 64, 
        seed: int = 42, 
        **kwargs
    ):
        super().__init__(batch_size, img_path, ucg, rank, dtype, seed, **kwargs)
        self.store.crop = self.crop
        self.target_area = target_area
        self.divisible = divisible

        self.generate_buckets()
        self.init_batches()
    
    def crop(self, entry: Entry, i: int) -> Entry:
        assert self.to_size is not None, "to_ratio is not initialized"
        H, W = entry.pixel.shape[-2:]
        h, w = self.to_size[i]
        bucket_width = w - w % self.divisible
        bucket_height = h - h % self.divisible
        
        if not entry.is_latent:
            resize_h, resize_w = h, w
            entry.pixel = Resize(
                size=(resize_h, resize_w), 
                interpolation=InterpolationMode.BILINEAR, 
                antialias=None
            )(entry.pixel)
        else:
            h, w = bucket_height // 8, bucket_width // 8

        H, W = entry.pixel.shape[-2:]
        if self.use_central_crop:
            dh, dw = (H - h) // 2, (W - w) // 2
        else:
            assert H >= h and W >= w, f"{H}<{h} or {W}<{w}"
            dh, dw = random.randint(0, H - h), random.randint(0, W - w)

        entry.pixel = entry.pixel[:, dh : dh + h, dw : dw + w]
        return entry, dh, dw

    def generate_buckets(self):
        pass
    
    def assign_buckets(self):
        img_res = np.array(self.store.raw_res)
        self.to_size = {}
        self.bucket_content = defaultdict(list)

        # Assign images to buckets
        for idx, (img_width, img_height) in enumerate(img_res):
            img_area = img_width * img_height

            # Check if the image needs to be resized (i.e., only allow downsizing)
            if img_area > self.target_area:
                scale_factor = math.sqrt(self.target_area / img_area)
                img_width = math.floor(img_width * scale_factor / self.divisible) * self.divisible
                img_height = math.floor(img_height * scale_factor / self.divisible) * self.divisible

            bucket_width = img_width - img_width % self.divisible
            bucket_height = img_height - img_height % self.divisible
            reso = (bucket_width, bucket_height)
            self.bucket_content[reso].append(idx)
            self.to_size[idx] = (bucket_width, bucket_height)

        self.bucket_content = [v for k, v in self.bucket_content.items()]



# class MultiSourceDataset(Dataset):
#     def __init__(self, datasets, config=None, shuffle=True):
#         """
#         datasets: List[RatioDataset]
#         config: Configuration object containing `repeats_source`
#         repeats_source:  List[int], same length as datasets
#                           default = [1, 1, ..., 1]
#         """
#         # 从配置文件读取 repeats_source，如果没有则使用默认值
#         self.repeats_source = config.get("repeats_source", [1] * len(datasets))
#         logger.warning(f"repeats_source: {self.repeats_source}")
        
#         self.datasets = datasets
#         self._log(f"repeats_source: {self.repeats_source}")
#         assert len(self.datasets) == len(self.repeats_source)

#         self.shuffle = shuffle

#         self.epoch_plan = None
#         self.ptrs = None

#         self.start_epoch()

#     def _log(self, msg):
#         print(msg)

#     def start_epoch(self):
#         plan = []
        
#         for ds_idx, (ds, rep) in enumerate(zip(self.datasets, self.repeats_source)):
#             self._log(f"Preparing dataset {ds_idx} (source: {ds.name}), Repeat: {rep}")
#             for _ in range(rep):
#                 plan.extend([ds_idx] * len(ds)) 

#         if self.shuffle:
#             rng = torch.randperm(len(plan))
#             plan = [plan[i] for i in rng.tolist()]

#         self.epoch_plan = plan
#         self.ptrs = [0 for _ in self.datasets]

#     def __len__(self):
#         return len(self.epoch_plan)

#     def __getitem__(self, idx):
#         ds_idx = self.epoch_plan[idx]  # Get the dataset index from the epoch plan
#         ds = self.datasets[ds_idx]  # Get the actual dataset

#         ptr = self.ptrs[ds_idx]  # Get the pointer for the dataset
#         if ptr >= len(ds):
#             raise RuntimeError(
#                 f"Dataset {ds.name} exhausted early. "
#                 f"ptr={ptr}, len={len(ds)}"
#             )

#         batch = ds[ptr]  # Get the batch from the dataset
#         self.ptrs[ds_idx] += 1  # Increment the pointer for this dataset

#         batch["_source_idx"] = ds_idx  # Add the source index to the batch
#         batch["source_dataset"] = ds.name  # Add the source dataset name to the batch

#         return batch

class MultiSourceDataset(Dataset):

    def __init__(self, datasets, repeats_source=None, shuffle=True):
        """
        datasets: List[RatioDataset]
        repeats_source:  List[int], same length as datasets
                          default = [1, 1, ..., 1]
        """
        logger.warning(f"repeats_source:{repeats_source}")
        self.datasets = datasets
        self.repeats_source = repeats_source or [1] * len(datasets)
        self._log(f"repeats_source: {self.repeats_source}")
        assert len(self.datasets) == len(self.repeats_source)

        self.shuffle = shuffle

        self.epoch_plan = None
        self.ptrs = None

        self.start_epoch()

    def _log(self, msg):
        print(msg)

    def start_epoch(self):
        plan = []
        
        for ds_idx, (ds, rep) in enumerate(zip(self.datasets, self.repeats_source)):
            self._log(f"Preparing dataset {ds_idx} (source: {ds.name}), Repeat: {rep}")
            for _ in range(rep):
                plan.extend([ds_idx] * len(ds)) 

        if self.shuffle:
            rng = torch.randperm(len(plan))
            plan = [plan[i] for i in rng.tolist()]

        self.epoch_plan = plan
        self.ptrs = [0 for _ in self.datasets]

    def __len__(self):
        return len(self.epoch_plan)

    def __getitem__(self, idx):
        ds_idx = self.epoch_plan[idx]  # Get the dataset index from the epoch plan
        ds = self.datasets[ds_idx]  # Get the actual dataset

        ptr = self.ptrs[ds_idx]  # Get the pointer for the dataset
        if ptr >= len(ds):
            raise RuntimeError(
                f"Dataset {ds.name} exhausted early. "
                f"ptr={ptr}, len={len(ds)}"
            )

        batch = ds[ptr]  # Get the batch from the dataset
        self.ptrs[ds_idx] += 1  # Increment the pointer for this dataset

        batch["_source_idx"] = ds_idx  # Add the source index to the batch
        batch["source_dataset"] = ds.name  # Add the source dataset name to the batch

        return batch
