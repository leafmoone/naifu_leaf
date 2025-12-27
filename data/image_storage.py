import os
import hashlib
import json
import h5py as h5
import numpy as np
import torch
import torch.distributed as dist
from tqdm.auto import tqdm
from dataclasses import dataclass
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from typing import Callable, Generator, Optional  # type: ignore
from torchvision import transforms
from common.logging import logger
import imagesize
json_lib = json
try:
    import rapidjson as json_lib
except ImportError:
    pass
import shutil
import tempfile
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm
import psutil
import threading
import re
import logging
from PIL import Image

image_suffix = set([".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif", ".webp"])

IMAGE_TRANSFORMS = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)

REPEAT_RE = re.compile(r"^(\d+)_")

def extract_repeat_key(path: Path):
    for p in path.parents:
        m = REPEAT_RE.match(p.name)
        if m:
            return int(m.group(1)), p.name
    return 1, "default"


def get_class(name: str):
    import importlib

    module_name, class_name = name.rsplit(".", 1)
    module = importlib.import_module(module_name, package=None)
    return getattr(module, class_name)


def is_img(path: Path):
    return path.suffix in image_suffix


def sha1sum(txt):
    return hashlib.sha1(txt.encode()).hexdigest()


@dataclass
class Entry:
    is_latent: bool
    pixel: torch.Tensor
    prompt: str
    original_size: tuple[int, int]  # h, w
    cropped_size: Optional[tuple[int, int]]  # h, w
    dhdw: Optional[tuple[int, int]]  # dh, dw
    extras: dict = None
    # mask: torch.Tensor | None = None


def dirwalk(path: Path, cond: Optional[Callable] = None) -> Generator[Path, None, None]:
    for p in path.iterdir():
        if p.is_dir():
            yield from dirwalk(p, cond)
        else:
            if isinstance(cond, Callable):
                if not cond(p):
                    continue
            yield p


class StoreBase(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        root_path,
        rank=0,
        dtype=torch.float16,
        process_batch_fn = "data.processors.identical",
        **kwargs,
    ):
        self.rank = rank
        self.root_path = Path(root_path)
        self.dtype = dtype
        self.kwargs = kwargs
        self.process_batch_fn = process_batch_fn
            
        self.length = 0
        self.rand_list: list = []
        self.raw_res: list[tuple[int, int]] = []
        self.curr_res: list[tuple[int, int]] = []

        assert self.root_path.exists()

    def get_raw_entry(self, index) -> tuple[bool, np.ndarray, str, (int, int)]:
        raise NotImplementedError

    def fix_aspect_randomness(self, rng: np.random.Generator):
        raise NotImplementedError
    
    def crop(self, entry: Entry, index: int) -> Entry:
        return entry, 0, 0
    
    @torch.no_grad()
    def get_batch(self, indices: list[int]) -> Entry:
        entries = [self._get_entry(i) for i in indices]
        crop_pos = []
        pixels = []
        prompts = []
        original_sizes = []
        cropped_sizes = []
        extras = []

        for e, i in zip(entries, indices):
            e = self.process_batch(e)
            e, dh, dw = self.crop(e, i)
            pixels.append(e.pixel)
            original_size = torch.asarray(e.original_size)
            original_sizes.append(original_size)

            cropped_size = e.pixel.shape[-2:]
            cropped_size = (
                (cropped_size[0] * 8, cropped_size[1] * 8)
                if e.is_latent
                else cropped_size
            )
            cropped_size = torch.asarray(cropped_size)
            cropped_sizes.append(cropped_size)

            cropped_pos = (dh, dw)
            cropped_pos = (
                (cropped_pos[0] * 8, cropped_pos[1] * 8) if e.is_latent else cropped_pos
            )
            cropped_pos = (cropped_pos[0] + e.dhdw[0], cropped_pos[1] + e.dhdw[1])
            cropped_pos = torch.asarray(cropped_pos)
            crop_pos.append(cropped_pos)
            prompts.append(e.prompt)
            extras.append(e.extras)

        is_latent = entries[0].is_latent
        shape = entries[0].pixel.shape

        for e in entries[1:]:
            assert e.is_latent == is_latent
            assert (
                e.pixel.shape == shape
            ), f"{e.pixel.shape} != {shape} for the same batch"

        pixel = torch.stack(pixels, dim=0).contiguous()
        cropped_sizes = torch.stack(cropped_sizes)
        original_sizes = torch.stack(original_sizes)
        crop_pos = torch.stack(crop_pos)

        return {
            "prompts": prompts,
            "pixels": pixel,
            "is_latent": is_latent,
            "target_size_as_tuple": cropped_sizes,
            "original_size_as_tuple": original_sizes,
            "crop_coords_top_left": crop_pos,
            "extras": extras,
        }

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        raise NotImplementedError

    def get_batch_extras(self, path):
        return None

    def process_batch(self, inputs: Entry):
        if isinstance(self.process_batch_fn, str):
            self.process_batch_fn = get_class(self.process_batch_fn)
            
        return self.process_batch_fn(inputs)

    def _get_entry(self, index) -> Entry:
        is_latent, pixel, prompt, original_size, dhdw, extras = self.get_raw_entry(
            index
        )
        pixel = pixel.to(dtype=self.dtype)
        shape = pixel.shape
        if shape[-1] == 3 and shape[-1] < shape[0] and shape[-1] < shape[1]:
            pixel = pixel.permute(2, 0, 1)  # HWC -> CHW

        return Entry(is_latent, pixel, prompt, original_size, None, dhdw, extras)

    def repeat_entries(self, k, res, index=None):
        repeat_strategy = self.kwargs.get("repeat_strategy", None)
        if repeat_strategy is not None:
            assert index is not None
            index_new = index.copy()
            for i, ent in enumerate(index):
                for strategy, mult in repeat_strategy:
                    if strategy in str(ent):
                        k.extend([k[i]] * (mult - 1))
                        res.extend([res[i]] * (mult - 1))
                        index_new.extend([index_new[i]] * (mult - 1))
                        break
        else:
            index_new = index
        return k, res, index_new

class LatentStore(StoreBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        prompt_mapping = next(dirwalk(self.root_path, lambda p: p.suffix == ".json"))
        prompt_mapping = json_lib.loads(Path(prompt_mapping).read_text())

        self.h5_paths = list(
            dirwalk(
                self.root_path,
                lambda p: p.suffix == ".h5" and "prompt_cache" not in p.stem,
            )
        )
        
        self.h5_keymap = {}
        self.h5_filehandles = {}
        self.paths = []
        self.keys = []
        progress = tqdm(
            total=len(prompt_mapping),
            desc=f"Loading latents",
            disable=self.rank != 0,
            leave=False,
            ascii=True,
        )

        has_h5_loc = "h5_path" in next(iter(prompt_mapping.values()))
        for idx, h5_path in enumerate(self.h5_paths):
            fs = h5.File(h5_path, "r", libver="latest")
            h5_name = h5_path.name
            
            for k in fs.keys():
                hashkey = k[:-8]  # ".latents"
                if hashkey not in prompt_mapping:
                    logger.warning(f"Key {k} not found in prompt_mapping")
                    continue
                
                it = prompt_mapping[hashkey]
                if not it["train_use"] or (has_h5_loc and it["h5_path"] != h5_name):
                    continue
                
                height, width, fp = it["train_height"], it["train_width"], it["file_path"]
                self.paths.append(fp)
                self.keys.append(k)
                self.raw_res.append((height, width))
                self.h5_keymap[k] = (h5_path, it, (height, width))
                progress.update(1)
                
        progress.close()
        self.length = len(self.keys)
        self.scale_factor = 0.13025
        logger.debug(f"Loaded {self.length} latent codes from {self.root_path}")

        self.keys, self.raw_res, self.paths = self.repeat_entries(self.keys, self.raw_res, index=self.paths)
        new_length = len(self.keys)
        if new_length != self.length:
            self.length = new_length
            logger.debug(f"Using {self.length} entries after applied repeat strategy")

    def setup_filehandles(self):
        self.h5_filehandles = {}
        for h5_path in self.h5_paths:
            self.h5_filehandles[h5_path] = h5.File(h5_path, "r", libver="latest")

    def get_raw_entry(self, index) -> tuple[bool, torch.tensor, str, (int, int)]:
        if len(self.h5_filehandles) == 0:
            self.setup_filehandles()
            
        latent_key = self.keys[index]
        h5_path, entry, original_size = self.h5_keymap[latent_key]
        
        # modify here if you want to use a different format
        prompt = entry["train_caption"]
        latent = torch.asarray(self.h5_filehandles[h5_path][latent_key][:]).float()
        dhdw = self.h5_filehandles[h5_path][latent_key].attrs.get("dhdw", (0, 0))
    
        # if scaled, we need to unscale the latent (training process will scale it back)
        scaled = self.h5_filehandles[h5_path][latent_key].attrs.get("scale", True)
        if scaled:
            latent = 1.0 / self.scale_factor * latent

        extras = self.get_batch_extras(self.paths[index])
        return True, latent, prompt, original_size, dhdw, extras


class DirectoryImageStore(StoreBase):
    def __init__(self, *args, **kwargs):

        self.load_into_memory = kwargs.get("load_into_memory", False)
        super().__init__(*args, **kwargs)
        label_ext = self.kwargs.get("label_ext", ".txt")
        self.paths = list(dirwalk(self.root_path, is_img))
        self.length = len(self.paths)
        self.transforms = IMAGE_TRANSFORMS
        logger.debug(f"Found {self.length} images in {self.root_path}")


        
        remove_paths = []
        for p in tqdm(
            self.paths,
            desc="Loading image sizes",
            leave=False,
            ascii=True,
        ):
            try:
                width, height = imagesize.get(p)
                self.raw_res.append((height, width)) 
            except Exception as e:

                print(f"\033[33mSkipped: error processing {p}: {e}\033[0m")
                print("use imagesize")
                remove_paths.append(p)

        remove_paths = set(remove_paths)
        self.paths = [p for p in self.paths if p not in remove_paths]
        self.length = len(self.raw_res)

        self.length = len(self.paths)
        self.prompts: list[str] = []
        for path in tqdm(
            self.paths,
            desc="Loading prompts",
            disable=self.rank != 0,
            leave=False,
            ascii=True,
        ):
            p = path.with_suffix(label_ext)
            try:
                with open(p, "r") as f:
                    self.prompts.append(f.read())
            except Exception as e:
                logger.warning(f"Skipped: error processing {p}: {e}")
                self.prompts.append("")

                
        self.prompts, self.raw_res, self.paths = self.repeat_entries(
            self.prompts, self.raw_res, index=self.paths
        )
        self.length = len(self.paths) 
        
        if len(self.prompts) != len(self.paths):
             logger.debug(f"Dataset length updated to {self.length} after applying repeat strategy.")

        if self.load_into_memory:
            logger.info("Loading all images into RAM based on the final entry list...")
            self.in_memory_images = []
            for p in tqdm(self.paths, desc="Reading images into memory", ascii=True):
                try:
                    _img = Image.open(p)
                    if _img.mode == "RGB":
                        img_array = np.array(_img)
                    elif _img.mode == "RGBA":
                        baimg = Image.new('RGB', _img.size, (255, 255, 255))
                        baimg.paste(_img, (0, 0), _img)
                        img_array = np.array(baimg)
                    else:
                        img_array = np.array(_img.convert("RGB"))
                    
                    self.in_memory_images.append(img_array)
                except Exception as e:
                    logger.warning(f"Failed to load image {p} into memory: {e}")
                    self.in_memory_images.append(None)

            
    def get_raw_entry(self, index) -> tuple[bool, torch.tensor, str, (int, int)]:
        prompt = self.prompts[index]
        p = self.paths[index]
        
        if self.load_into_memory:
            img_array = self.in_memory_images[index]
            if img_array is None:
                logger.error(f"Image at index {index} ({p}) failed to load into memory.")
                return False, torch.zeros(3, 224, 224), "", (224, 224), (0, 0), None
        else:
            try:
                _img = Image.open(p)
                if _img.mode == "RGB":
                    img_array = np.array(_img)
                elif _img.mode == "RGBA":
                    baimg = Image.new('RGB', _img.size, (255, 255, 255))
                    baimg.paste(_img, (0, 0), _img)
                    img_array = np.array(baimg)
                else:
                    img_array = np.array(_img.convert("RGB"))
            except Exception as e:
                logger.error(f"Failed to load image from disk {p}: {e}")
                return False, torch.zeros(3, 224, 224), "", (224, 224), (0, 0), None

        img = self.transforms(img_array)
        
        h, w = self.raw_res[index]
        dhdw = (0, 0)
        extras = self.get_batch_extras(p)
        return False, img, prompt, (h, w), dhdw, extras

        

class ChunkedDirectoryImageStore(StoreBase):
    def __init__(self, *args, **kwargs):
        self.chunk_size_gb = kwargs.get("chunk_size_gb", 5)
        self.copy_workers = kwargs.get("copy_workers", 8)
        self.use_shm = True
        self.transforms = IMAGE_TRANSFORMS
        super().__init__(*args, **kwargs)
        label_ext = self.kwargs.get("label_ext", ".txt")
        # copy_workers = self.kwargs.get("copy_workers", 8)

        logger.warning(f"chunk_size_gb:{self.chunk_size_gb},copy_workers:{self.copy_workers}")

        self._log(f"Initializing ChunkedStore. Root: {self.root_path}")
        
        
        
        cache_path = self.root_path / "dataset_index_cache.json"
        
        try:
            rank = dist.get_rank() if dist.is_initialized() else 0
        except:
            rank = 0

        self.all_entries = []
        loaded_from_cache = False
        expanded = []

        if cache_path.exists():
            self._log(f"Found cache file: {cache_path}. Loading directly...")
            try:
                with open(cache_path, "r") as f:
                    cache_data = json.load(f)
                    # print("Loaded cache data:", cache_data) 
                
                # for entry in cache_data:
                #     entry["path"] = Path(entry["path"])
                #     self.all_entries.append(entry)
                #     repeat, key = extract_repeat_key(entry["path"])
                #     # self._log(f"repeat:{repeat}, key:{key}")
                #     entry["dataset_key"] = key  
                #     # for _ in range(repeat):
                #     #     expanded.append(entry)
                # expanded.extend([entry] * repeat)

                for entry in cache_data:
                    entry["path"] = Path(entry["path"])
                    self.all_entries.append(entry)
                    repeat, key = extract_repeat_key(entry["path"])
                    # self._log(f"repeat:{repeat}, key:{key}")
                    entry["dataset_key"] = key  
                
                    expanded.extend([entry] * repeat)
                logger.warning(f"repeat:{repeat},new use cache")
                

                self.all_entries = expanded
            
                loaded_from_cache = True
                self._log(f"Loaded {len(self.all_entries)} entries from cache.")
            except Exception as e:
                self._log(f"Failed to load cache: {e}. Fallback to scanning.")

       
        if not loaded_from_cache:
            start_t = time.time()
            self._log(f"Scanning files in {self.root_path}...")
            all_paths = list(dirwalk(self.root_path, is_img))
            total_files = len(all_paths)
            self._log(f"Scanned {total_files} paths in {time.time()-start_t:.2f}s.")
            self._log(f"Starting PARALLEL metadata extraction (Thread Pool)...")

            def process_single_entry(p):
                try:
                    f_size = p.stat().st_size
                    w, h = imagesize.get(str(p))
                    
                    p_txt = p.with_suffix(label_ext)
                    prompt = ""
                    if p_txt.exists():
                        try:
                            with open(p_txt, "r", encoding="utf-8", errors="ignore") as f:
                                prompt = f.read().strip()
                        except Exception as e:
                            self._log(f"Error reading {p_txt}: {e}")
            
                    # self._log(f"Processed {p} with prompt: {prompt} and resolution: {w}x{h}")
                    return {
                        "path": p,
                        "prompt": prompt,
                        "res": (h, w),
                        "bytes": f_size
                    }
                except Exception as e:
                    self._log(f"Failed to process {p}: {e}")  # Log the error for debugging
                    return None
    
    
            max_workers = 64 
            results = []
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(process_single_entry, p) for p in all_paths]
                
                iterator = as_completed(futures)
                if rank == 0:
                    iterator = tqdm(iterator, total=total_files, desc="Indexing Metadata", unit="img", mininterval=1.0)
                
                for future in iterator:
                    res = future.result()
                    if res is not None:
                        results.append(res)

            self.all_entries = results
            for entry in self.all_entries:
                repeat, key = extract_repeat_key(entry["path"])
                entry["dataset_key"] = key   
                # self._log(f"key:{key},repeat:{repeat}")

                # for _ in range(repeat):
                #     expanded.append(entry)
                expanded.extend([entry] * repeat)
                
            # self._log(f"{expanded}")

            self.all_entries = expanded
            # for entry in self.all_entries:
            #     self._log(f"new: {entry["dataset_key"]}")
    
    
            
            scan_duration = time.time() - start_t
            if rank == 0:
                print(f"[Rank {rank}] Indexing finished. Valid: {len(self.all_entries)}/{total_files}. Time: {scan_duration:.2f}s")

            if rank == 0:
                self._log(f"Rank 0: Saving index to {cache_path} for next time...")
                try:
                    serializable_entries = []
                    for entry in self.all_entries:
                        e_copy = entry.copy()
                        e_copy["path"] = str(e_copy["path"])
                        serializable_entries.append(e_copy)
                    
                    with open(cache_path, "w") as f:
                        json.dump(serializable_entries, f)
                    self._log("Cache saved successfully.")
                except Exception as e:
                    self._log(f"Failed to save cache: {e}")
            
            if dist.is_initialized():
                dist.barrier()

        self._log(f"Indexed {len(self.all_entries)} valid entries.")

        self.chunks = []
        current_chunk = []
        current_chunk_size = 0
        limit_bytes = self.chunk_size_gb * 1024**3
        
        # np.random.seed(42) 
        np.random.shuffle(self.all_entries)

        for entry in self.all_entries:
            if current_chunk_size + entry["bytes"] > limit_bytes and len(current_chunk) > 0:
                self.chunks.append(current_chunk)
                current_chunk = []
                current_chunk_size = 0
            
            current_chunk.append(entry)
            current_chunk_size += entry["bytes"]
        
        if current_chunk:
            self.chunks.append(current_chunk)

        self._log(f"Dataset split into {len(self.chunks)} chunks (Target: {self.chunk_size_gb} GB).")
        
        self.current_chunk_idx = 0
        
        # self.buffers = [
        #     Path("/dev/shm/naifu_buffer_0"),
        #     Path("/dev/shm/naifu_buffer_1")
        # ]
        # self.buffers = [
        #     Path(f"/dev/shm/{entry['dataset_key']}_buffer_0") for entry in self.all_entries
        # ]
        self.buffers = [
            Path(f"/dev/shm/{entry['dataset_key']}_buffer_{i}") 
            for entry in self.all_entries 
            for i in [0, 1]
        ]


        self.current_buffer_idx = 0 
        
        self.prefetch_thread = None
        self.next_chunk_idx = 1 

        if len(self.chunks) > 0:
            self._log(f"!!! INITIALIZATION START !!!")
            self._load_chunk_to_shm(chunk_idx=0, buffer_path=self.buffers[0], is_background=False)
            if dist.is_initialized():
                self._log("Waiting for Rank 0 to finish initial copy...")
                dist.barrier()
            self._build_map_and_switch(chunk_idx=0, buffer_path=self.buffers[0])
            
            if len(self.chunks) > 1:
                self._log(f"Triggering initial background prefetch for Chunk 2...")
                self._start_prefetch(chunk_idx=1, buffer_path=self.buffers[1])
            self._log(f"!!! INITIALIZATION DONE !!!")
        else:
            logger.error("Dataset is empty!")

        # logger.info(f"repeat:{repeat},key:{key}")

    def _log(self, msg):
        try:
            if dist.is_initialized():
                rank = dist.get_rank()
            else:
                rank = 0
        except:
            rank = 0
        
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        print(f"[{timestamp}][Rank {rank}] {msg}", flush=True)

    def _load_chunk_to_shm(self, chunk_idx, buffer_path, is_background=False):
        try:
            if dist.is_initialized():
                rank = dist.get_rank()
            else:
                rank = 0
        except:
            rank = 0

        thread_name = "BACKGROUND" if is_background else "FOREGROUND"
        chunk_data = self.chunks[chunk_idx]
        
        if rank == 0:
            try:
                start_t = time.time()
                self._log(f"[{thread_name}] Rank 0: START Copying {len(chunk_data)} files to {buffer_path}...")

                if buffer_path.exists():
                    shutil.rmtree(buffer_path, ignore_errors=True)
                os.makedirs(buffer_path, exist_ok=True)
                
                def copy_task(entry):
                    src = entry["path"]
                    # dst_name = f"{hashlib.sha1(str(src).encode()).hexdigest()}{src.suffix}"
                    dst_name = f"{hashlib.sha1(str(src).encode()).hexdigest()}_{src.name}"
                    dst = buffer_path / dst_name
                    try:
                        shutil.copy2(src, dst)
                        return True
                    except Exception:
                        return False

                success_count = 0
                with ThreadPoolExecutor(max_workers=self.copy_workers) as executor:
                    futures = [executor.submit(copy_task, e) for e in chunk_data]
                    
                    for i, f in enumerate(as_completed(futures)):
                        if f.result():
                            success_count += 1
                        if i > 0 and i % 500000 == 0:
                            self._log(f"[{thread_name}] Copy Progress: {i}/{len(chunk_data)}")

                end_t = time.time()
                duration = end_t - start_t
                speed = len(chunk_data) / duration if duration > 0 else 0
                self._log(f"[{thread_name}] Rank 0: FINISHED Copy. Success: {success_count}/{len(chunk_data)}. Time: {duration:.2f}s ({speed:.1f} img/s)")
            
            except Exception as e:
                import traceback
                self._log(f"!!! CRITICAL ERROR IN COPY !!! : {e}")
                traceback.print_exc()



    def _start_prefetch(self, chunk_idx, buffer_path):
        self._log(f"Creating background thread for Chunk {chunk_idx + 1} -> {buffer_path.name}")
        self.prefetch_thread = threading.Thread(
            target=self._load_chunk_to_shm,
            args=(chunk_idx, buffer_path, True)
        )
        self.prefetch_thread.daemon = True 
        self.prefetch_thread.start()
        self._log(f"Background thread started.")

    def _build_map_and_switch(self, chunk_idx, buffer_path):
        map_start = time.time()
        self._log(f"Building path map for Chunk {chunk_idx + 1} from {buffer_path}...")
        
        chunk_data = self.chunks[chunk_idx]
        self.path_map = {}
        
        for entry in chunk_data:
            src = entry["path"]
            dst_name = f"{hashlib.sha1(str(src).encode()).hexdigest()}{src.suffix}"
            dst = buffer_path / dst_name
            
            if dst.exists():
                self.path_map[src] = dst
            else:
                pass 

        self.paths = []
        self.prompts = []
        self.raw_res = []
        
        for entry in chunk_data:
            if entry["path"] in self.path_map:
                self.paths.append(entry["path"])
            else:
                self.paths.append(entry["path"])
            self.prompts.append(entry["prompt"])
            self.raw_res.append(entry["res"])
        
        self.length = len(self.paths)
        map_duration = time.time() - map_start
        self._log(f"Map built in {map_duration:.4f}s. Total images: {self.length}. Valid in SHM: {len(self.path_map)}")

    def load_next_chunk(self):
        # self._log(f"--- load_next_chunk CALLED ---")
        
        next_buffer_idx = (self.current_buffer_idx + 1) % 2
        target_chunk_idx = self.next_chunk_idx
        
        self._log(f"Target Chunk: {target_chunk_idx + 1}. Target Buffer: {self.buffers[next_buffer_idx].name}")

        if self.prefetch_thread:
            if self.prefetch_thread.is_alive():
                self._log("Background thread is still running. Waiting for it to finish (Blocking)...")
                join_start = time.time()
                self.prefetch_thread.join()
                self._log(f"Background thread joined after {time.time() - join_start:.2f}s.")
            else:
                self._log("Background thread already finished.")
        
        if dist.is_initialized():
            # self._log("Entering Barrier (Waiting for Rank 0)...")
            dist.barrier()
            # self._log("Exited Barrier.")

        self._build_map_and_switch(target_chunk_idx, self.buffers[next_buffer_idx])
        
        self.current_chunk_idx = target_chunk_idx
        self.current_buffer_idx = next_buffer_idx
        
        next_prefetch_chunk_idx = (self.current_chunk_idx + 1) % len(self.chunks)
        prefetch_buffer_idx = (self.current_buffer_idx + 1) % 2
        
        self.next_chunk_idx = next_prefetch_chunk_idx
        
        self._log(f"Scheduling next prefetch: Chunk {next_prefetch_chunk_idx + 1} -> {self.buffers[prefetch_buffer_idx].name}")
        self._start_prefetch(next_prefetch_chunk_idx, self.buffers[prefetch_buffer_idx])
        
        # self._log(f"--- load_next_chunk COMPLETED ---")


    def get_raw_entry(self, index, visited=None):
        if visited is None:
            visited = set()
    
        if index in visited:
            raise RuntimeError(
                f"ChunkedDirectoryImageStore: all candidates in current chunk are bad. "
                f"visited={len(visited)}"
            )
        visited.add(index)
    
        # Get entry from all_entries using the index
        entry = self.all_entries[index]
        p = entry["path"]
        prompt = entry["prompt"]
        h, w = entry["res"]
        load_path = self.path_map.get(p, p)

        try:
            with Image.open(load_path) as _img:
                img_pil = _img.convert("RGB")
            img = self.transforms(img_pil)
            return False, img, prompt, (h, w), (0, 0), entry
    
        except Exception as e:
            bad = getattr(self, "_bad_indices", None)
            if bad is None:
                self._bad_indices = set()
                bad = self._bad_indices
            bad.add(index)
            logger.error(
                f"Error loading {load_path} (index {index}): {e}. "
                f"Skipping this sample and trying another one."
            )
            next_index = (index + 1) % self.length
            return self.get_raw_entry(next_index, visited)


    def __len__(self):
        return self.length