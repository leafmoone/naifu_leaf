#!/usr/bin/env python3
# -*- coding: utf-8 -*-



API_BASE = "https://danbooru.donmai.us"  
ID_START = 1                     
ID_END = 9697148                         
USE_AUTH = True
USERNAME = "waw1w1"               
API_KEY = "eTDrya4HPR962xZsq4BUPWDd"              
METADATA_THREADS = 256                    
DOWNLOAD_THREADS = 1024                
API_MAX_REQ_PER_SEC = 70         
API_BURST = 100              
API_CONNECT_TIMEOUT = 8
API_READ_TIMEOUT = 20
DL_CONNECT_TIMEOUT = 8
DL_READ_TIMEOUT = 90
API_RETRIES = 5
DL_RETRIES = 5
RETRY_BACKOFF_BASE = 1.5               
CHUNK_SIZE = 8388608                 
PIPELINE_BUFFER = 32768                  
ALLOWED_IMAGE_EXTS = {"jpg", "jpeg", "png", "webp"}
KEEP_EXTENSION = True          
SKIP_EXISTING = True                 
TRANSPARENT_ALPHA_THRESHOLD = 250      
TRANSPARENT_MIN_FRACTION = 0.02         
OUTPUT_DIR = "/mnt/raid0/linux-train/danbroou-data/images"         
TRANSPARENT_DIR = "/mnt/raid0/linux-train/danbroou-data/trabsparent" 
TMP_DIR = "/mnt/raid0/linux-train/danbroou-data/tmp"             
LOG_EVERY_N = 50                     
# ====== 配置结束 ======

from pathlib import Path
import time
import shutil
import threading
from queue import Queue
from typing import Optional, Tuple, Dict, Any

import socket
import requests
from urllib3.exceptions import ReadTimeoutError as URLLib3ReadTimeoutError
from PIL import Image

def log(msg: str):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")

def ensure_dirs():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(TRANSPARENT_DIR).mkdir(parents=True, exist_ok=True)
    Path(TMP_DIR).mkdir(parents=True, exist_ok=True)

class RateLimiter:
    """令牌桶限速：支持突发"""
    def __init__(self, rate_per_sec: float, burst: int = 1):
        self.rate = max(rate_per_sec, 0.0)
        self.capacity = max(int(burst), 1)
        self.tokens = float(self.capacity)
        self.ts = time.monotonic()
        self.lock = threading.Lock()
    def wait(self):
        if self.rate <= 0:
            return
        with self.lock:
            now = time.monotonic()
            elapsed = now - self.ts
            self.ts = now
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            if self.tokens < 1.0:
                sleep_s = (1.0 - self.tokens) / self.rate
                time.sleep(sleep_s)
                self.ts += sleep_s
                self.tokens = min(self.capacity, self.tokens + sleep_s * self.rate)
            self.tokens -= 1.0

api_rate = RateLimiter(API_MAX_REQ_PER_SEC, burst=API_BURST)


_thread_local = threading.local()
def get_session() -> requests.Session:
    s = getattr(_thread_local, "session", None)
    if s is None:
        s = requests.Session()
        if USE_AUTH and USERNAME and API_KEY:
            s.auth = (USERNAME, API_KEY)
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=max((METADATA_THREADS + DOWNLOAD_THREADS) * 2, 64),
            pool_maxsize=max((METADATA_THREADS + DOWNLOAD_THREADS) * 2, 64),
            max_retries=0,
        )
        s.mount("https://", adapter)
        s.mount("http://", adapter)
        _thread_local.session = s
    return s

def safe_write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(text)

def build_tags_text(post: Dict[str, Any]) -> str:
    def split_space(s: str):
        return [t for t in (s or "").split() if t]
    order_fields = [
        "tag_string_character",
        "tag_string_copyright",
        "tag_string_artist",
        "tag_string_general",
    ]
    tags = []
    for f in order_fields:
        tags.extend(split_space(post.get(f, "")))
    def transform(tag: str) -> str:
        t = tag.replace("_", " ")
        t = t.replace("(", r"\(").replace(")", r"\)")
        return t
    return ", ".join(transform(t) for t in tags)

def choose_file_url(post: Dict[str, Any]) -> str:
    return post.get("file_url") or post.get("large_file_url") or ""

def is_allowed_image_post(post: Dict[str, Any]) -> bool:
    ext = (post.get("file_ext") or "").lower()
    return (ext in ALLOWED_IMAGE_EXTS) and bool(choose_file_url(post))

def is_transparent_image_file(path: Path, ext: Optional[str] = None) -> bool:

    try:
        if ext and ext.lower() in {"jpg", "jpeg"}:
            return False

        with Image.open(path) as im:
            im.load()
            has_alpha_mode = im.mode in ("RGBA", "LA")
            has_transparency_flag = "transparency" in im.info

            if has_alpha_mode:
                alpha = im.getchannel("A")
            elif has_transparency_flag:
                im = im.convert("RGBA")
                alpha = im.getchannel("A")
            else:
                return False

            hist = alpha.histogram()  # 长度 256
            total = sum(hist)
            if total == 0:
                return False

            extrema = alpha.getextrema()
            if not extrema or extrema[0] >= 255:
                return False

            thr = max(0, min(255, TRANSPARENT_ALPHA_THRESHOLD))
            transparent_pixels = sum(hist[0:thr + 1])
            frac = transparent_pixels / total

            return frac >= TRANSPARENT_MIN_FRACTION

    except Exception:
        return False

def find_existing_image(post_id: int) -> Tuple[Optional[Path], Optional[Path]]:
    if KEEP_EXTENSION:
        for ext in ALLOWED_IMAGE_EXTS:
            p_norm = Path(OUTPUT_DIR) / f"{post_id}.{ext}"
            if p_norm.exists():
                return p_norm, Path(OUTPUT_DIR) / f"{post_id}.txt"
            p_trans = Path(TRANSPARENT_DIR) / f"{post_id}.{ext}"
            if p_trans.exists():
                return p_trans, Path(TRANSPARENT_DIR) / f"{post_id}.txt"
        return None, None
    else:
        p_norm = Path(OUTPUT_DIR) / f"{post_id}"
        if p_norm.exists():
            return p_norm, Path(OUTPUT_DIR) / f"{post_id}.txt"
        p_trans = Path(TRANSPARENT_DIR) / f"{post_id}"
        if p_trans.exists():
            return p_trans, Path(TRANSPARENT_DIR) / f"{post_id}.txt"
        return None, None

def is_fully_completed(post_id: int) -> bool:
    img_path, txt_path = find_existing_image(post_id)
    return (img_path is not None) and txt_path.exists()

def fetch_post_json(post_id: int) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    返回 (status, data)
    status: ok | 404 | 429 | 5xx | http_err | json_err | exception
    """
    url = f"{API_BASE.rstrip('/')}/posts/{post_id}.json"
    for attempt in range(1, API_RETRIES + 1):
        try:
            api_rate.wait()
            r = get_session().get(url, timeout=(API_CONNECT_TIMEOUT, API_READ_TIMEOUT))
        except requests.RequestException:
            if attempt >= API_RETRIES:
                return "exception", None
            time.sleep(RETRY_BACKOFF_BASE ** (attempt - 1))
            continue

        if r.status_code == 404:
            return "404", None
        if r.status_code == 429:
            if attempt >= API_RETRIES:
                return "429", None
            retry_after = r.headers.get("Retry-After")
            sleep_s = float(retry_after) if retry_after else max(1.0, RETRY_BACKOFF_BASE ** (attempt - 1))
            time.sleep(sleep_s)
            continue
        if 500 <= r.status_code < 600:
            if attempt >= API_RETRIES:
                return "5xx", None
            time.sleep(RETRY_BACKOFF_BASE ** (attempt - 1))
            continue
        if not r.ok:
            return "http_err", None

        try:
            return "ok", r.json()
        except Exception:
            return "json_err", None

def download_file(url: str, tmp_path: Path) -> Tuple[bool, Optional[str]]:
 
    for attempt in range(1, DL_RETRIES + 1):
        try:
            with get_session().get(url, stream=True, timeout=(DL_CONNECT_TIMEOUT, DL_READ_TIMEOUT)) as r:
                if r.status_code == 429:
                    if attempt >= DL_RETRIES:
                        return False, "http_429"
                    retry_after = r.headers.get("Retry-After")
                    sleep_s = float(retry_after) if retry_after else max(1.0, RETRY_BACKOFF_BASE ** (attempt - 1))
                    time.sleep(sleep_s)
                    continue
                if 500 <= r.status_code < 600:
                    if attempt >= DL_RETRIES:
                        return False, "http_5xx"
                    time.sleep(RETRY_BACKOFF_BASE ** (attempt - 1))
                    continue
                if not r.ok:
                    return False, f"http_{r.status_code}"

                tmp_path.parent.mkdir(parents=True, exist_ok=True)
                with tmp_path.open("wb") as f:
                    r.raw.decode_content = True
                    shutil.copyfileobj(r.raw, f, length=CHUNK_SIZE)
                return True, None

        except (requests.exceptions.ReadTimeout,
                requests.exceptions.ConnectionError,
                requests.exceptions.ChunkedEncodingError,
                URLLib3ReadTimeoutError,
                socket.timeout):
            try:
                if tmp_path.exists():
                    tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
            if attempt >= DL_RETRIES:
                return False, "read_timeout"
            time.sleep(RETRY_BACKOFF_BASE ** (attempt - 1))
            continue

        except Exception:
            try:
                if tmp_path.exists():
                    tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
            if attempt >= DL_RETRIES:
                return False, "io_err"
            time.sleep(RETRY_BACKOFF_BASE ** (attempt - 1))
            continue

    return False, "net_err"


class DownloadTask:
    __slots__ = ("post_id", "ext", "url", "tags_text")
    def __init__(self, post_id: int, ext: str, url: str, tags_text: str):
        self.post_id = post_id
        self.ext = ext
        self.url = url
        self.tags_text = tags_text

def producer_worker(id_queue: Queue, dl_queue: Queue, stats: Dict[str, int], stats_lock: threading.Lock, progress: Dict[str, int]):
    while True:
        post_id = id_queue.get()
        if post_id is None:
            id_queue.task_done()
            break

        if SKIP_EXISTING and is_fully_completed(post_id):
            with stats_lock:
                stats["exists"] = stats.get("exists", 0) + 1
                progress["produced"] += 1
            id_queue.task_done()
            continue

        status, post = fetch_post_json(post_id)
        if status != "ok" or not post:
            with stats_lock:
                if status == "404":
                    stats["not_found"] = stats.get("not_found", 0) + 1
                else:
                    stats["api_err"] = stats.get("api_err", 0) + 1
                progress["produced"] += 1
            id_queue.task_done()
            continue

        ext = (post.get("file_ext") or "").lower()
        if ext == "gif":
            with stats_lock:
                stats["skipped_gif"] = stats.get("skipped_gif", 0) + 1
                progress["produced"] += 1
            id_queue.task_done()
            continue

        if not is_allowed_image_post(post):
            with stats_lock:
                stats["non_image"] = stats.get("non_image", 0) + 1
                progress["produced"] += 1
            id_queue.task_done()
            continue

        tags_text = build_tags_text(post)
        url = choose_file_url(post)

        img_path, txt_path = find_existing_image(post_id)
        if img_path is not None:
            try:
                if not txt_path.exists():
                    safe_write_text(txt_path, tags_text)
                with stats_lock:
                    stats["exists+tag_refreshed"] = stats.get("exists+tag_refreshed", 0) + 1
                    progress["produced"] += 1
            finally:
                id_queue.task_done()
            continue

        dl_queue.put(DownloadTask(post_id, ext, url, tags_text))
        with stats_lock:
            progress["produced"] += 1
        id_queue.task_done()

def download_worker(dl_queue: Queue, stats: Dict[str, int], stats_lock: threading.Lock, progress: Dict[str, int]):
    while True:
        task = dl_queue.get()
        if task is None:
            dl_queue.task_done()
            break

        try:
            post_id, ext, url, tags_text = task.post_id, task.ext, task.url, task.tags_text
            fname = f"{post_id}.{ext}" if KEEP_EXTENSION else str(post_id)
            tmp_path = Path(TMP_DIR) / fname

            ok, err = download_file(url, tmp_path)
            if not ok:
                with stats_lock:
                    if err and err.startswith("http_"):
                        stats[err] = stats.get(err, 0) + 1
                    else:
                        stats["download_err"] = stats.get("download_err", 0) + 1
                    progress["downloaded"] += 1
                try:
                    if tmp_path.exists():
                        tmp_path.unlink(missing_ok=True)
                except Exception:
                    pass
                dl_queue.task_done()
                continue

            is_transp = is_transparent_image_file(tmp_path, ext)
            dst_dir = Path(TRANSPARENT_DIR) if is_transp else Path(OUTPUT_DIR)
            final_img = dst_dir / fname
            final_txt = dst_dir / f"{post_id}.txt"

            try:
                final_img.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(tmp_path), str(final_img))
            finally:
                try:
                    if tmp_path.exists():
                        tmp_path.unlink(missing_ok=True)
                except Exception:
                    pass

            safe_write_text(final_txt, tags_text)

            with stats_lock:
                if is_transp:
                    stats["downloaded_transparent"] = stats.get("downloaded_transparent", 0) + 1
                else:
                    stats["downloaded"] = stats.get("downloaded", 0) + 1
                progress["downloaded"] += 1

            dl_queue.task_done()

        except Exception:
            try:
                if "tmp_path" in locals() and isinstance(tmp_path, Path) and tmp_path.exists():
                    tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
            with stats_lock:
                stats["download_err"] = stats.get("download_err", 0) + 1
                progress["downloaded"] += 1
            dl_queue.task_done()
            continue

def main():
    ensure_dirs()
    lo, hi = (ID_START, ID_END) if ID_START <= ID_END else (ID_END, ID_START)
    ids = list(range(lo, hi + 1))
    total = len(ids)

    log(f"开始：ID范围 {lo}-{hi}（共{total}） | API {API_BASE}")
    log(f"元数据线程 {METADATA_THREADS} | 下载线程 {DOWNLOAD_THREADS} | API速率 {API_MAX_REQ_PER_SEC}/s, burst={API_BURST}")
    log(f"超时：API({API_CONNECT_TIMEOUT},{API_READ_TIMEOUT}) 下载({DL_CONNECT_TIMEOUT},{DL_READ_TIMEOUT}) | 重试 API={API_RETRIES}, DL={DL_RETRIES}")
    log(f"透明判定：alpha≤{TRANSPARENT_ALPHA_THRESHOLD} 且占比≥{TRANSPARENT_MIN_FRACTION*100:.1f}% 视为透明")
    log("已禁爬格式：GIF（不下载、不写标签）")

    id_queue: Queue = Queue(maxsize=0)
    dl_queue: Queue = Queue(maxsize=PIPELINE_BUFFER)

    for i in ids:
        id_queue.put(i)
    for _ in range(METADATA_THREADS):
        id_queue.put(None)

    stats: Dict[str, int] = {}
    stats_lock = threading.Lock()
    progress = {"produced": 0, "downloaded": 0}

    prod_threads = [threading.Thread(target=producer_worker, name=f"producer-{i+1}",
                                     args=(id_queue, dl_queue, stats, stats_lock, progress), daemon=True)
                    for i in range(METADATA_THREADS)]
    for t in prod_threads:
        t.start()

    dl_threads = [threading.Thread(target=download_worker, name=f"downloader-{i+1}",
                                   args=(dl_queue, stats, stats_lock, progress), daemon=True)
                  for i in range(DOWNLOAD_THREADS)]
    for t in dl_threads:
        t.start()

    last_logged = 0
    try:
        while True:
            time.sleep(0.5)
            done = progress["produced"]
            dled = progress["downloaded"]
            if done >= total:
                break
            if done - last_logged >= LOG_EVERY_N:
                last_logged = done
                log(f"进度：API处理 {done}/{total} | 已完成下载 {dled}")
    except KeyboardInterrupt:
        log("用户中断，准备安全退出……")

    for t in prod_threads:
        t.join()

    for _ in range(DOWNLOAD_THREADS):
        dl_queue.put(None)
    for t in dl_threads:
        t.join()

    log("==== 完成统计 ====")
    with stats_lock:
        for k in sorted(stats.keys()):
            log(f"{k}: {stats[k]}")
        log(f"总计 API 处理: {progress['produced']}/{total} | 下载完成: {progress['downloaded']}")
    log("全部完成。")

if __name__ == "__main__":
    main()
