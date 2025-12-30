import requests
import duckdb
import threading
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

USER_AGENT = "leafmoone_pro_v5"
DB_NAME = "e621_metadata.duckdb"
BASE_IMAGE_DIR = "/workspace/dataset/1_e621_db/image"
PARQUET_OUTPUT = "/workspace/dataset/1_e621_db/e621_metadata.parquet"
THREADS = 108        
MAX_POSTS = 10000    
PAGE_LIMIT = 320    

PROXIES = None 

class E621SpiderPro:
    def __init__(self):
        os.makedirs(BASE_IMAGE_DIR, exist_ok=True)
        os.makedirs(os.path.dirname(PARQUET_OUTPUT), exist_ok=True)
        self.init_db()
        self.headers = {'User-Agent': USER_AGENT}

    def init_db(self):
        with duckdb.connect(DB_NAME) as conn:
            conn.execute(f'''
                CREATE TABLE IF NOT EXISTS posts (
                    id BIGINT PRIMARY KEY,
                    width INTEGER,
                    height INTEGER,
                    general TEXT,
                    artist TEXT,
                    copyright TEXT,
                    character TEXT,
                    species TEXT,
                    rating TEXT,
                    year TEXT,
                    nsfw TEXT,
                    resolution TEXT,
                    e621 TEXT,
                    file_ext TEXT,
                    is_downloaded BOOLEAN DEFAULT FALSE
                )
            ''')

    def download_worker(self, post, page_folder):
        try:
            p_id = post['id']
            file_info = post.get('file', {})
            img_url = file_info.get('url')
            ext = file_info.get('ext')
            
            download_success = False
            if img_url:
                save_path = os.path.join(page_folder, f"{p_id}.{ext}")
                if os.path.exists(save_path):
                    download_success = True
                else:
                    resp = requests.get(img_url, headers=self.headers, timeout=10, 
                                        stream=True, proxies=PROXIES)
                    if resp.status_code == 200:
                        with open(save_path, 'wb') as f:
                            for chunk in resp.iter_content(chunk_size=8192):
                                f.write(chunk)
                        download_success = True

            w, h = file_info.get('width', 0), file_info.get('height', 0)
            pixels = w * h
            res_tag = 'low resolution' if pixels < 1048576 else ('medium resolution' if pixels < 4194304 else 'high resolution')
            
            rating_map = {'s': 'general', 'q': 'questionable', 'e': 'explicit'}
            full_rating = rating_map.get(post['rating'], post['rating'])
            nsfw_tag = 'nsfw' if full_rating == 'explicit' else ('sfw' if full_rating == 'general' else full_rating)

            year_tag = None
            if post.get('created_at'):
                yyyy = int(post['created_at'][:4])
                year_tag = f"year {yyyy}"
                if yyyy >= 2022: year_tag += ", newest"
                elif yyyy <= 2018: year_tag += ", oldest"

            tags = post['tags']
            return (
                p_id, w, h, 
                ", ".join(tags['general']), ", ".join(tags['artist']), 
                ", ".join(tags['copyright']), ", ".join(tags['character']), 
                ", ".join(tags['species']), 
                full_rating, year_tag, nsfw_tag, res_tag, 
                "e621", ext, download_success
            )
        except Exception:
            return None

    def fetch_page(self, last_id):
        page_param = "" if last_id >= 99999999 else f"&page=b{last_id}"
        url = f"https://e621.net/posts.json?limit={PAGE_LIMIT}{page_param}"
        try:
            resp = requests.get(url, headers=self.headers, timeout=15, proxies=PROXIES)
            if resp.status_code == 200:
                posts = resp.json().get('posts', [])
                return posts
            elif resp.status_code == 429:
                print("\n[Rate Limit] 触发频率限制，等待 15 秒...")
                time.sleep(15)
            else:
                print(f"\n[HTTP Error] 状态码: {resp.status_code}")
        except Exception as e:
            print(f"\n[Network Error] 无法连接到 e621 API: {e}")
        return []

    def run(self):
        with duckdb.connect(DB_NAME) as conn:
            min_id_row = conn.execute("SELECT MIN(id) FROM posts").fetchone()[0]
            current_min_id = min_id_row if min_id_row else 99999999
            start_count = conn.execute("SELECT COUNT(*) FROM posts").fetchone()[0]

        print(f"Starting Scraper. Initial DB count: {start_count}")
        
        pbar = tqdm(total=MAX_POSTS, desc="Total Progress", unit="post")
        pbar.update(min(start_count, MAX_POSTS))
        
        total_saved = start_count
        while total_saved < MAX_POSTS:
            posts = self.fetch_page(current_min_id)
            if not posts:
                print("\nNo more posts or Error fetching. Exiting.")
                break

            page_last_id = posts[0]['id']
            page_folder = os.path.join(BASE_IMAGE_DIR, f"batch_{page_last_id}")
            os.makedirs(page_folder, exist_ok=True)

            page_results = []
            with ThreadPoolExecutor(max_workers=THREADS) as executor:
                futures = {executor.submit(self.download_worker, p, page_folder): p for p in posts}
                
                for future in as_completed(futures):
                    res = future.result()
                    if res:
                        page_results.append(res)
                        pbar.update(1) 
            
            if page_results:
                with duckdb.connect(DB_NAME) as conn:
                    conn.executemany('INSERT OR REPLACE INTO posts VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)', page_results)
            
            total_saved += len(page_results)
            current_min_id = posts[-1]['id']
            
            time.sleep(1)

        pbar.close()
        self.export_to_parquet()

    def export_to_parquet(self):
        print(f"Exporting to: {PARQUET_OUTPUT}")
        try:
            with duckdb.connect(DB_NAME) as conn:
                conn.execute(f"COPY posts TO '{PARQUET_OUTPUT}' (FORMAT PARQUET)")
            print("Export complete.")
        except Exception as e:
            print(f"Export failed: {e}")

if __name__ == "__main__":
    spider = E621SpiderPro()
    spider.run()