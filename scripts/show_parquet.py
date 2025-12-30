import duckdb

con = duckdb.connect("e621_metadata.duckdb")
res = con.execute("""
    SELECT 
        COUNT(*) as total_metadata, 
        SUM(CAST(is_downloaded AS INT)) as actual_images 
    FROM posts
""").fetchone()

print(f"元数据总量: {res[0]}")
print(f"成功下载的图片量: {res[1]}")
print(f"下载失败的数量: {res[0] - res[1]}")