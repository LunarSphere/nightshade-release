import boto3
import os
from pathlib import Path

s3 = boto3.client("s3")

SOURCE_BUCKET = "memoryscapes-media-dev"
SOURCE_PREFIX = "uploads/raw"

def download_photos():
    objects = s3.list_objects_v2(Bucket=SOURCE_BUCKET, Prefix=SOURCE_PREFIX)
    download_dir = Path("/app/Data/input_images")
    download_dir.mkdir(parents=True, exist_ok=True)
    for obj in objects.get("Contents", []):
        key = obj["Key"]
        if key.endswith("/"): 
            continue
        file_path = download_dir / Path(key).name
        s3.download_file(SOURCE_BUCKET, key, str(file_path))
        # print(f"Downloaded: {file_path}")
    return download_dir

def main():
    photos_dir = download_photos()

if __name__ == "__main__":
    main()
