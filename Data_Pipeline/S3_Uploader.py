import boto3
import os
from pathlib import Path

s3 = boto3.client("s3")

SOURCE_BUCKET = "memoryscapes-media-dev"
DEST_PREFIX = "processed_photos/"

def upload_file(file_path):
    dest_key = f"{DEST_PREFIX}{file_path.name}"
    s3.upload_file(str(file_path), SOURCE_BUCKET, dest_key)
    print(f"Uploaded -> s3://{SOURCE_BUCKET}/{dest_key}")

    
def main():
    processed_photos = Path("app/Data/s3_image_upload")
    for img_file in processed_photos.iterdir():
        upload_file(img_file)

if __name__ == "__main__":
    main()