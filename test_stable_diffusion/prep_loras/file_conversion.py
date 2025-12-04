#!/usr/bin/env python3
import argparse
import pathlib
from PIL import Image
import pillow_heif  # pip install pillow-heif pillow

def convert_all(src_dir: pathlib.Path, dest_dir: pathlib.Path) -> None:
    pillow_heif.register_heif_opener()
    dest_dir.mkdir(parents=True, exist_ok=True)

    exts = {".png", ".heic", ".HEIC", ".PNG", ".DNG", ".dng"}
    for path in src_dir.iterdir():
        if path.suffix not in exts or not path.is_file():
            continue
        out_path = dest_dir / (path.stem + ".jpg")
        with Image.open(path) as img:
            if img.mode in ("RGBA", "LA"):
                background = Image.new("RGB", img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[-1])
                background.save(out_path, "JPEG", quality=95)
            else:
                img.convert("RGB").save(out_path, "JPEG", quality=95)
        ##delete original file after conversion
        path.unlink()
    
        print(f"Converted {path.name} -> {out_path.name}")

def main():
    parser = argparse.ArgumentParser(description="Convert PNG/HEIC images to JPEG.")
    parser.add_argument("--src", type=pathlib.Path, help="Folder with PNG/HEIC images")
    parser.add_argument("--dest", type=pathlib.Path, help="Folder to write JPEGs")
    args = parser.parse_args()
    convert_all(args.src, args.dest)

if __name__ == "__main__":
    main()

# example usage: python file_conversion.py --src /path/to/png_heic_folder --dest /path/to/jpeg_folder