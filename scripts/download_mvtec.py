"""Download a user-provided official archive URL; safely extract bottle only."""
import argparse
from pathlib import Path
import tarfile
import urllib.request
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("--url", required=True, help="Official download URL obtained from the MVTec AD download form")
parser.add_argument("--root", default="data/mvtec_ad")
args = parser.parse_args()
root = Path(args.root); root.mkdir(parents=True, exist_ok=True)
archive = root/"download.tar"
with urllib.request.urlopen(args.url, timeout=60) as response, archive.open("wb") as output:
    shutil.copyfileobj(response, output)
with tarfile.open(archive) as package:
    for member in package:
        path = Path(member.name)
        if path.is_absolute() or ".." in path.parts or member.issym() or member.islnk():
            raise ValueError("Unsafe archive member")
        if path.parts and path.parts[0] == "bottle":
            package.extract(member, root, filter="data")
if not (root/"bottle"/"train"/"good").is_dir():
    raise ValueError("Archive did not contain bottle/train/good")
archive.unlink()
print(f"Bottle extracted to {root}")
