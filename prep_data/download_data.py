"""CLI entry-point: download Danbooru2019 Portraits into data/raw/.

Danbooru2019 Portraits is distributed over rsync. This downloader lets you ask
for a fixed number of images and resumes cleanly by skipping files that already
exist in the target directory.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from dotenv import load_dotenv

from utils.paths import DEFAULT_RAW_IMAGE_DIR, project_path

load_dotenv()

DANBOORU2019_PORTRAITS_RSYNC_URL = (
    "rsync://176.9.41.242:873/biggan/portraits/"
)
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
MANIFEST_NAME = ".danbooru2019-portraits-files.txt"


@dataclass(frozen=True)
class RemoteImage:
    path: str
    size: int


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc

    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")

    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a limited number of Danbooru2019 Portraits images."
    )
    parser.add_argument(
        "--num-images",
        type=positive_int,
        required=True,
        help=(
            "Number of Danbooru2019 Portraits images that should exist locally "
            "after this command finishes."
        ),
    )
    parser.add_argument(
        "--target-dir",
        default=os.getenv(
            "DOWNLOAD_TARGET_DIR",
            os.getenv("DATA_RAW_IMAGE_DIR", str(DEFAULT_RAW_IMAGE_DIR)),
        ),
        help="Destination directory for raw anime images.",
    )
    parser.add_argument(
        "--rsync-url",
        default=os.getenv(
            "DANBOORU2019_PORTRAITS_RSYNC_URL",
            DANBOORU2019_PORTRAITS_RSYNC_URL,
        ),
        help="rsync URL for Danbooru2019 Portraits.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be downloaded without copying files.",
    )
    parser.add_argument(
        "--bwlimit",
        help="Optional rsync bandwidth limit, for example 5m or 1000.",
    )
    parser.add_argument(
        "--rsync-extra-arg",
        action="append",
        default=[],
        help=(
            "Additional rsync option. Repeat as needed, e.g. "
            "--rsync-extra-arg=--max-size=2m."
        ),
    )
    return parser.parse_args()


def get_rsync_path() -> str:
    rsync = shutil.which("rsync")
    if not rsync:
        raise RuntimeError(
            "rsync is required for Danbooru2019 Portraits. Install it with your "
            "system package manager, for example: sudo apt install rsync"
        )

    return rsync


def normalize_rsync_url(source_url: str) -> str:
    return source_url.rstrip("/") + "/"


def manifest_path(target_path: Path) -> Path:
    return target_path / MANIFEST_NAME


def is_image_path(path: str) -> bool:
    return PurePosixPath(path).suffix.lower() in IMAGE_EXTENSIONS


def safe_remote_path(path: str) -> str | None:
    if path.startswith("./"):
        path = path[2:]

    rel_path = PurePosixPath(path)
    if rel_path.is_absolute() or ".." in rel_path.parts or not rel_path.name:
        return None

    return rel_path.as_posix()


def parse_rsync_list_line(line: str) -> RemoteImage | None:
    parts = line.split(maxsplit=4)
    if len(parts) != 5:
        return None

    mode, size_text, _, _, path = parts
    if not mode or mode[0] != "-":
        return None

    rel_path = safe_remote_path(path)
    if not rel_path or not is_image_path(rel_path):
        return None

    try:
        size = int(size_text.replace(",", ""))
    except ValueError:
        return None

    return RemoteImage(path=rel_path, size=size)


def parse_manifest_line(line: str) -> RemoteImage | None:
    line = line.rstrip("\n")
    if not line:
        return None

    if "\t" not in line:
        rel_path = safe_remote_path(line)
        if not rel_path or not is_image_path(rel_path):
            return None
        return RemoteImage(path=rel_path, size=-1)

    size_text, rel_path = line.split("\t", 1)
    rel_path = safe_remote_path(rel_path)
    if not rel_path or not is_image_path(rel_path):
        return None

    try:
        size = int(size_text)
    except ValueError:
        return None

    return RemoteImage(path=rel_path, size=size)


def list_remote_images(
    rsync: str,
    source_url: str,
    extra_args: list[str] | None = None,
) -> list[RemoteImage]:
    cmd = [rsync, "--recursive", "--list-only"]
    if extra_args:
        cmd.extend(extra_args)
    cmd.append(normalize_rsync_url(source_url))

    print("[download] Listing remote image files ...")
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)

    images: list[RemoteImage] = []
    seen: set[str] = set()
    for line in result.stdout.splitlines():
        image = parse_rsync_list_line(line)
        if image and image.path not in seen:
            images.append(image)
            seen.add(image.path)

    if not images:
        raise RuntimeError("No image files were found in the remote rsync source.")

    return images


def local_image_installed(target_path: Path, image: RemoteImage) -> bool:
    image_path = target_path / Path(image.path)
    if not image_path.is_file():
        return False

    if image.size < 0:
        return True

    return image_path.stat().st_size == image.size


def count_manifest_images(target_path: Path) -> int:
    path = manifest_path(target_path)
    if not path.exists():
        return 0

    count = 0
    with path.open() as fh:
        for line in fh:
            image = parse_manifest_line(line)
            if image and local_image_installed(target_path, image):
                count += 1

    return count


def write_manifest(target_path: Path, images: list[RemoteImage]) -> None:
    path = manifest_path(target_path)
    with path.open("w") as fh:
        for image in images:
            fh.write(f"{image.size}\t{image.path}\n")


def rsync_base_command(
    rsync: str,
    dry_run: bool,
    bwlimit: str | None,
    extra_args: list[str] | None,
) -> list[str]:
    cmd = [
        rsync,
        "--archive",
        "--verbose",
        "--human-readable",
        "--partial",
        "--append-verify",
        "--info=progress2",
    ]

    if dry_run:
        cmd.append("--dry-run")
    if bwlimit:
        cmd.append(f"--bwlimit={bwlimit}")
    if extra_args:
        cmd.extend(extra_args)

    return cmd


def download_selected_images(
    rsync: str,
    source_url: str,
    target_path: Path,
    images: list[RemoteImage],
    dry_run: bool,
    bwlimit: str | None,
    extra_args: list[str] | None,
) -> None:
    if not images:
        print("[download] Nothing new to download.")
        return

    source_url = normalize_rsync_url(source_url)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8") as files_from:
        for image in images:
            files_from.write(f"{image.path}\n")
        files_from.flush()

        cmd = rsync_base_command(rsync, dry_run, bwlimit, extra_args)
        cmd.extend(
            [
                f"--files-from={files_from.name}",
                source_url,
                str(target_path) + "/",
            ]
        )
        subprocess.run(cmd, check=True)


def download_danbooru_images(
    source_url: str,
    target_path: str | Path,
    num_images: int,
    dry_run: bool = False,
    bwlimit: str | None = None,
    extra_args: list[str] | None = None,
) -> str:
    rsync = get_rsync_path()
    source_url = normalize_rsync_url(source_url)
    target_path = project_path(target_path).resolve()
    target_path.mkdir(parents=True, exist_ok=True)

    print("[download] Source            : Danbooru2019 Portraits")
    print(f"[download] rsync URL         : {source_url}")
    print(f"[download] Destination       : {target_path}")
    print(f"[download] Requested images  : {num_images:,}")
    if dry_run:
        print("[download] Dry run           : enabled")

    manifest_count = count_manifest_images(target_path)
    if manifest_count >= num_images:
        print(
            f"[download] Already installed : {manifest_count:,} images found. "
            "Skipping download."
        )
        return str(target_path)

    remote_images = list_remote_images(rsync, source_url, extra_args=extra_args)
    if len(remote_images) < num_images:
        raise RuntimeError(
            f"Requested {num_images:,} images, but the remote source only lists "
            f"{len(remote_images):,} images."
        )

    local_images = [
        image for image in remote_images if local_image_installed(target_path, image)
    ]

    if len(local_images) >= num_images:
        selected = local_images[:num_images]
        if not dry_run:
            write_manifest(target_path, selected)
        print(
            f"[download] Already installed : {len(local_images):,} matching "
            "Danbooru images found. Skipping download."
        )
        return str(target_path)

    needed = num_images - len(local_images)
    missing_images = [
        image for image in remote_images if not local_image_installed(target_path, image)
    ]
    selected_missing = missing_images[:needed]
    if len(selected_missing) < needed:
        raise RuntimeError(
            f"Need {needed:,} new images, but only {len(selected_missing):,} "
            "download candidates were found."
        )

    print(f"[download] Existing images   : {len(local_images):,}")
    print(f"[download] New images needed : {len(selected_missing):,}")

    download_selected_images(
        rsync,
        source_url,
        target_path,
        selected_missing,
        dry_run,
        bwlimit,
        extra_args,
    )

    if not dry_run:
        installed = [
            image for image in remote_images if local_image_installed(target_path, image)
        ][:num_images]
        write_manifest(target_path, installed)
        print(f"[download] Installed images  : {len(installed):,}")

    print(f"[download] Stored at         : {target_path}")
    return str(target_path)


def main() -> None:
    args = parse_args()

    try:
        download_danbooru_images(
            args.rsync_url,
            args.target_dir,
            num_images=args.num_images,
            dry_run=args.dry_run,
            bwlimit=args.bwlimit,
            extra_args=args.rsync_extra_arg,
        )
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
