"""Fetch a pinned official comparison source without installing or running it."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent


def fetch(method: str, directory: Path) -> Path:
    record = json.loads((HERE / "registry.json").read_text())[method]
    directory = directory.resolve()
    if directory.exists():
        actual = subprocess.check_output(
            ["git", "-C", str(directory), "rev-parse", "HEAD"], text=True
        ).strip()
        if actual != record["commit"]:
            raise RuntimeError("Existing checkout has a different revision; refusing to replace it")
        if subprocess.check_output(
            ["git", "-C", str(directory), "status", "--porcelain"], text=True
        ).strip():
            raise RuntimeError("Existing comparison checkout contains local changes")
        return directory
    directory.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", str(directory)], check=True)
    subprocess.run(["git", "-C", str(directory), "remote", "add", "origin", record["repository"]], check=True)
    subprocess.run(["git", "-C", str(directory), "fetch", "--depth", "1", "origin", record["commit"]], check=True)
    subprocess.run(["git", "-C", str(directory), "checkout", "--detach", "FETCH_HEAD"], check=True)
    if not (directory / record["license_file"]).is_file():
        raise RuntimeError("Pinned official license was not found; inspect the checkout")
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=tuple(json.loads((HERE / "registry.json").read_text())))
    parser.add_argument("--directory", required=True, type=Path)
    args = parser.parse_args()
    print(fetch(args.method, args.directory))


if __name__ == "__main__":
    main()
