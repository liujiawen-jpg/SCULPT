#!/usr/bin/env python3
"""Download original benchmark datasets, checking immutable source metadata.

Only source expression data is downloaded. No task checkpoints or predictions
are obtained. Run --dry-run to inspect the complete plan without downloading.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import urllib.request

import yaml


def checksum(path: Path, algorithm: str = "md5") -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download_file(spec: dict, destination: Path) -> None:
    """Stream to a partial file; publish only after size and MD5 validation."""
    expected_bytes = int(spec["bytes"])
    if destination.exists():
        if destination.stat().st_size != expected_bytes or checksum(destination) != spec["md5"]:
            raise ValueError(f"Existing file failed source verification: {destination}")
        print(f"Verified existing {destination}", flush=True)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".partial")
    request = urllib.request.Request(spec["url"], headers={"User-Agent": "CoSIT-J0-dataset-downloader/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as stream:
        while block := response.read(8 << 20):
            stream.write(block)
    if partial.stat().st_size != expected_bytes or checksum(partial) != spec["md5"]:
        raise ValueError(f"Downloaded file failed size/MD5 verification (kept for inspection): {partial}")
    os.replace(partial, destination)
    print(f"Downloaded and verified {destination}", flush=True)


def download_data(config: Path, raw_root: Path, contexts=None, dry_run=False):
    catalog = yaml.safe_load(Path(config).read_text())
    selected = list(contexts) if contexts else list(catalog["datasets"])
    unknown = set(selected) - set(catalog["datasets"])
    if unknown:
        raise ValueError(f"Unknown contexts: {sorted(unknown)}")
    plan = []
    for context in selected:
        spec = catalog["datasets"][context]
        destination = raw_root / ("X-Atlas-Orion" if spec["format"] == "orion_parquet" else spec["path"])
        record = {"context": context, "destination": str(destination), "license": spec["license"]}
        if "url" in spec:
            record.update(url=spec["url"], bytes=spec["bytes"], md5=spec["md5"])
            if not dry_run:
                download_file(spec, destination)
        else:
            record.update(repository=spec["repository"], revision=spec["revision"], allow_patterns=spec["allow_patterns"])
            if not dry_run:
                from huggingface_hub import snapshot_download
                snapshot_download(repo_id=spec["repository"], repo_type="dataset", revision=spec["revision"],
                                  allow_patterns=spec["allow_patterns"], local_dir=destination,
                                  max_workers=4)
                shards = list((destination / "data").glob("HCT116_Batch*.parquet"))
                if len(shards) != int(spec["expected_shards"]):
                    raise ValueError(f"Expected {spec['expected_shards']} HCT116 shards; found {len(shards)}")
        plan.append(record)
    if dry_run:
        print(json.dumps(plan, indent=2))
    else:
        raw_root.mkdir(parents=True, exist_ok=True)
        (raw_root / "DOWNLOAD_RECEIPT.json").write_text(json.dumps({"schema": "cosit_source_download_v1", "sources": plan}, indent=2) + "\n")
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/datasets.yaml"))
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--contexts", nargs="+")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    download_data(args.config, args.raw_root, args.contexts, args.dry_run)


if __name__ == "__main__":
    main()
