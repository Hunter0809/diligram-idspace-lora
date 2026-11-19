#!/usr/bin/env python3
"""Parallel HTTP Range downloader for the fixed IDSpace revision."""

from __future__ import annotations

import concurrent.futures
import fcntl
import hashlib
import json
import os
from pathlib import Path
import threading
import time
from urllib.parse import quote

import requests

ROOT = Path("/home/mr_jiang/qkr")
DATASET_ROOT = ROOT / "datasets/IDSpace"
FILES_ROOT = DATASET_ROOT / "files"
PARTS_ROOT = DATASET_ROOT / ".parts"
VERIFIED_ROOT = DATASET_ROOT / ".verified"
LOG_ROOT = ROOT / "logs"
MANIFEST_PATH = DATASET_ROOT / "manifest.json"
LOCK_PATH = DATASET_ROOT / ".download.lock"
COMPLETED_PATH = DATASET_ROOT / "download.completed"
REVISION = "a3de23b950d55a39fbd36cfaee16bd260c4d457d"
API_URL = "https://huggingface.co/api/datasets/cactuslab/IDSpace/tree/main?recursive=true&expand=true"
BASE_URL = f"https://huggingface.co/datasets/cactuslab/IDSpace/resolve/{REVISION}"
CHUNK_SIZE = 64 * 1024 * 1024
RANGE_WORKERS = 8
RETRIES = 8


def log(message: str) -> None:
    print(message, flush=True)


def marker_for(path: str) -> Path:
    return VERIFIED_ROOT / f"{path.replace('/', '__')}.ok"


def part_for(path: str) -> Path:
    return PARTS_ROOT / f"{path.replace('/', '__')}.part"


def state_for(path: str) -> Path:
    return PARTS_ROOT / f"{path.replace('/', '__')}.part.state.json"


def url_for(path: str) -> str:
    return f"{BASE_URL}/{quote(path, safe='/')}?download=true"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest() -> dict:
    if MANIFEST_PATH.exists() and MANIFEST_PATH.stat().st_size:
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    response = requests.get(API_URL, timeout=(30, 120))
    response.raise_for_status()
    files = []
    for item in response.json():
        if item.get("type") != "file":
            continue
        files.append({
            "path": item["path"],
            "size": int(item.get("size", 0)),
            "sha256": item.get("lfs", {}).get("oid"),
        })
    manifest = {
        "dataset": "cactuslab/IDSpace",
        "revision": REVISION,
        "files": files,
        "file_count": len(files),
        "total_bytes": sum(item["size"] for item in files),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def verified_destination(item: dict) -> bool:
    path = item["path"]
    destination = FILES_ROOT / path
    expected_size = int(item["size"])
    expected_sha = item.get("sha256") or ""
    if not destination.exists() or destination.stat().st_size != expected_size:
        return False
    if not expected_sha:
        return True
    marker = marker_for(path)
    if marker.exists() and marker.read_text(encoding="utf-8").strip() == expected_sha:
        return True
    actual_sha = sha256_file(destination)
    if actual_sha != expected_sha:
        return False
    VERIFIED_ROOT.mkdir(parents=True, exist_ok=True)
    marker.write_text(expected_sha + "\n", encoding="utf-8")
    return True


def fetch_range(url: str, part: Path, start: int, end: int) -> int:
    expected = end - start + 1
    headers = {
        "Range": f"bytes={start}-{end}",
        "Accept-Encoding": "identity",
        "User-Agent": "diligram-idspace-range/1.0",
    }
    last_error: Exception | None = None
    for attempt in range(1, RETRIES + 1):
        try:
            with requests.get(url, headers=headers, stream=True, timeout=(30, 120), allow_redirects=True) as response:
                content_range = response.headers.get("Content-Range", "")
                if response.status_code != 206 or not content_range.startswith(f"bytes {start}-{end}/"):
                    raise RuntimeError(f"unexpected range response status={response.status_code} content_range={content_range!r}")
                received = 0
                with part.open("r+b", buffering=0) as handle:
                    handle.seek(start)
                    for block in response.iter_content(1024 * 1024):
                        if not block:
                            continue
                        handle.write(block)
                        received += len(block)
                if received != expected:
                    raise RuntimeError(f"short range {start}-{end}: got {received}, expected {expected}")
                return received
        except Exception as exc:  # network failures are expected during long downloads
            last_error = exc
            log(f"RANGE_RETRY {start}-{end} attempt={attempt}/{RETRIES} error={type(exc).__name__}: {exc}")
            time.sleep(min(2 * attempt, 15))
    raise RuntimeError(f"range failed {start}-{end}: {last_error}")


def load_state(logical_path: str, part: Path, size: int, ranges: list[tuple[int, int]]) -> set[int]:
    state_path = state_for(logical_path)
    if not state_path.exists() or not part.exists() or part.stat().st_size != size:
        return set()
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("size") != size or state.get("ranges") != [[a, b] for a, b in ranges]:
            return set()
        return {int(index) for index in state.get("done", [])}
    except (OSError, ValueError, TypeError):
        return set()


def save_state(logical_path: str, size: int, ranges: list[tuple[int, int]], done: set[int], state_lock: threading.Lock) -> None:
    state_path = state_for(logical_path)
    payload = {"size": size, "ranges": [[a, b] for a, b in ranges], "done": sorted(done)}
    with state_lock:
        temporary = state_path.with_suffix(state_path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        os.replace(temporary, state_path)


def download_file(item: dict) -> None:
    path = item["path"]
    expected_size = int(item["size"])
    expected_sha = item.get("sha256") or ""
    destination = FILES_ROOT / path
    part = part_for(path)
    state_path = state_for(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    PARTS_ROOT.mkdir(parents=True, exist_ok=True)
    VERIFIED_ROOT.mkdir(parents=True, exist_ok=True)

    if verified_destination(item):
        log(f"SKIP_VERIFIED {path}")
        return
    if destination.exists():
        invalid = destination.with_name(destination.name + f".invalid.{int(time.time())}")
        destination.replace(invalid)

    if expected_size == 0:
        part.write_bytes(b"")
        os.replace(part, destination)
        if expected_sha:
            marker_for(path).write_text(expected_sha + "\n", encoding="utf-8")
        log(f"DONE {path} sha256={expected_sha}")
        return

    ranges = [(start, min(expected_size - 1, start + CHUNK_SIZE - 1)) for start in range(0, expected_size, CHUNK_SIZE)]
    done = load_state(path, part, expected_size, ranges)
    if not done:
        if part.exists():
            part.unlink()
        if state_path.exists():
            state_path.unlink()
        with part.open("wb") as handle:
            handle.truncate(expected_size)
    elif part.stat().st_size != expected_size:
        done = set()
        if part.exists():
            part.unlink()
        if state_path.exists():
            state_path.unlink()
        with part.open("wb") as handle:
            handle.truncate(expected_size)

    log(f"RANGE_DOWNLOAD {path} size={expected_size} chunks={len(ranges)} workers={RANGE_WORKERS} resumed={len(done)}")
    state_lock = threading.Lock()
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(RANGE_WORKERS, len(ranges))) as executor:
        futures = {}
        for index, (start, end) in enumerate(ranges):
            if index in done:
                continue
            futures[executor.submit(fetch_range, url_for(path), part, start, end)] = index
        for future in concurrent.futures.as_completed(futures):
            index = futures[future]
            received = future.result()
            done.add(index)
            save_state(path, expected_size, ranges, done, state_lock)
            log(f"RANGE_DONE {path} chunk={index + 1}/{len(ranges)} bytes={received}")

    actual_sha = sha256_file(part)
    if expected_sha and actual_sha != expected_sha:
        raise RuntimeError(f"SHA256_MISMATCH {path} expected={expected_sha} actual={actual_sha}")
    os.replace(part, destination)
    if state_path.exists():
        state_path.unlink()
    if expected_sha:
        marker_for(path).write_text(actual_sha + "\n", encoding="utf-8")
    log(f"DONE {path} sha256={actual_sha}")


def main() -> int:
    for directory in (DATASET_ROOT, FILES_ROOT, PARTS_ROOT, VERIFIED_ROOT, LOG_ROOT):
        directory.mkdir(parents=True, exist_ok=True)
    with LOCK_PATH.open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            log("another IDSpace download process holds the lock")
            return 2
        manifest = load_manifest()
        for item in manifest["files"]:
            for attempt in range(1, 4):
                try:
                    download_file(item)
                    break
                except Exception as exc:
                    log(f"FILE_RETRY {item['path']} attempt={attempt}/3 error={type(exc).__name__}: {exc}")
                    if attempt == 3:
                        raise
                    part = part_for(item["path"])
                    state = state_for(item["path"])
                    if part.exists():
                        part.unlink()
                    if state.exists():
                        state.unlink()
        COMPLETED_PATH.write_text(time.strftime("%Y-%m-%dT%H:%M:%S%z") + "\n", encoding="utf-8")
        log(f"IDSpace download completed: {DATASET_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
