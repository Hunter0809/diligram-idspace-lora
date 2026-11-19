#!/usr/bin/env python3
"""Prepare the IDSpace positive National-ID subset for E3 LoRA training."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import tarfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

COUNTRIES = ("ALB", "AZE", "ESP", "EST", "FIN", "GRC", "LVA", "RUS", "SRB", "SVK")
FIELDS = (
    "document_type", "issuing_country", "given_name", "surname", "date_of_birth",
    "gender", "nationality", "personal_identification_number", "document_number", "card_num", "personal_num", "birthday", "expire_date", "issue_authority",
    "issue_date", "expiry_date", "place_of_issue",
)
FIELD_MAP = {
    "name": "given_name", "given_name": "given_name", "given_name_aze": "given_name", "given_name_en": "given_name", "given_name_english": "given_name", "given_name_greek": "given_name", "surname": "surname", "surname_aze": "surname", "surname_en": "surname", "surname_english": "surname", "surname_greek": "surname",
    "gender": "gender", "nationality": "nationality", "id_number": "personal_identification_number",
    "personal_identification_number": "personal_identification_number", "personal_num": "personal_identification_number", "number": "document_number",
    "document_number": "document_number", "card_num": "document_number", "card_num1": "document_number", "birth_date": "date_of_birth", "date_of_birth": "date_of_birth", "birthday": "date_of_birth",
    "issue_date": "issue_date", "expiry_date": "expiry_date", "expire_date": "expiry_date", "issue_place": "place_of_issue",
    "place_of_issue": "place_of_issue", "issue_authority": "place_of_issue",
}
DATE_RE = re.compile(r"^(\d{1,2})[./-](\d{1,2})[./-](\d{4})$")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
SPLITS = ("train", "validation", "test")
PREPARE_VERSION = "idspace-group-aware-split-v2"


def normalize_date(value: Any) -> str | None:
    if value is None or value == "":
        return None
    text = str(value).strip()
    match = DATE_RE.match(text)
    if match:
        day, month, year = (int(x) for x in match.groups())
        if 1 <= month <= 12 and 1 <= day <= 31:
            return f"{year:04d}-{month:02d}-{day:02d}"
    return text


def value_of(value: Any) -> Any:
    return value.get("value") if isinstance(value, dict) and "value" in value else value


def is_record(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    keys = {str(k).lower() for k in value}
    return bool(keys & (set(FIELD_MAP) | {"fraud", "document_type", "filename", "file_name"}))


def flatten_records(obj: Any, prefix: str = "") -> list[tuple[str, dict[str, Any]]]:
    out: list[tuple[str, dict[str, Any]]] = []
    if isinstance(obj, dict):
        if is_record(obj):
            out.append((prefix, obj))
        else:
            for key, value in obj.items():
                child = f"{prefix}/{key}" if prefix else str(key)
                out.extend(flatten_records(value, child))
    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            out.extend(flatten_records(value, f"{prefix}/{index}" if prefix else str(index)))
    return out


def stem_key(name: str) -> str:
    text = Path(name).name.lower()
    text = re.sub(r"^scanned_\d+_", "", text)
    text = re.sub(r"\.(jpg|jpeg|png|webp)$", "", text)
    return re.sub(r"^generated\.photos_v3_", "", text)


def document_id(name: str) -> str:
    return stem_key(name)


def template_id(country: str, annotation: dict[str, Any]) -> str:
    lowered = {str(key).lower(): value for key, value in annotation.items()}
    for key in ("template_id", "template", "template_name", "layout_id", "document_template"):
        value = value_of(lowered.get(key))
        if value not in (None, ""):
            return str(value).strip()
    return f"{country}_identity_card"


def annotation_payload(annotation: dict[str, Any], country: str) -> dict[str, Any] | None:
    lowered = {str(k).lower(): v for k, v in annotation.items()}
    fraud = str(value_of(lowered.get("fraud", "False"))).lower()
    if fraud in {"true", "1", "yes"}:
        return None
    doc_type = " ".join(str(value_of(lowered[k])) for k in lowered if k in {"document_type", "type", "doctype"}).lower()
    if any(word in doc_type for word in ("passport", "driver", "residence")):
        return None
    keys = set(lowered)
    if not ((keys & {"surname", "surname_aze", "surname_en", "surname_english", "surname_greek"}) and (keys & {"given_name", "given_name_aze", "given_name_en", "given_name_english", "given_name_greek"})):
        return None
    if not ((keys & {"surname", "surname_aze", "surname_en", "surname_english", "surname_greek"}) and (keys & {"given_name", "given_name_aze", "given_name_en", "given_name_english", "given_name_greek"})):
        return None
    result: dict[str, Any] = {field: None for field in FIELDS}
    result["document_type"] = "identity_card"
    result["issuing_country"] = country
    for source, target in FIELD_MAP.items():
        if source not in lowered:
            continue
        value = value_of(lowered[source])
        if target in {"date_of_birth", "issue_date", "expiry_date"}:
            value = normalize_date(value)
        elif value is not None:
            value = str(value)
        result[target] = value
    return result


def find_meta_member(tf: tarfile.TarFile) -> tarfile.TarInfo:
    for member in tf:
        name = member.name.replace("\\", "/").lower()
        if "original_annotation.json" in name:
            return member
    raise RuntimeError("no original annotation JSON in template archive")


def load_country_annotations(archive: Path, country: str) -> dict[str, dict[str, Any]]:
    with tarfile.open(archive, "r:gz") as tf:
        member = find_meta_member(tf)
        handle = tf.extractfile(member)
        if handle is None:
            raise RuntimeError(f"cannot read annotation for {country}")
        raw = json.load(handle)
    records: dict[str, dict[str, Any]] = {}
    for source_key, annotation in flatten_records(raw):
        payload = annotation_payload(annotation, country)
        if payload is None:
            continue
        keys = [source_key, str(annotation.get("filename", "")), str(annotation.get("file_name", ""))]
        for key in keys:
            if key and key != "None":
                records[stem_key(key)] = payload
    if not records:
        raise RuntimeError(f"no identity-card annotations found for {country}")
    return records

def positive_members(archive: Path) -> list[str]:
    result: list[str] = []
    with tarfile.open(archive, "r:gz") as tf:
        for member in tf:
            name = member.name.replace("\\", "/")
            if "/positive/" in f"/{name}" and Path(name).suffix.lower() in IMAGE_EXTS:
                result.append(name)
    return result


def mobile_members(archive: Path) -> list[str]:
    result: list[str] = []
    with tarfile.open(archive, "r:") as tf:
        for member in tf:
            name = member.name.replace("\\", "/")
            if Path(name).suffix.lower() in IMAGE_EXTS and not name.endswith("/"):
                result.append(name)
    return result


def mobile_country(member: str) -> str | None:
    parts = {part.upper() for part in Path(member).parts}
    for country in COUNTRIES:
        if country in parts or Path(member).stem.upper().startswith(country):
            return country
    return None


def build_candidates(root: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]]]:
    candidates: list[dict[str, Any]] = []
    stats: dict[str, dict[str, int]] = {}
    annotation_cache: dict[str, dict[str, dict[str, Any]]] = {}
    for country in COUNTRIES:
        template = root / "files" / f"{country}_template.tar.gz"
        scan = root / "files" / f"{country}_scan.tar.gz"
        if not template.is_file() or not scan.is_file():
            raise RuntimeError(f"missing complete archives for {country}")
        annotations = load_country_annotations(template, country)
        annotation_cache[country] = annotations
        local = Counter()
        for modality, archive in (("template", template), ("scan", scan)):
            for member in positive_members(archive):
                payload = annotations.get(stem_key(member))
                if payload is None:
                    continue
                candidates.append({
                    "country": country,
                    "modality": modality,
                    "member": member,
                    "image": f"images/{country}/{modality}/{Path(member).name}",
                    "archive": f"{country}_{modality}.tar.gz",
                    "template_id": template_id(country, annotations[stem_key(member)]),
                    "document_id": document_id(member),
                    "target": payload,
                })
                local[modality] += 1
        stats[country] = dict(local)
    mobile = root / "files" / "Mobile_images.tar"
    if not mobile.is_file():
        raise RuntimeError("missing complete mobile archive")
    for member in mobile_members(mobile):
        country = mobile_country(member)
        if country is None:
            continue
        annotations = annotation_cache[country]
        payload = annotations.get(stem_key(member))
        if payload is None:
            continue
        candidates.append({
            "country": country,
            "modality": "mobile",
            "member": member,
            "image": f"images/{country}/mobile/{Path(member).name}",
            "archive": "Mobile_images.tar",
            "template_id": template_id(country, annotations[stem_key(member)]),
            "document_id": document_id(member),
            "target": payload,
        })
        stats.setdefault(country, {})["mobile"] = stats.setdefault(country, {}).get("mobile", 0) + 1
    return candidates, stats


def split_name(group_key: tuple[str, str, str], seed: int) -> str:
    value = hashlib.sha256((f"{seed}:" + "\x1f".join(group_key)).encode()).digest()[0] / 255
    return "train" if value < 0.80 else "validation" if value < 0.90 else "test"


def extract_selected(root: Path, rows: list[dict[str, Any]], output: Path) -> None:
    grouped: dict[Path, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[root / "files" / row["archive"]].append(row)
    for archive, archive_rows in grouped.items():
        wanted = {row["member"]: row for row in archive_rows}
        mode = "r:" if archive.suffix == ".tar" else "r:gz"
        with tarfile.open(archive, mode) as tf:
            for member in tf:
                row = wanted.get(member.name.replace("\\", "/"))
                if row is None:
                    continue
                source = tf.extractfile(member)
                if source is None:
                    continue
                destination = output / row["image"]
                destination.parent.mkdir(parents=True, exist_ok=True)
                with destination.open("wb") as handle:
                    handle.write(source.read())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    marker = args.output / "prepare.completed"
    if marker.is_file() and marker.read_text(encoding="utf-8").strip() == PREPARE_VERSION:
        print(f"already prepared: {args.output}")
        return 0
    if marker.is_file():
        backup = args.output.with_name(f"{args.output.name}.backup-{int(time.time())}")
        shutil.move(str(args.output), str(backup))
    args.output.mkdir(parents=True, exist_ok=True)
    candidates, stats = build_candidates(args.root)
    by_group: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_group[(row["country"], row["template_id"], row["document_id"])].append(row)
    selected_groups = set(by_group)
    selected = [row for key in sorted(selected_groups) for row in by_group[key]]
    for row in selected:
        row["split"] = split_name((row["country"], row["template_id"], row["document_id"]), args.seed)
    extract_selected(args.root, selected, args.output)
    counts = Counter()
    for row in selected:
        image = args.output / row["image"]
        if not image.is_file() or image.stat().st_size == 0:
            raise RuntimeError(f"missing extracted image: {image}")
        counts[(row["split"], row["country"], row["modality"])] += 1
    split_groups: dict[str, set[tuple[str, str, str]]] = {split: set() for split in SPLITS}
    for row in selected:
        split_groups[row["split"]].add((row["country"], row["template_id"], row["document_id"]))
    if any(
        split_groups[left] & split_groups[right]
        for index, left in enumerate(SPLITS)
        for right in SPLITS[index + 1:]
    ):
        raise RuntimeError("group leakage detected between train, validation, and test")
    for split in SPLITS:
        with (args.output / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            rows = sorted(
                (row for row in selected if row["split"] == split),
                key=lambda row: (
                    row["country"], row["template_id"], row["document_id"], row["modality"], row["member"]
                ),
            )
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    report = {
        "schema": list(FIELDS), "countries": list(COUNTRIES), "selected_rows": len(selected),
        "stats_from_archives": stats, "counts": {"|".join(key): value for key, value in sorted(counts.items())},
        "group_counts": {split: len(split_groups[split]) for split in SPLITS},
        "policy": {"positive_only": True, "fraud_excluded": True, "mobile_excluded": True, "models_excluded": True,
                   "group_key": ["country", "template_id", "document_id"],
                   "group_split": "deterministic sha256 80/10/10",
                   "train_shuffle": "per_epoch",
                   "validation_shuffle": False,
                   "test_shuffle": False,
                   "seed": args.seed,
                   "e3_sampling": "all eligible positive template, scan, and mobile groups"},
    }
    (args.output / "prepare_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    marker.write_text(PREPARE_VERSION + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())















