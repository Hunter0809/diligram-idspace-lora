#!/usr/bin/env python3
"""Measure the inference and parameter overhead of the IDSpace E3 LoRA setup."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from pathlib import Path

import torch
import transformers
from peft import LoraConfig, TaskType, get_peft_model
from PIL import Image

PROMPT = """Extract identity information from this ID card.

Instructions:
1. Extract only information visible in the image.
2. Do not infer missing information.
3. Preserve original names and identification numbers.
4. Normalize unambiguous dates to YYYY-MM-DD.
5. Return null for missing or unreadable fields.
6. Return only valid JSON.

Required fields:
{
  "document_type": null,
  "issuing_country": null,
  "given_name": null,
  "surname": null,
  "date_of_birth": null,
  "gender": null,
  "nationality": null,
  "personal_identification_number": null,
  "document_number": null,
  "issue_date": null,
  "expiry_date": null,
  "place_of_issue": null
}"""


def load_model(model_path: Path):
    loader = getattr(transformers, "AutoModelForImageTextToText", None)
    if loader is not None:
        try:
            return loader.from_pretrained(model_path, local_files_only=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
        except (ValueError, OSError, TypeError):
            pass
    return transformers.AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)


def target_modules(model) -> list[str]:
    suffixes = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj")
    excluded = ("vision", "visual", "projector", "merger")
    targets = [name for name, module in model.named_modules() if isinstance(module, torch.nn.Linear) and name.endswith(suffixes) and not any(part in name.lower() for part in excluded)]
    counts = {suffix: sum(name.endswith(suffix) for name in targets) for suffix in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj")}
    if any(value != 18 for value in counts.values()):
        raise RuntimeError(f"Unexpected Linear Attention target counts: {counts}")
    return targets


def make_batch(processor, manifest: Path, batch_size: int, max_pixels: int) -> dict[str, torch.Tensor]:
    rows = [json.loads(line) for line in (manifest / "train.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()][:batch_size]
    images = [Image.open(manifest / row["image"]).convert("RGB") for row in rows]
    texts = []
    for row in rows:
        user = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
        target = json.dumps(row["target"], ensure_ascii=False, separators=(",", ":"))
        full = user + [{"role": "assistant", "content": [{"type": "text", "text": target}]}]
        texts.append(processor.apply_chat_template(full, tokenize=False, add_generation_prompt=False))
    return processor(text=texts, images=images, return_tensors="pt", padding=True, min_pixels=min(65536, max_pixels), max_pixels=max_pixels)


def measure(model, batch: dict[str, torch.Tensor], device: torch.device, warmup: int, repeats: int) -> dict:
    model.eval()
    batch = {key: value.to(device) for key, value in batch.items()}
    with torch.inference_mode():
        for _ in range(warmup):
            model(**batch, use_cache=False)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        timings = []
        for _ in range(repeats):
            start = time.perf_counter()
            model(**batch, use_cache=False)
            torch.cuda.synchronize(device)
            timings.append((time.perf_counter() - start) * 1000.0)
    ordered = sorted(timings)
    return {"warmup": warmup, "repeats": repeats, "latency_ms": {"mean": statistics.mean(timings), "median": statistics.median(timings), "p95": ordered[max(0, int(len(ordered) * 0.95) - 1)], "min": min(timings), "max": max(timings)}, "timings_ms": timings, "peak_allocated_mb": torch.cuda.max_memory_allocated(device) / 1024**2, "peak_reserved_mb": torch.cuda.max_memory_reserved(device) / 1024**2}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-pixels", type=int, default=65536)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=12)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the overhead benchmark")
    device = torch.device("cuda", 0)
    processor = transformers.AutoProcessor.from_pretrained(args.model, local_files_only=True)
    batch = make_batch(processor, args.manifest, args.batch_size, args.max_pixels)
    result = {"timestamp": time.time(), "device": torch.cuda.get_device_name(device), "visible_device_index": 0, "batch_size": args.batch_size, "max_pixels": args.max_pixels, "dtype": "bfloat16", "model": str(args.model), "benchmark": "eval_forward_only"}

    base = load_model(args.model).to(device)
    result["baseline"] = {"all_params": sum(parameter.numel() for parameter in base.parameters()), **measure(base, batch, device, args.warmup, args.repeats)}
    del base
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)

    lora_base = load_model(args.model)
    targets = target_modules(lora_base)
    lora = get_peft_model(lora_base, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=targets, task_type=TaskType.CAUSAL_LM, bias="none")).to(device)
    result["lora"] = {"all_params": sum(parameter.numel() for parameter in lora.parameters()), "trainable_params": sum(parameter.numel() for parameter in lora.parameters() if parameter.requires_grad), "target_module_count": len(targets), "linear_attention_target_count": sum(name.endswith(("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj")) for name in targets), **measure(lora, batch, device, args.warmup, args.repeats)}
    result["overhead"] = {"added_params": result["lora"]["all_params"] - result["baseline"]["all_params"], "trainable_fraction_percent": 100.0 * result["lora"]["trainable_params"] / result["lora"]["all_params"], "bf16_parameter_storage_mb": result["lora"]["trainable_params"] * 2 / 1024**2, "latency_mean_delta_percent": 100.0 * (result["lora"]["latency_ms"]["mean"] / result["baseline"]["latency_ms"]["mean"] - 1.0), "latency_median_delta_percent": 100.0 * (result["lora"]["latency_ms"]["median"] / result["baseline"]["latency_ms"]["median"] - 1.0), "peak_allocated_delta_mb": result["lora"]["peak_allocated_mb"] - result["baseline"]["peak_allocated_mb"], "peak_reserved_delta_mb": result["lora"]["peak_reserved_mb"] - result["baseline"]["peak_reserved_mb"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())