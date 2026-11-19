#!/usr/bin/env python3
"""One-step BF16 LoRA memory smoke test for E3 micro-batch sizing."""

from __future__ import annotations

import gc
import json
from io import BytesIO
from pathlib import Path

import pyarrow.parquet as pq
import torch
import transformers
from peft import LoraConfig, TaskType, get_peft_model
from PIL import Image
from transformers import AutoProcessor


PROMPT = "Extract identity information from this ID card. Return only valid JSON."


def load_model(path: Path):
    loader = getattr(transformers, "AutoModelForImageTextToText", None)
    if loader is not None:
        try:
            return loader.from_pretrained(path, local_files_only=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
        except (ValueError, OSError, TypeError):
            pass
    return transformers.AutoModelForCausalLM.from_pretrained(
        path, local_files_only=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )


def make_batch(processor, rows, batch_size: int, device: torch.device):
    images = [Image.open(BytesIO(rows[i % len(rows)]["image"]["bytes"])).convert("RGB") for i in range(batch_size)]
    user_messages = [[{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}] for _ in images]
    full_messages = [user + [{"role": "assistant", "content": [{"type": "text", "text": '{"document_type":"identity_card"}'}]}] for user in user_messages]
    user_text = processor.apply_chat_template(user_messages, tokenize=False, add_generation_prompt=False)
    full_text = processor.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)
    prompt = processor(text=user_text, images=images, return_tensors="pt", padding=True)
    inputs = processor(text=full_text, images=images, return_tensors="pt", padding=True)
    labels = inputs["input_ids"].clone()
    for index in range(batch_size):
        labels[index, : prompt["input_ids"].shape[1]] = -100
    labels[inputs["attention_mask"] == 0] = -100
    inputs["labels"] = labels
    return {key: value.to(device) for key, value in inputs.items()}


def main() -> int:
    root = Path("/home/mr_jiang/qkr")
    model_path = root / "models/xiaomi_ocr0/Xiaomi-OCR-0"
    parquet = root / "tmp/idspace_preview/ALB_template.parquet"
    rows = pq.read_table(parquet).slice(0, 4).to_pylist()
    device = torch.device("cuda:0")
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    model = load_model(model_path)
    model.config.use_cache = False
    linear_attention_suffixes = (
        "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj",
    )
    suffixes = (
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
        *linear_attention_suffixes,
    )
    targets = [
        name for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear) and name.endswith(suffixes)
        and not any(x in name.lower() for x in ("vision", "visual", "projector", "merger"))
    ]
    linear_attention_counts = {
        suffix: sum(name.endswith(suffix) for name in targets)
        for suffix in linear_attention_suffixes
    }
    if any(count != 18 for count in linear_attention_counts.values()):
        raise RuntimeError(f"Expected 18 Linear Attention modules per projection, got {linear_attention_counts}")
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=targets, task_type=TaskType.CAUSAL_LM, bias="none"))
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.to(device).train()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4)
    results = []
    for batch_size in (1, 2, 4):
        try:
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            batch = make_batch(processor, rows, batch_size, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(**batch).loss
            loss.backward()
            optimizer.zero_grad(set_to_none=True)
            peak = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            results.append({"batch_size": batch_size, "status": "ok", "loss": float(loss.detach()), "peak_gib": round(peak, 2)})
        except torch.cuda.OutOfMemoryError as exc:
            results.append({"batch_size": batch_size, "status": "oom", "error": str(exc)[:200]})
            torch.cuda.empty_cache()
            break
        except Exception as exc:
            results.append({"batch_size": batch_size, "status": "error", "error": f"{type(exc).__name__}: {exc}"[:400]})
            break
    print(json.dumps({
        "results": results,
        "gpu": torch.cuda.get_device_name(device),
        "target_count": len(targets),
        "linear_attention_counts": linear_attention_counts,
    }, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
