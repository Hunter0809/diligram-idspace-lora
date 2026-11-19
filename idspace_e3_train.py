#!/usr/bin/env python3
"""Two-GPU supervised LoRA training for the prepared IDSpace E3 manifest."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from PIL import Image
from peft import LoraConfig, TaskType, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Dataset
from transformers import AutoProcessor, get_cosine_schedule_with_warmup
import transformers


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
  \"document_type\": null,
  \"issuing_country\": null,
  \"given_name\": null,
  \"surname\": null,
  \"date_of_birth\": null,
  \"gender\": null,
  \"nationality\": null,
  \"personal_identification_number\": null,
  \"document_number\": null,
  \"issue_date\": null,
  \"expiry_date\": null,
  \"place_of_issue\": null
}"""


class JsonlDataset(Dataset):
    def __init__(self, path: Path, root: Path):
        self.root = root
        self.rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        return self.rows[index]


def make_loader(dataset: Dataset, batch_size: int, processor, manifest: Path, *, world: int, rank: int,
                shuffle: bool, seed: int, drop_last: bool) -> tuple[DataLoader, DistributedSampler | None]:
    sampler = None
    if world > 1:
        sampler = DistributedSampler(
            dataset, num_replicas=world, rank=rank, shuffle=shuffle, seed=seed, drop_last=drop_last
        )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle if sampler is None else False,
        drop_last=drop_last,
        collate_fn=lambda batch: collate_batch(batch, processor, manifest, 65536),
    )
    return loader, sampler


def collate_batch(batch: list[dict], processor, root: Path, max_pixels: int) -> dict[str, torch.Tensor]:
    images = [Image.open(root / row["image"]).convert("RGB") for row in batch]
    user_texts = []
    full_texts = []
    prompt_lengths = []
    for index, row in enumerate(batch):
        target = json.dumps(row["target"], ensure_ascii=False, separators=(",", ":"))
        user = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
        full = user + [{"role": "assistant", "content": [{"type": "text", "text": target}]}]
        user_text = processor.apply_chat_template(user, tokenize=False, add_generation_prompt=False)
        full_text = processor.apply_chat_template(full, tokenize=False, add_generation_prompt=False)
        user_texts.append(user_text)
        full_texts.append(full_text)
        prompt_lengths.append(processor(text=user_text, images=[images[index]], return_tensors="pt",
                                        min_pixels=min(65536, max_pixels), max_pixels=max_pixels)["input_ids"].shape[1])
    inputs = processor(text=full_texts, images=images, return_tensors="pt", padding=True,
                       min_pixels=min(65536, max_pixels), max_pixels=max_pixels)
    labels = inputs["input_ids"].clone()
    for index, prompt_length in enumerate(prompt_lengths):
        labels[index, :prompt_length] = -100
    labels[inputs["attention_mask"] == 0] = -100
    inputs["labels"] = labels
    return inputs


def evaluate_split(model, loader, device, split, epoch, optimizer_step):
    model.eval()
    loss_sum = 0.0
    valid_tokens = 0
    sample_count = 0
    batch_count = 0
    started = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            output = model(**batch)
            tokens = int((batch["labels"] != -100).sum().item())
            loss_sum += float(output.loss.detach().item()) * tokens
            valid_tokens += tokens
            sample_count += int(batch["input_ids"].shape[0])
            batch_count += 1
    if dist.is_available() and dist.is_initialized():
        totals = torch.tensor([loss_sum, valid_tokens, sample_count, batch_count], dtype=torch.float64, device=device)
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        loss_sum, valid_tokens, sample_count, batch_count = totals.tolist()
    result = {
        "schema": "eval_v1",
        "timestamp": time.time(),
        "epoch": epoch,
        "optimizer_step": optimizer_step,
        "split": split,
        "loss": loss_sum / max(valid_tokens, 1),
        "valid_label_tokens": int(valid_tokens),
        "samples": int(sample_count),
        "batches": int(batch_count),
        "seconds": time.perf_counter() - started,
    }
    model.train()
    return result


def setup_distributed() -> tuple[int, int, torch.device]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group("nccl")
        rank = int(os.environ["RANK"])
        local = int(os.environ.get("LOCAL_RANK", rank))
        return rank, int(os.environ["WORLD_SIZE"]), torch.device("cuda", local)
    return 0, 1, torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_path: Path):
    loader = getattr(transformers, "AutoModelForImageTextToText", None)
    if loader is not None:
        try:
            return loader.from_pretrained(model_path, local_files_only=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
        except (ValueError, OSError, TypeError):
            pass
    return transformers.AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-pixels", type=int, default=262144)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=0)
    args = parser.parse_args()
    rank, world, device = setup_distributed()
    torch.manual_seed(args.seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + rank)
    args.output.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        (args.output / "config.json").write_text(json.dumps(vars(args), default=str, indent=2), encoding="utf-8")
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    if hasattr(processor, "image_processor"):
        processor.image_processor.max_pixels = args.max_pixels
        processor.image_processor.min_pixels = min(65536, args.max_pixels)
    model = load_model(args.model)
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
        if isinstance(module, torch.nn.Linear)
        and name.endswith(suffixes)
        and not any(x in name.lower() for x in ("vision", "visual", "projector", "merger"))
    ]
    if not targets:
        raise RuntimeError("No language-model LoRA target modules found")
    linear_attention_counts = {
        suffix: sum(name.endswith(suffix) for name in targets)
        for suffix in linear_attention_suffixes
    }
    if any(count != 18 for count in linear_attention_counts.values()):
        raise RuntimeError(f"Expected 18 Linear Attention modules per projection, got {linear_attention_counts}")
    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, target_modules=targets,
        task_type=TaskType.CAUSAL_LM, bias="none",
    )
    model = get_peft_model(model, lora)
    if rank == 0:
        print(json.dumps({
            "target_module_count": len(targets),
            "linear_attention_counts": linear_attention_counts,
            "targets": targets[:8],
        }, ensure_ascii=False), flush=True)
        model.print_trainable_parameters()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.to(device)
    if world > 1:
        model = DDP(model, device_ids=[device.index], find_unused_parameters=False)
    raw_model = model.module if isinstance(model, DDP) else model
    train = JsonlDataset(args.manifest / "train.jsonl", args.manifest)
    validation = JsonlDataset(args.manifest / "validation.jsonl", args.manifest)
    test = JsonlDataset(args.manifest / "test.jsonl", args.manifest)
    loader, sampler = make_loader(
        train, args.batch_size, processor, args.manifest,
        world=world, rank=rank, shuffle=True, seed=args.seed, drop_last=True,
    )
    validation_loader, validation_sampler = make_loader(
        validation, args.batch_size, processor, args.manifest,
        world=world, rank=rank, shuffle=False, seed=args.seed, drop_last=False,
    )
    test_loader, test_sampler = make_loader(
        test, args.batch_size, processor, args.manifest,
        world=world, rank=rank, shuffle=False, seed=args.seed, drop_last=False,
    )
    if rank == 0:
        print(json.dumps({
            "split_sizes": {"train": len(train), "validation": len(validation), "test": len(test)},
            "train_shuffle": "per_epoch",
            "validation_shuffle": False,
            "test_shuffle": False,
            "seed": args.seed,
        }, ensure_ascii=False), flush=True)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in raw_model.parameters() if parameter.requires_grad),
        lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01,
    )
    total_steps = max(1, (len(loader) * args.epochs + args.grad_accum - 1) // args.grad_accum)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=max(1, int(total_steps * 0.03)), num_training_steps=total_steps
    )
    metrics_path = args.output / "metrics.jsonl"
    evaluation_path = args.output / "evaluation.jsonl"
    if rank == 0:
        trainable_params = sum(parameter.numel() for parameter in raw_model.parameters() if parameter.requires_grad)
        metrics_context = {
            "run": str(args.output),
            "started_at": time.time(),
            "use_cache": bool(getattr(model.config, "use_cache", False)),
            "gradient_checkpointing": bool(getattr(raw_model, "is_gradient_checkpointing", False)),
            "batch_size_per_gpu": args.batch_size,
            "grad_accum": args.grad_accum,
            "world_size": world,
            "max_pixels": args.max_pixels,
            "epochs": args.epochs,
            "target_module_count": len(targets),
            "linear_attention_counts": linear_attention_counts,
            "trainable_params": trainable_params,
            "causal_conv1d": "installed",
            "flash_linear_attention": "installed",
            "metrics_schema": "v3",
            "effective_batch_size": args.batch_size * args.grad_accum * world,
            "records_per_micro_step": True,
        }
        (args.output / "metrics_context.json").write_text(json.dumps(metrics_context, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        metrics_path.write_text("", encoding="utf-8")
        evaluation_path.write_text("", encoding="utf-8")
    if world > 1:
        dist.barrier()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    step = 0
    micro_step = 0
    accumulated_loss = 0.0
    accumulated_count = 0
    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        for index, batch in enumerate(loader):
            micro_step += 1
            micro_started = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(device)
            batch = {key: value.to(device) for key, value in batch.items()}
            data_ready = time.perf_counter()
            model_loss = model(**batch).loss
            raw_micro_loss = float(model_loss.detach().item())
            accumulated_loss += raw_micro_loss
            accumulated_count += 1
            (model_loss / args.grad_accum).backward()
            forward_backward_done = time.perf_counter()
            update_now = (index + 1) % args.grad_accum == 0 or index + 1 == len(loader)
            grad_norm_value = None
            optimizer_seconds = 0.0
            optimizer_step = step
            if update_now:
                grad_norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
                grad_norm_value = float(grad_norm.detach().item())
                optimizer_started = time.perf_counter()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_seconds = time.perf_counter() - optimizer_started
                step += 1
                optimizer_step = step
                accumulated_loss = 0.0
                accumulated_count = 0
            micro_seconds = time.perf_counter() - micro_started
            attention_tokens = int(batch["attention_mask"].sum().item())
            label_tokens = int((batch["labels"] != -100).sum().item())
            pixel_elements = int(batch["pixel_values"].numel()) if "pixel_values" in batch else 0
            image_grid_tokens = int(batch["image_grid_thw"].prod(dim=-1).sum().item()) if "image_grid_thw" in batch else 0
            metric = {
                "schema": "v3",
                "timestamp": time.time(),
                "epoch": epoch + 1,
                "micro_step": micro_step,
                "optimizer_step": optimizer_step,
                "optimizer_step_applied": update_now,
                "loss": raw_micro_loss,
                "loss_accumulation_scale": args.grad_accum,
                "learning_rate": float(scheduler.get_last_lr()[0]),
                "grad_norm_preclip": grad_norm_value,
                "micro_step_seconds": micro_seconds,
                "data_to_device_seconds": data_ready - micro_started,
                "forward_backward_seconds": forward_backward_done - data_ready,
                "optimizer_seconds": optimizer_seconds,
                "samples": len(batch["input_ids"]),
                "sequence_length": int(batch["input_ids"].shape[1]),
                "attention_tokens": attention_tokens,
                "label_tokens": label_tokens,
                "pixel_elements": pixel_elements,
                "image_grid_tokens": image_grid_tokens,
                "samples_per_second": len(batch["input_ids"]) / max(micro_seconds, 1e-9),
                "tokens_per_second": attention_tokens / max(micro_seconds, 1e-9),
                "peak_memory_allocated_mb": torch.cuda.max_memory_allocated(device) / 1024**2 if torch.cuda.is_available() else 0.0,
                "peak_memory_reserved_mb": torch.cuda.max_memory_reserved(device) / 1024**2 if torch.cuda.is_available() else 0.0,
                "use_cache": False,
                "gradient_checkpointing": True,
            }
            if rank == 0:
                with metrics_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(metric, ensure_ascii=False) + "\n")
                (args.output / "progress.json").write_text(json.dumps({"epoch": epoch + 1, "micro_step": micro_step, "optimizer_step": optimizer_step, "loss": raw_micro_loss, "timestamp": metric["timestamp"]}, ensure_ascii=False) + "\n", encoding="utf-8")
                if micro_step == 1 or micro_step % 10 == 0 or update_now:
                    print(json.dumps({"epoch": epoch + 1, "micro_step": micro_step, "optimizer_step": optimizer_step, "loss": raw_micro_loss, "learning_rate": metric["learning_rate"], "micro_step_seconds": micro_seconds, "grad_norm": grad_norm_value}, ensure_ascii=False), flush=True)
            if args.max_steps and step >= args.max_steps:
                break
        model.eval()
        validation_result = evaluate_split(model, validation_loader, device, "validation", epoch + 1, step)
        test_result = evaluate_split(model, test_loader, device, "test", epoch + 1, step)
        model.train()
        if rank == 0:
            with evaluation_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(validation_result, ensure_ascii=False) + "\n")
                stream.write(json.dumps(test_result, ensure_ascii=False) + "\n")
            (args.output / "evaluation_summary.json").write_text(
                json.dumps({"validation": validation_result, "test": test_result}, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(json.dumps({"evaluation": [validation_result, test_result]}, ensure_ascii=False), flush=True)
            raw_model.save_pretrained(args.output / f"checkpoint-epoch-{epoch + 1}")
            processor.save_pretrained(args.output / f"checkpoint-epoch-{epoch + 1}")
    if rank == 0:
        raw_model.save_pretrained(args.output / "adapter")
        processor.save_pretrained(args.output / "adapter")
        if args.max_steps and step >= args.max_steps:
            (args.output / "train.max_steps").write_text(f"{step}\n", encoding="utf-8")
        else:
            (args.output / "train.completed").write_text("completed\n", encoding="utf-8")
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
