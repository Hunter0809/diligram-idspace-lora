# Diligram IDSpace LoRA

Reproducible scripts for the Xiaomi-OCR-0 IDSpace identity-card LoRA experiment.

## Scope

- ID card images only: template, scan, and mobile images after documented filtering.
- Group-aware train/validation/test splitting before train-only shuffling.
- LoRA injection into language-model projections, including Linear Attention projections.
- Per-micro-step telemetry and a local live dashboard.
- Post-training validation/test evaluation using the final adapter.

## Files

- idspace_download.sh and idspace_parallel_download.py: dataset download helpers.
- idspace_e3_prepare.py: filtering, grouping, split generation, and integrity checks.
- idspace_e3_train.py: Xiaomi-OCR-0 LoRA SFT with gradient accumulation and telemetry.
- idspace_e3_evaluate.py: validation/test token-loss evaluation after training.
- idspace_dashboard.py: local live dashboard with per-metric charts, 100-step EMA loss, and PNG export.
- idspace_e3_lora_overhead.py: adapter parameter and latency measurement.
- idspace_e3_smoke.py: smoke checks.
- control/remote_4090.py: remote execution helper; password is read from the local diligram environment variable.

## E3 configuration

- Epochs: 1
- Per-GPU batch size: 2
- Gradient accumulation: 4
- Effective batch size: 8 on one GPU
- Max pixels: 65,536
- Seed: 42
- Peak learning rate: 1e-4 with cosine schedule and 3% warmup
- Gradient checkpointing: enabled
- KV cache: disabled during training

## Run

Prepare a manifest, then run idspace_e3_train.py. After training, run idspace_e3_evaluate.py against the final adapter. The dashboard reads remote metrics through control/remote_4090.py.

Do not commit datasets, model weights, checkpoints, SSH credentials, or generated metric history.

## Metrics

Training JSONL records raw micro-batch token loss, learning rate, pre-clip gradient norm, stage timings, throughput, token counts, sequence length, image pixel/grid counts, and CUDA allocated/reserved peak memory. Validation and test summaries are written as evaluation_summary.json after the adapter is complete.
