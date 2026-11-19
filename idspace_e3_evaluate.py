#!/usr/bin/env python3
"""Evaluate the completed IDSpace E3 LoRA adapter on validation and test splits."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from peft import PeftModel
from idspace_e3_train import JsonlDataset, collate_batch, load_model
from transformers import AutoProcessor

def evaluate(model, loader, device, split):
    model.eval(); loss_sum=0.0; tokens=0; samples=0; batches=0; started=time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            batch={k:v.to(device) for k,v in batch.items()}
            out=model(**batch)
            valid=int((batch['labels'] != -100).sum().item())
            loss_sum += float(out.loss.detach().item()) * valid
            tokens += valid; samples += int(batch['input_ids'].shape[0]); batches += 1
    return {'schema':'eval_v1','timestamp':time.time(),'split':split,'loss':loss_sum/max(tokens,1),'valid_label_tokens':tokens,'samples':samples,'batches':batches,'seconds':time.perf_counter()-started}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--manifest',type=Path,required=True); ap.add_argument('--model',type=Path,required=True); ap.add_argument('--adapter',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--batch-size',type=int,default=2); ap.add_argument('--max-pixels',type=int,default=65536); args=ap.parse_args()
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    processor=AutoProcessor.from_pretrained(args.model,local_files_only=True)
    base=load_model(args.model); base.config.use_cache=False
    model=PeftModel.from_pretrained(base,args.adapter,local_files_only=True).to(device)
    results=[]
    for split in ('validation','test'):
        ds=JsonlDataset(args.manifest/f'{split}.jsonl',args.manifest)
        loader=DataLoader(ds,batch_size=args.batch_size,shuffle=False,drop_last=False,collate_fn=lambda b: collate_batch(b,processor,args.manifest,args.max_pixels))
        results.append(evaluate(model,loader,device,split))
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'evaluation.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in results),encoding='utf-8')
    (args.output/'evaluation_summary.json').write_text(json.dumps({'results':results},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'evaluation':results},ensure_ascii=False),flush=True)
if __name__=='__main__': main()
