#!/usr/bin/env python3
"""Local read-only dashboard for the remote IDSpace LoRA training run."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REMOTE = ROOT / "reproduction" / "mineru421" / "control" / "remote_4090.py"
REMOTE_ROOT = "/home/mr_jiang/qkr/runs/idspace_e3_full_metrics"
POLL_SECONDS = 10
HISTORY_FILE = ROOT / "reproduction" / "idspace_dashboard_history.json"

CACHE = {"status": "starting", "history": [], "latest": None, "context": {}, "gpu": [], "error": None, "updated_at": None}
STOP_EVENT = threading.Event()
LOCK = threading.Lock()

PAGE = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>IDSpace LoRA 训练监控</title>
<style>
:root{--bg:#11161c;--panel:#1a222b;--line:#2b3743;--text:#e8edf2;--muted:#93a1ae;--cyan:#55c7d9;--amber:#e8b45b;--green:#6bd39a;--red:#ed7777}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 system-ui,-apple-system,Segoe UI,sans-serif}main{max-width:1440px;margin:0 auto;padding:24px}.top{display:flex;justify-content:space-between;align-items:flex-start;border-bottom:1px solid var(--line);padding-bottom:18px;margin-bottom:18px}.title{font-size:22px;font-weight:650}.sub{color:var(--muted);margin-top:5px}.status{border:1px solid var(--line);padding:7px 11px;border-radius:6px;color:var(--green)}.status.off{color:var(--red)}.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin-bottom:14px}.card{background:var(--panel);border:1px solid var(--line);border-radius:7px;padding:14px}.label{color:var(--muted);font-size:12px}.value{font-size:23px;font-variant-numeric:tabular-nums;margin-top:4px}.unit{font-size:12px;color:var(--muted);margin-left:3px}.charts{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}.chart-card{background:var(--panel);border:1px solid var(--line);border-radius:7px;padding:12px}.chart-title{font-weight:600;margin-bottom:5px}.chart-meta{color:var(--muted);font-size:12px}.chart{width:100%;height:230px;display:block}.gpu-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px;margin-top:14px}.gpu-name{font-weight:600;color:var(--cyan)}.gpu-line{display:flex;justify-content:space-between;color:var(--muted);padding-top:4px}.gpu-line b{color:var(--text);font-weight:500}.footer{color:var(--muted);font-size:12px;margin-top:15px}@media(max-width:900px){main{padding:14px}.grid{grid-template-columns:repeat(2,1fr)}.charts,.gpu-grid{grid-template-columns:1fr}}@media(max-width:520px){.grid{grid-template-columns:1fr 1fr}.value{font-size:18px}.top{gap:10px;flex-direction:column}}
</style></head><body><main>
<section class="top"><div><div class="title">IDSpace LoRA 训练监控</div><div class="sub">GPU1 · batch 2 · gradient accumulation 4 · 每 10 秒刷新</div></div><div><button id="saveCharts" type="button">保存图表</button><button id="disconnect" type="button">断开监听</button><div id="status" class="status">连接中</div></div></section>
<section class="grid"><div class="card"><div class="label">Optimizer Step</div><div class="value" id="step">-</div></div><div class="card"><div class="label">Micro Step</div><div class="value" id="micro">-</div></div><div class="card"><div class="label">Raw Loss</div><div class="value" id="loss">-</div></div><div class="card"><div class="label">学习率</div><div class="value" id="lr">-</div></div><div class="card"><div class="label">单步耗时</div><div class="value" id="stepsec">-<span class="unit">s</span></div></div><div class="card"><div class="label">Tokens / s</div><div class="value" id="tps">-</div></div><div class="card"><div class="label">峰值显存</div><div class="value" id="mem">-<span class="unit">MB</span></div></div><div class="card"><div class="label">梯度范数</div><div class="value" id="grad">-</div></div></section>
<section class="charts"><div class="chart-card"><div class="chart-title">Loss</div><div class="chart-meta">横轴：micro step · 纵轴：当前 micro-batch 原始 token loss</div><canvas class="chart" id="lossChart" data-key="loss"></canvas></div><div class="chart-card"><div class="chart-title">100-Step EMA Loss</div><div class="chart-meta">横轴：optimizer step · 纵轴：100 步指数移动平均 token loss</div><canvas class="chart" id="emaLossChart"></canvas></div><div class="chart-card"><div class="chart-title">Learning Rate</div><div class="chart-meta">横轴：optimizer step · 纵轴：学习率</div><canvas class="chart" id="lrChart" data-key="learning_rate"></canvas></div><div class="chart-card"><div class="chart-title">Gradient Norm (Pre-Clip)</div><div class="chart-meta">横轴：optimizer step · 纵轴：梯度裁剪前范数</div><canvas class="chart" id="gradChart" data-key="grad_norm_preclip"></canvas></div><div class="chart-card"><div class="chart-title">Micro-step Duration</div><div class="chart-meta">横轴：optimizer step · 纵轴：单个 micro-step 耗时 (s)</div><canvas class="chart" id="microTimeChart" data-key="micro_step_seconds"></canvas></div><div class="chart-card"><div class="chart-title">Data-to-Device Duration</div><div class="chart-meta">横轴：optimizer step · 纵轴：数据搬运耗时 (s)</div><canvas class="chart" id="dataTimeChart" data-key="data_to_device_seconds"></canvas></div><div class="chart-card"><div class="chart-title">Forward + Backward Duration</div><div class="chart-meta">横轴：optimizer step · 纵轴：前向反向耗时 (s)</div><canvas class="chart" id="forwardTimeChart" data-key="forward_backward_seconds"></canvas></div><div class="chart-card"><div class="chart-title">Optimizer Duration</div><div class="chart-meta">横轴：optimizer step · 纵轴：optimizer 更新耗时 (s)</div><canvas class="chart" id="optimTimeChart" data-key="optimizer_seconds"></canvas></div><div class="chart-card"><div class="chart-title">Samples per Second</div><div class="chart-meta">横轴：optimizer step · 纵轴：样本吞吐 (samples/s)</div><canvas class="chart" id="sampleRateChart" data-key="samples_per_second"></canvas></div><div class="chart-card"><div class="chart-title">Tokens per Second</div><div class="chart-meta">横轴：optimizer step · 纵轴：token 吞吐 (tokens/s)</div><canvas class="chart" id="tokenRateChart" data-key="tokens_per_second"></canvas></div><div class="chart-card"><div class="chart-title">Attention Tokens</div><div class="chart-meta">横轴：optimizer step · 纵轴：attention token 数</div><canvas class="chart" id="attentionChart" data-key="attention_tokens"></canvas></div><div class="chart-card"><div class="chart-title">Valid Label Tokens</div><div class="chart-meta">横轴：optimizer step · 纵轴：有效 label token 数</div><canvas class="chart" id="labelChart" data-key="label_tokens"></canvas></div><div class="chart-card"><div class="chart-title">Sequence Length</div><div class="chart-meta">横轴：optimizer step · 纵轴：序列长度 (tokens)</div><canvas class="chart" id="seqChart" data-key="sequence_length"></canvas></div><div class="chart-card"><div class="chart-title">Pixel Elements</div><div class="chart-meta">横轴：optimizer step · 纵轴：图像像素元素数</div><canvas class="chart" id="pixelChart" data-key="pixel_elements"></canvas></div><div class="chart-card"><div class="chart-title">Image Grid Tokens</div><div class="chart-meta">横轴：optimizer step · 纵轴：图像网格 token 数</div><canvas class="chart" id="gridChart" data-key="image_grid_tokens"></canvas></div><div class="chart-card"><div class="chart-title">CUDA Allocated Peak</div><div class="chart-meta">横轴：optimizer step · 纵轴：CUDA allocated 峰值显存 (MB)</div><canvas class="chart" id="allocMemChart" data-key="peak_memory_allocated_mb"></canvas></div><div class="chart-card"><div class="chart-title">CUDA Reserved Peak</div><div class="chart-meta">横轴：optimizer step · 纵轴：CUDA reserved 峰值显存 (MB)</div><canvas class="chart" id="reservedMemChart" data-key="peak_memory_reserved_mb"></canvas></div></section>
<section class="gpu-grid" id="gpus"></section><div class="footer" id="footer">等待远端数据…</div>
</main><script>
const $=id=>document.getElementById(id); const fmt=(x,n=2)=>x==null||Number.isNaN(Number(x))?'-':Number(x).toFixed(n); const esc=s=>String(s).replace(/[&<>"']/g,m=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m]));
function draw(id,series,colors,labels,steps=[],yLabel='') {const c=$(id),d=c.getBoundingClientRect(),ratio=devicePixelRatio||1,w=Math.max(320,d.width),h=230;c.width=w*ratio;c.height=h*ratio;const x=c.getContext('2d');x.scale(ratio,ratio);x.clearRect(0,0,w,h);x.strokeStyle='#2b3743';x.lineWidth=1;for(let i=0;i<5;i++){let y=18+i*(h-42)/4;x.beginPath();x.moveTo(42,y);x.lineTo(w-12,y);x.stroke()}const vals=series.flat().filter(v=>Number.isFinite(v));if(!vals.length){x.fillStyle='#93a1ae';x.fillText('暂无有效数据',50,30);return}let min=Math.min(...vals),max=Math.max(...vals);if(min===max){min-=1;max+=1}const px=i=>42+i*Math.max(1,(w-54)/(Math.max(1,series[0].length-1))),py=v=>h-28-(v-min)*(h-56)/(max-min);x.font='11px system-ui';x.fillStyle='#93a1ae';x.fillText(fmt(max,2),4,20);x.fillText(fmt(min,2),4,h-28);if(steps.length){x.fillText(String(steps[0]??''),42,h-8);x.fillText(String(steps[steps.length-1]??''),w-55,h-8)}x.save();x.translate(11,h/2);x.rotate(-Math.PI/2);x.fillText(yLabel,0,0);x.restore();x.fillText('optimizer step',w/2-35,h-8);series.forEach((a,j)=>{x.strokeStyle=colors[j];x.lineWidth=1.8;x.beginPath();a.forEach((v,i)=>{if(!Number.isFinite(v))return;if(i===0||!Number.isFinite(a[i-1]))x.moveTo(px(i),py(v));else x.lineTo(px(i),py(v))});x.stroke();if(labels&&labels[j]){x.fillStyle=colors[j];x.fillText(labels[j],w-100+j*45,16)}})}
function update(d){
  var status=document.getElementById('status');
  status.textContent=d.status==='active'?'运行中':d.status==='completed'?'已完成':d.status==='stopped'?'已停止':'异常';
  status.className='status '+(d.status==='active'?'':'off');
  var r=d.latest||{};
  $('step').textContent=r.optimizer_step==null?'-':r.optimizer_step;
  $('micro').textContent=r.micro_step==null?'-':r.micro_step;
  $('loss').textContent=fmt(r.loss,4);
  $('lr').textContent=r.learning_rate==null?'-':Number(r.learning_rate).toExponential(3);
  $('stepsec').firstChild.textContent=fmt(r.micro_step_seconds,3);
  $('tps').textContent=fmt(r.tokens_per_second,1);
  $('mem').firstChild.textContent=fmt(r.peak_memory_allocated_mb,0);
  var hist=d.history||[], lastGrad=null;
  for(var gi=hist.length-1;gi>=0;gi--){if(Number.isFinite(hist[gi].grad_norm_preclip)){lastGrad=hist[gi].grad_norm_preclip;break;}}
  $('grad').textContent=fmt(lastGrad,3);
  var grouped={}; hist.forEach(function(item){var step=item.optimizer_step==null?item.micro_step:item.optimizer_step;(grouped[step]||(grouped[step]=[])).push(item);});
  var steps=Object.keys(grouped).map(Number).sort(function(x,y){return x-y;});
  function means(key){return steps.map(function(step){var vals=grouped[step].map(function(x){return Number(x[key]);}).filter(Number.isFinite);return vals.length?vals.reduce(function(x,y){return x+y;},0)/vals.length:NaN;});}
  draw('lossChart',[hist.map(function(x){return Number(x.loss);})],['#55c7d9'],[],hist.map(function(x){return Number(x.micro_step);}),'raw token loss');
  var stepLoss=means('loss'), ema=[], alpha=2/101; stepLoss.forEach(function(v,i){ema.push(i?alpha*v+(1-alpha)*ema[i-1]:v);});
  draw('emaLossChart',[ema],['#6bd39a'],[],steps,'ema loss');
  var plots=[['lrChart','learning_rate','#e8b45b','learning rate'],['gradChart','grad_norm_preclip','#ed7777','grad norm'],['microTimeChart','micro_step_seconds','#55c7d9','seconds'],['dataTimeChart','data_to_device_seconds','#e8b45b','seconds'],['forwardTimeChart','forward_backward_seconds','#6bd39a','seconds'],['optimTimeChart','optimizer_seconds','#ed7777','seconds'],['sampleRateChart','samples_per_second','#6bd39a','samples/s'],['tokenRateChart','tokens_per_second','#55c7d9','tokens/s'],['attentionChart','attention_tokens','#e8b45b','tokens'],['labelChart','label_tokens','#6bd39a','tokens'],['seqChart','sequence_length','#55c7d9','tokens'],['pixelChart','pixel_elements','#e8b45b','elements'],['gridChart','image_grid_tokens','#6bd39a','tokens'],['allocMemChart','peak_memory_allocated_mb','#55c7d9','MB'],['reservedMemChart','peak_memory_reserved_mb','#e8b45b','MB']];
  plots.forEach(function(p){draw(p[0],[means(p[1])],[p[2]],[],steps,p[3]);});
  $('lossmeta').textContent=steps.length+' 个 optimizer step · 横轴为 step';
  $('gpus').innerHTML=(d.gpu||[]).map(function(g){return '<div class="card"><div class="gpu-name">GPU '+esc(g.index)+'</div><div class="gpu-line"><span>显存</span><b>'+fmt(g.used_mb,0)+' / '+fmt(g.total_mb,0)+' MB</b></div><div class="gpu-line"><span>利用率</span><b>'+fmt(g.util,0)+'%</b></div><div class="gpu-line"><span>功耗</span><b>'+fmt(g.power,1)+' W</b></div><div class="gpu-line"><span>温度</span><b>'+fmt(g.temp,0)+' °C</b></div></div>';}).join('');
  $('footer').textContent='最后更新：'+(d.updated_at||'-')+' · 数据源：远端 '+(d.source||'IDSpace E3');
}
async function poll(){try{let r=await fetch('/api/metrics',{cache:'no-store'});update(await r.json())}catch(e){$('status').textContent='连接错误';$('status').className='status off'}}poll();const pollTimer=setInterval(poll,10000);$('disconnect').onclick=async()=>{if(!confirm('断开监听，不会停止训练，确认继续？'))return;await fetch('/api/disconnect',{method:'POST'});clearInterval(pollTimer);$('disconnect').disabled=true;$('status').textContent='监听已断开';$('status').className='status off'};$('saveCharts').onclick=()=>{const canvases=[...document.querySelectorAll('canvas')];const out=document.createElement('canvas');out.width=1200;out.height=canvases.length*260;const ctx=out.getContext('2d');ctx.fillStyle='#11161c';ctx.fillRect(0,0,out.width,out.height);canvases.forEach((c,i)=>ctx.drawImage(c,0,i*260,1200,240));const a=document.createElement('a');a.download='idspace_training_metrics.png';a.href=out.toDataURL('image/png');a.click()};window.addEventListener('resize',poll);
</script></body></html>'''

def parse_gpu(value: str):
    out=[]
    for row in value.split(';'):
        fields=[x.strip() for x in row.split(',')]
        if len(fields)<6: continue
        try: out.append({"index":int(fields[0]),"used_mb":float(fields[1]),"total_mb":float(fields[1])+float(fields[2]),"util":float(fields[3]),"power":float(fields[4]),"temp":float(fields[5])})
        except ValueError: pass
    return out

def remote_snapshot():
    script='''set -e\nroot=/home/mr_jiang/qkr/runs/idspace_e3_full_metrics\nprintf "__STATUS__\\n"\nif pgrep -f "/home/mr_jiang/qkr/reproduction/idspace_e3_train.py.*idspace_e3_full_metrics" >/dev/null 2>&1; then echo active; elif [ -f "$root/train.completed" ]; then echo completed; else echo stopped; fi\nprintf "__CONTEXT__\\n"\ncat "$root/metrics_context.json" 2>/dev/null || echo {}\nprintf "__METRICS__\\n"\ntail -n 500 "$root/metrics.jsonl" 2>/dev/null || true\nprintf "__END__\\n"\nprintf "__GPU__\\n"\nnvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu --format=csv,noheader,nounits\n'''
    with tempfile.NamedTemporaryFile('w', suffix='.sh', delete=False, encoding='utf-8', newline='\n') as f:
        f.write(script); path=Path(f.name)
    try:
        proc=subprocess.run([sys.executable, str(REMOTE), str(path)], cwd=ROOT, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=35)
    finally: path.unlink(missing_ok=True)
    if proc.returncode: raise RuntimeError(proc.stderr.strip() or proc.stdout[-500:])
    text=proc.stdout
    def section(a,b):
        if a not in text:return ''
        s=text.split(a,1)[1]; return s.split(b,1)[0].strip() if b in s else s.strip()
    status_lines=[line.strip() for line in section('__STATUS__','__CONTEXT__').splitlines() if line.strip()]; status=status_lines[-1] if status_lines else 'stopped'
    context_text=section('__CONTEXT__','__METRICS__'); context=json.loads(context_text) if context_text else {}
    records=[]; gpu=[]
    for line in section('__METRICS__','__END__').splitlines():
        try:
            item=json.loads(line)
            if item.get('schema')=='v3': records.append(item)
            elif item.get('gpu'):
                gpu=parse_gpu(item['gpu'])
        except json.JSONDecodeError: pass
    gpu_text=(text.split('__GPU__',1)[1] if '__GPU__' in text else '').splitlines()
    if gpu_text:
        gpu=parse_gpu(';'.join(gpu_text))
    return {"status":status,"context":context,"history":records[-300:],"latest":records[-1] if records else None,"gpu":gpu,"updated_at":time.strftime('%Y-%m-%d %H:%M:%S'),'source':'runs/idspace_e3_full_metrics'}

def poller():
    global CACHE
    try:
        if HISTORY_FILE.exists():
            saved=json.loads(HISTORY_FILE.read_text(encoding='utf-8'))
            CACHE['history']=saved.get('history', [])
    except Exception:
        pass
    while not STOP_EVENT.is_set():
        try:
            data=remote_snapshot()
            with LOCK:
                previous={int(x.get('micro_step')): x for x in CACHE.get('history', []) if x.get('micro_step') is not None}
                for item in data.get('history', []):
                    if item.get('micro_step') is not None:
                        previous[int(item['micro_step'])]=item
                data['history']=[previous[k] for k in sorted(previous)]
                CACHE=data
                try:
                    HISTORY_FILE.write_text(json.dumps({'history': CACHE['history']}, ensure_ascii=False), encoding='utf-8')
                except OSError:
                    pass
        except Exception as exc:
            with LOCK: CACHE={**CACHE,"error":str(exc),"updated_at":time.strftime('%Y-%m-%d %H:%M:%S')}
        STOP_EVENT.wait(POLL_SECONDS)

def disconnect_watch():
    script = "set -eu\ntmux kill-session -t idspace_e3_metrics_watch 2>/dev/null || true\necho remote_watch_stopped\n"
    with tempfile.NamedTemporaryFile('w', suffix='.sh', delete=False, encoding='utf-8', newline='\n') as f:
        f.write(script); path = Path(f.name)
    try:
        subprocess.run(["python", str(REMOTE), str(path)], cwd=ROOT, capture_output=True, text=True, timeout=35)
    finally:
        path.unlink(missing_ok=True)

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.split('?',1)[0]=='/api/metrics':
            with LOCK: body=json.dumps(CACHE,ensure_ascii=False).encode()
            self.send_response(200); self.send_header('Content-Type','application/json; charset=utf-8'); self.send_header('Cache-Control','no-store'); self.send_header('Content-Length',str(len(body))); self.end_headers(); self.wfile.write(body); return
        if self.path.split('?',1)[0] in ('/','/index.html'):
            body=PAGE.encode(); self.send_response(200); self.send_header('Content-Type','text/html; charset=utf-8'); self.send_header('Content-Length',str(len(body))); self.end_headers(); self.wfile.write(body); return
        self.send_error(404)
    def do_POST(self):
        if self.path.split('?', 1)[0] == '/api/disconnect':
            disconnect_watch(); STOP_EVENT.set()
            self.send_response(200); self.send_header('Content-Type', 'application/json; charset=utf-8'); self.end_headers(); self.wfile.write(b'{"ok":true}')
            return
        self.send_error(404)
    def log_message(self,*args): pass

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--host',default='127.0.0.1'); ap.add_argument('--port',type=int,default=8765); args=ap.parse_args()
    threading.Thread(target=poller,daemon=True).start()
    server=ThreadingHTTPServer((args.host,args.port),Handler); print(f'IDSpace dashboard: http://{args.host}:{args.port}',flush=True); server.serve_forever()
if __name__=='__main__': main()