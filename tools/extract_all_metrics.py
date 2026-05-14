#!/usr/bin/env python3
"""Extract ALL metrics from ALL experiment directories."""

import os
import re
import json
import glob
import yaml

BASE = "/scratch/dr/o.iraqy/Urban-ReID/urban-reid-modular/outputs"

EXPERIMENTS = [
    "convnextv2_large_in22k",
    "efficientnet_b5_in1k",
    "efficientnetv2_xl_in1k",
    "eva02_large_448",
    "specialist_trafficsignal_vit_large",
    "swin_base_in22k",
    "swin_base_in22k_freeze50",
    "swin_base_in22k_freeze80",
    "swin_base_in22k_freeze90",
    "swin_large_in22k",
    "swin_large_in22k_freeze50",
    "swin_large_in22k_freeze90",
    "swin_large_in22k_freeze95",
    "swinv2_base_in1k",
    "vit_huge_plus_dinov3",
    "vit_huge_plus_dinov3_full",
    "vit_huge_plus_dinov3_full_freeze50",
    "vit_huge_plus_dinov3_full_freeze80",
    "vit_huge_plus_dinov3_full_freeze95",
    "vit_large_dinov3",
    "vit_large_dinov3_arcface",
    "vit_large_dinov3_arcface2",
    "vit_large_dinov3_aug",
    "vit_large_dinov3_aug2",
    "vit_large_dinov3_aug3",
    "vit_large_dinov3_center",
    "vit_large_dinov3_center2",
    "vit_large_dinov3_circle",
    "vit_large_dinov3_dist",
    "vit_large_dinov3_freeze100",
    "vit_large_dinov3_freeze50",
    "vit_large_dinov3_freeze50_longer",
    "vit_large_dinov3_freeze80",
    "vit_large_dinov3_freeze90",
    "vit_large_dinov3_freeze95",
    "vit_large_dinov3_gblur",
    "vit_large_dinov3_rankedlist",
    "vit_large_dinov3_rankedlist2",
    "vit_large_in1k",
]


def parse_log(log_path):
    """Parse a log_rank0.txt file and extract all evaluation metrics."""
    if not os.path.exists(log_path):
        return None
    
    with open(log_path, 'r') as f:
        content = f.read()
    
    results = {
        'evals': [],
        'best_mAP': None,
        'best_epoch': None,
        'last_epoch': 0,
        'training_complete': False,
        'best_line': None,
    }
    
    # Find all evaluation lines: "Evaluation: mAP=X R1=X R5=X R10=X"
    eval_pattern = re.compile(
        r'Epoch \[(\d+)(?:/(\d+))?\].*?val_mAP=([\d.]+)\s+train_mAP=([\d.]+)\s+gap=([\d.]+)'
    )
    eval_raw_pattern = re.compile(
        r'Evaluation: mAP=([\d.]+)\s+R1=([\d.]+)\s+R5=([\d.]+)\s+R10=([\d.]+)'
    )
    
    # Gather evaluation epochs
    eval_lines = content.split('\n')
    
    i = 0
    while i < len(eval_lines):
        line = eval_lines[i]
        
        # Check for evaluation line
        eval_match = eval_raw_pattern.search(line)
        if eval_match:
            mAP = float(eval_match.group(1))
            r1 = float(eval_match.group(2))
            r5 = float(eval_match.group(3))
            r10 = float(eval_match.group(4))
            
            # Look at next lines for train eval and epoch summary
            train_mAP = None
            epoch = None
            max_epoch = None
            gap = None
            
            for j in range(i+1, min(i+5, len(eval_lines))):
                train_match = re.search(r'Train eval: mAP=([\d.]+)', eval_lines[j])
                if train_match:
                    train_mAP = float(train_match.group(1))
                
                epoch_match = re.search(r'Epoch \[(\d+)(?:/(\d+))?\]\s+val_mAP', eval_lines[j])
                if epoch_match:
                    epoch = int(epoch_match.group(1))
                    if epoch_match.group(2):
                        max_epoch = int(epoch_match.group(2))
                
                # Also check for epoch in the eval line context
                epoch_match2 = re.search(r'Epoch \[(\d+)/(\d+)\]', eval_lines[j])
                if epoch_match2 and epoch is None:
                    epoch = int(epoch_match2.group(1))
                    max_epoch = int(epoch_match2.group(2))
            
            # Also look backward for epoch context
            if epoch is None:
                for j in range(max(0, i-5), i):
                    epoch_match3 = re.search(r'Epoch \[(\d+)/(\d+)\]', eval_lines[j])
                    if epoch_match3:
                        epoch = int(epoch_match3.group(1))
                        max_epoch = int(epoch_match3.group(2))
            
            results['evals'].append({
                'epoch': epoch,
                'val_mAP': mAP,
                'R1': r1,
                'R5': r5,
                'R10': r10,
                'train_mAP': train_mAP,
            })
            
            if max_epoch:
                results['max_epochs_config'] = max_epoch
        
        i += 1
    
    # Find best mAP line
    best_match = re.search(r'Best mAP: ([\d.]+)', content)
    if best_match:
        results['best_mAP'] = float(best_match.group(1))
        results['training_complete'] = True
    
    # Find "New best mAP" lines
    new_best_matches = re.findall(r'New best mAP: ([\d.]+) at epoch (\d+)', content)
    if new_best_matches:
        last_best = new_best_matches[-1]
        if results['best_mAP'] is None:
            results['best_mAP'] = float(last_best[0])
        results['best_epoch'] = int(last_best[1])
    
    # Find last epoch trained
    last_epoch_matches = re.findall(r'Epoch \[(\d+)/(\d+)\]', content)
    if last_epoch_matches:
        results['last_epoch'] = max(int(m[0]) for m in last_epoch_matches)
        results['max_epochs_config'] = int(last_epoch_matches[-1][1])
    
    # Check for per-class mAP
    per_class_pattern = re.compile(r'Per-class mAP.*?Container.*?([\d.]+).*?Crosswalk.*?([\d.]+).*?RubbishBin.*?([\d.]+).*?TrafficSign.*?([\d.]+)', re.DOTALL)
    pc_match = per_class_pattern.search(content)
    if pc_match:
        results['per_class_mAP'] = {
            'Container': float(pc_match.group(1)),
            'Crosswalk': float(pc_match.group(2)),
            'RubbishBins': float(pc_match.group(3)),
            'TrafficSign': float(pc_match.group(4)),
        }
    
    return results


def parse_config(config_path):
    """Parse config.yaml to extract key settings."""
    if not os.path.exists(config_path):
        return None
    
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    info = {}
    
    # Backbone
    info['backbone'] = cfg.get('backbone', {}).get('name', 'unknown')
    
    # Objectives/losses
    objectives = cfg.get('objectives', [])
    info['losses'] = [f"{o['name']}({o['type']})" for o in objectives]
    
    # Solver
    solver = cfg.get('solver', {})
    info['lr'] = solver.get('lr', None)
    info['max_epochs'] = solver.get('max_epochs', None)
    info['optimizer'] = solver.get('optimizer', None)
    info['warmup_epochs'] = solver.get('warmup_epochs', None)
    info['scheduler'] = solver.get('scheduler', None)
    info['amp'] = solver.get('amp', None)
    info['grad_accum'] = solver.get('grad_accum_steps', None)
    
    # Input
    inp = cfg.get('input', {})
    info['size_train'] = inp.get('size_train', None)
    info['autoaug'] = inp.get('autoaug', False)
    info['color_jitter'] = inp.get('color_jitter', False)
    info['random_erasing'] = inp.get('random_erasing', False)
    info['gaussian_blur'] = inp.get('gaussian_blur', False) if 'gaussian_blur' in inp else None
    
    # Freeze
    info['freeze_percent'] = cfg.get('backbone', {}).get('freeze_percent', None)
    if info['freeze_percent'] is None:
        info['freeze_percent'] = cfg.get('backbone', {}).get('freeze_pct', None)
    
    # Dataloader
    dl = cfg.get('dataloader', {})
    info['batch_size'] = dl.get('batch_size', None)
    
    # Head
    head = cfg.get('head', {})
    info['head_type'] = head.get('type', None)
    
    return info


def get_checkpoints(exp_dir):
    """List available checkpoints."""
    ckpts = glob.glob(os.path.join(exp_dir, 'checkpoint_ep*.pth'))
    epochs = sorted([int(re.search(r'ep(\d+)', os.path.basename(c)).group(1)) for c in ckpts])
    has_best = os.path.exists(os.path.join(exp_dir, 'best_model.pth'))
    has_final = os.path.exists(os.path.join(exp_dir, 'final_model.pth'))
    return {
        'checkpoint_epochs': epochs,
        'has_best_model': has_best,
        'has_final_model': has_final,
    }


def get_best_eval(evals):
    """Get the eval entry with the best val_mAP."""
    if not evals:
        return None
    return max(evals, key=lambda x: x['val_mAP'])


# ============================================================
# MAIN
# ============================================================
print("=" * 120)
print("COMPREHENSIVE EXPERIMENT METRICS EXTRACTION")
print("=" * 120)

all_results = {}

for exp_name in EXPERIMENTS:
    exp_dir = os.path.join(BASE, exp_name)
    
    if not os.path.isdir(exp_dir):
        print(f"\n{'='*80}")
        print(f"EXPERIMENT: {exp_name} -- DIRECTORY NOT FOUND")
        continue
    
    log_path = os.path.join(exp_dir, 'log_rank0.txt')
    config_path = os.path.join(exp_dir, 'config.yaml')
    
    log_data = parse_log(log_path)
    config_data = parse_config(config_path)
    ckpt_data = get_checkpoints(exp_dir)
    
    # Check for any extra files
    extra_files = [f for f in os.listdir(exp_dir) if f.endswith(('.json', '.csv', '.npy'))]
    
    print(f"\n{'='*100}")
    print(f"EXPERIMENT: {exp_name}")
    print(f"{'='*100}")
    
    if config_data:
        print(f"  Backbone: {config_data['backbone']}")
        print(f"  Losses: {', '.join(config_data['losses'])}")
        print(f"  LR: {config_data['lr']}, Optimizer: {config_data['optimizer']}, Scheduler: {config_data['scheduler']}")
        print(f"  Max Epochs: {config_data['max_epochs']}, Warmup: {config_data['warmup_epochs']}")
        print(f"  Batch Size: {config_data['batch_size']}, Grad Accum: {config_data['grad_accum']}, AMP: {config_data['amp']}")
        print(f"  Input Size: {config_data['size_train']}")
        print(f"  AutoAug: {config_data['autoaug']}, ColorJitter: {config_data['color_jitter']}, RE: {config_data['random_erasing']}")
        if config_data.get('gaussian_blur'):
            print(f"  Gaussian Blur: {config_data['gaussian_blur']}")
        if config_data.get('freeze_percent') is not None:
            print(f"  Freeze %: {config_data['freeze_percent']}")
    else:
        print("  Config: NOT FOUND")
    
    if log_data:
        best_eval = get_best_eval(log_data['evals'])
        
        print(f"\n  --- METRICS ---")
        if best_eval:
            print(f"  Best Val mAP:   {best_eval['val_mAP']:.4f}")
            print(f"  Best Rank-1:    {best_eval['R1']:.4f}")
            print(f"  Best Rank-5:    {best_eval['R5']:.4f}")
            print(f"  Best Rank-10:   {best_eval['R10']:.4f}")
            print(f"  Best Epoch:     {best_eval.get('epoch', log_data.get('best_epoch', '?'))}")
            if best_eval.get('train_mAP') is not None:
                print(f"  Train mAP @ best: {best_eval['train_mAP']:.4f}")
                print(f"  Overfit gap:    {best_eval['train_mAP'] - best_eval['val_mAP']:.4f}")
        
        if log_data.get('best_mAP') is not None:
            print(f"  Log's Best mAP: {log_data['best_mAP']:.4f} (epoch {log_data.get('best_epoch', '?')})")
        
        max_ep = log_data.get('max_epochs_config', config_data.get('max_epochs', '?') if config_data else '?')
        print(f"  Last Epoch:     {log_data['last_epoch']}/{max_ep}")
        status = "COMPLETED" if log_data['training_complete'] else "INCOMPLETE/RUNNING"
        print(f"  Status:         {status}")
        
        if log_data.get('per_class_mAP'):
            pc = log_data['per_class_mAP']
            print(f"  Per-class mAP:  Container={pc['Container']:.4f}  Crosswalk={pc['Crosswalk']:.4f}  RubbishBins={pc['RubbishBins']:.4f}  TrafficSign={pc['TrafficSign']:.4f}")
        
        # Print all eval points for progression tracking
        print(f"\n  --- EVAL PROGRESSION ---")
        for ev in log_data['evals']:
            train_str = f"  train_mAP={ev['train_mAP']:.4f}" if ev.get('train_mAP') else ""
            gap_str = f"  gap={ev['train_mAP'] - ev['val_mAP']:.4f}" if ev.get('train_mAP') else ""
            print(f"    Ep {ev.get('epoch', '?'):>3}: val_mAP={ev['val_mAP']:.4f}  R1={ev['R1']:.4f}  R5={ev['R5']:.4f}  R10={ev['R10']:.4f}{train_str}{gap_str}")
    else:
        print("  Log: NOT FOUND")
    
    print(f"\n  --- CHECKPOINTS ---")
    print(f"  Checkpoint epochs: {ckpt_data['checkpoint_epochs']}")
    print(f"  Has best_model.pth: {ckpt_data['has_best_model']}")
    print(f"  Has final_model.pth: {ckpt_data['has_final_model']}")
    
    if extra_files:
        print(f"  Extra files: {extra_files}")
    
    # Store for summary
    all_results[exp_name] = {
        'log': log_data,
        'config': config_data,
        'checkpoints': ckpt_data,
    }


# ============================================================
# SUMMARY TABLE
# ============================================================
print("\n\n")
print("=" * 160)
print("SUMMARY TABLE - ALL EXPERIMENTS SORTED BY BEST VAL mAP")
print("=" * 160)
print(f"{'Experiment':<45} {'Best mAP':>10} {'R1':>8} {'R5':>8} {'R10':>8} {'Train mAP':>10} {'Gap':>8} {'BestEp':>7} {'Last/Max':>10} {'Status':>12} {'Backbone':<40}")
print("-" * 160)

ranked = []
for name, data in all_results.items():
    log = data['log']
    cfg = data['config']
    if log and log['evals']:
        best = get_best_eval(log['evals'])
        max_ep = log.get('max_epochs_config', cfg.get('max_epochs', '?') if cfg else '?')
        ranked.append((
            name,
            best['val_mAP'],
            best['R1'],
            best['R5'],
            best['R10'],
            best.get('train_mAP', 0),
            best.get('train_mAP', 0) - best['val_mAP'] if best.get('train_mAP') else 0,
            best.get('epoch', log.get('best_epoch', '?')),
            f"{log['last_epoch']}/{max_ep}",
            "DONE" if log['training_complete'] else "INCOMPLETE",
            cfg.get('backbone', '?') if cfg else '?',
        ))

ranked.sort(key=lambda x: -x[1])

for r in ranked:
    name, mAP, r1, r5, r10, tr_mAP, gap, best_ep, last_max, status, backbone = r
    tr_str = f"{tr_mAP:.4f}" if tr_mAP else "N/A"
    gap_str = f"{gap:.4f}" if gap else "N/A"
    print(f"{name:<45} {mAP:>10.4f} {r1:>8.4f} {r5:>8.4f} {r10:>8.4f} {tr_str:>10} {gap_str:>8} {str(best_ep):>7} {last_max:>10} {status:>12} {backbone:<40}")


# ============================================================
# ENSEMBLE RESULTS
# ============================================================
print("\n\n")
print("=" * 120)
print("ENSEMBLE RESULTS")
print("=" * 120)

# Check ensemble_features
ens_dir = os.path.join(BASE, "ensemble_features")
if os.path.isdir(ens_dir):
    print("\n--- ensemble_features/ ---")
    # Check ensemble_results.json
    ens_results = os.path.join(ens_dir, "ensemble_results.json")
    if os.path.exists(ens_results):
        with open(ens_results, 'r') as f:
            ens_data = json.load(f)
        print(json.dumps(ens_data, indent=2))
    
    # Check manifest
    manifest = os.path.join(ens_dir, "manifest.json")
    if os.path.exists(manifest):
        with open(manifest, 'r') as f:
            man_data = json.load(f)
        print("\nManifest:")
        print(json.dumps(man_data, indent=2))
    
    # Check log
    log_file = os.path.join(ens_dir, "log_rank0.txt")
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            log_content = f.read()
        # Print relevant lines
        for line in log_content.split('\n'):
            if any(k in line.lower() for k in ['map', 'rank', 'ensemble', 'model', 'result', 'best', 'weight']):
                print(f"  {line.strip()}")

# Check ensemble_submissions
for sub_dir_name in ["ensemble_submissions", "ensemble_submissions2"]:
    sub_dir = os.path.join(BASE, sub_dir_name)
    if os.path.isdir(sub_dir):
        print(f"\n--- {sub_dir_name}/ ---")
        summary_file = os.path.join(sub_dir, "submissions_summary.json")
        if os.path.exists(summary_file):
            with open(summary_file, 'r') as f:
                summary_data = json.load(f)
            print(json.dumps(summary_data, indent=2))
        
        # List CSV files
        csvs = sorted(glob.glob(os.path.join(sub_dir, "*.csv")))
        print(f"  Submission files: {[os.path.basename(c) for c in csvs]}")


# ============================================================
# BACKFILL LOGS
# ============================================================
print("\n\n")
print("=" * 120)
print("BACKFILL LOGS")
print("=" * 120)
bf_dir = os.path.join(BASE, "backfill_logs")
if os.path.isdir(bf_dir):
    for f in sorted(os.listdir(bf_dir)):
        fp = os.path.join(bf_dir, f)
        print(f"\n--- {f} ---")
        with open(fp, 'r') as fh:
            content = fh.read()
        # Print lines with metrics
        for line in content.split('\n'):
            if any(k in line.lower() for k in ['map', 'rank', 'error', 'result', 'best', 'complete', 'backfill']):
                print(f"  {line.strip()}")
        if not any(k in content.lower() for k in ['map', 'rank']):
            # Print last 20 lines
            lines = content.strip().split('\n')
            for line in lines[-20:]:
                print(f"  {line.strip()}")
