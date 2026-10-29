"""Train temporal contact models using training people; select on validation people."""
import argparse
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from beta_translator.contact_features import FEATURE_VERSION
from beta_translator.contact_network import ContactTCN, predict_logits
from beta_translator.wholebody_pose import sha256


def load_clips(root, split):
    manifest = json.loads((root/'manifest.json').read_text())
    clips = []
    for entry in manifest['clips']:
        if entry['split'] != split: continue
        with np.load(root/(entry['id']+'.npz'), allow_pickle=False) as data:
            clip = {k: data[k].copy() for k in ('features', 'labels', 'occluded')}
        clip['id'] = entry['id']; clip['meta'] = entry
        clip['transitions'] = [np.flatnonzero((clip['labels'][1:, l] != clip['labels'][:-1, l]) &
                                 (clip['labels'][1:, l] >= 0) & (clip['labels'][:-1, l] >= 0))+1 for l in range(4)]
        clips.append(clip)
    if not clips: raise ValueError(f'No {split} clips')
    return clips


def sample_batch(clips, rng, batch_size, length, device):
    selected = []
    for _ in range(batch_size):
        clip = clips[rng.integers(len(clips))]
        limb = int(rng.integers(4)); count = len(clip['labels'])
        events = clip['transitions'][limb]
        center = int(rng.choice(events)) if len(events) and rng.random() < .5 else int(rng.integers(count))
        start = min(max(0, center-length//2), max(0, count-length))
        stop = min(count, start+length)
        selected.append((clip['features'][start:stop, limb], clip['labels'][start:stop, limb]))
    holds = max(x.shape[1] for x, _ in selected); features = selected[0][0].shape[-1]
    x = np.zeros((batch_size, length, holds, features), np.float32)
    y = np.full((batch_size, length), -100, np.int64)
    mask = np.zeros((batch_size, holds), bool)
    for i, (f, labels) in enumerate(selected):
        x[i, :len(f), :f.shape[1]] = f
        y[i, :len(f)] = labels; mask[i, :f.shape[1]] = True
    # Mild observation noise; semantic feature flags and candidate identities are preserved.
    x[..., :12] += rng.normal(0, .012, x[..., :12].shape).astype(np.float32)
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device), torch.from_numpy(mask).to(device)


def validate(model, clips, device):
    total = count = correct = 0
    for clip in clips:
        logits = predict_logits(model, clip['features'], device)
        labels = clip['labels']; valid = labels >= 0
        tensor = torch.from_numpy(logits[valid])
        target = torch.from_numpy(labels[valid])
        total += F.cross_entropy(tensor, target, reduction='sum').item()
        count += int(valid.sum()); correct += int((logits.argmax(-1)[valid] == labels[valid]).sum())
    return total/count, correct/count


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=17); p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--steps', type=int, default=80); p.add_argument('--width', type=int, default=96)
    p.add_argument('--layers', type=int, default=6); p.add_argument('--batch', type=int, default=6)
    p.add_argument('--length', type=int, default=256); p.add_argument('--device', default='cpu')
    p.add_argument('--resume', action='store_true')
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    if args.device == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable. Run with GPU access or choose cpu.')
    rng = np.random.default_rng(args.seed)
    train = load_clips(args.data, 'train'); validation = load_clips(args.data, 'validation')
    model = ContactTCN(train[0]['features'].shape[-1], args.width, args.layers).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=5e-5)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(model=model.config, feature_version=FEATURE_VERSION,
                  loss='Cross entropy; 1.5x free-state weight plus 1x within two frames of a transition',
                  parameters=sum(p.numel() for p in model.parameters()),
                  manifest_sha256=sha256(args.data/'manifest.json'),
                  trainer_sha256=sha256(__file__),
                  train_clips=[c['id'] for c in train], validation_clips=[c['id'] for c in validation])
    if (args.output/'config.json').exists() and not args.resume:
        raise FileExistsError('This run already exists. Use --resume or a new output directory.')
    if args.resume:
        old = json.loads((args.output/'config.json').read_text())
        for key in ('seed', 'epochs', 'steps', 'width', 'layers', 'batch', 'length', 'manifest_sha256', 'loss', 'trainer_sha256'):
            if config[key] != old.get(key): raise ValueError(f'Resume configuration changed: {key}')
    (args.output/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    print(json.dumps(config), flush=True)
    best = float('inf'); history = []; start = time.monotonic(); initial_epoch = 0
    if args.resume:
        last = torch.load(args.output/'last.pt', map_location='cpu', weights_only=True)
        model.load_state_dict(last['model']); optimizer.load_state_dict(last['optimizer'])
        scheduler.load_state_dict(last['scheduler']); rng.bit_generator.state = last['rng']
        torch.set_rng_state(last['torch_rng'])
        if args.device == 'mps' and 'mps_rng' in last: torch.mps.set_rng_state(last['mps_rng'])
        best = last['best']; initial_epoch = last['epoch']; history = last['history']
        start -= history[-1]['elapsed_sec'] if history else 0
    for epoch in range(initial_epoch, args.epochs):
        model.train(); losses = []
        for step in range(args.steps):
            x, y, mask = sample_batch(train, rng, args.batch, args.length, args.device)
            logits = model(x, mask)
            margin = min(model.radius, args.length//4)
            if margin:
                logits, target = logits[:, margin:-margin], y[:, margin:-margin]
            else: target = y
            per_frame = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1),
                                        ignore_index=-100, reduction='none').reshape_as(target)
            transitions = torch.zeros_like(target, dtype=torch.float32)
            transitions[:, 1:] = ((target[:, 1:] != target[:, :-1]) &
                                   (target[:, 1:] >= 0) & (target[:, :-1] >= 0)).float()
            nearby = F.max_pool1d(transitions[:, None], 5, stride=1, padding=2)[:, 0]
            free_state = mask.sum(dim=-1)[:, None]-2
            weights = (1. + .5*(target == free_state).float() + nearby) * (target >= 0).float()
            loss = (per_frame*weights).sum() / weights.sum().clamp_min(1)
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite training loss')
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step(); losses.append(loss.detach().item())
        scheduler.step()
        record = {'epoch': epoch+1, 'train_loss': float(np.mean(losses)), 'elapsed_sec': time.monotonic()-start}
        if epoch == 0 or (epoch+1) % 3 == 0 or epoch+1 == args.epochs:
            val_loss, val_acc = validate(model, validation, args.device)
            record.update(validation_nll=val_loss, validation_accuracy=val_acc)
            if val_loss < best:
                best = val_loss
                weights = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                torch.save({'state_dict': weights, 'model': model.config, 'feature_version': FEATURE_VERSION,
                            'epoch': epoch+1, 'validation_nll': val_loss, 'training': config}, args.output/'best.pt')
        history.append(record)
        (args.output/'history.json').write_text(json.dumps(history, indent=2)+'\n')
        state = dict(model={k: v.detach().cpu() for k, v in model.state_dict().items()},
            optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), rng=rng.bit_generator.state,
            torch_rng=torch.get_rng_state(), best=best, epoch=epoch+1, history=history)
        if args.device == 'mps': state['mps_rng'] = torch.mps.get_rng_state()
        torch.save(state, args.output/'last.part'); (args.output/'last.part').replace(args.output/'last.pt')
        print(json.dumps(record), flush=True)
    print(f'Completed training; selected validation NLL {best:.5f}', flush=True)


if __name__ == '__main__': main()
