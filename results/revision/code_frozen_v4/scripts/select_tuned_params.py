#!/usr/bin/env python3
"""Pick the FedQHD hyperparameters from the tuning suite and write a tuned params root.

Stage 1 (agent parameters):  results/revision/tune/<q>_lr.._dec.._g.._rg../seed*/<Env>/<q dir>/
Stage 2 (interval K):        results/revision/tune_K/<q>_K<K>/seed*/<Env>/<q dir>/

Score = mean over tuning seeds of the mean training return of the last 100 episodes.
Outputs
  --best_out    json {"<Env>/<q>": {param: value}} (input of `--suite tune --tune_stage 2`)
  --params_out  copy of the original params root with params_fedqhd_{homo,hetero}.json replaced
"""

import argparse
import glob
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
QDIR = {'q1': 'q1_homogeneous', 'q2': 'q2_heterogeneous'}
FILE = {'q1': 'params_fedqhd_homo.json', 'q2': 'params_fedqhd_hetero.json'}
RES = {'q1': 'FedQHD_(Homogeneous).json', 'q2': 'FedQHD_(Heterogeneous).json'}


def parse_variant(v):
    m = re.match(r'(q[12])_lr([\d.]+)_dec([\d.]+)_g([\d.]+)_rg([\d.]+)$', v)
    if m:
        q, lr, dec, g, rg = m.groups()
        return q, {'qhd_lr': float(lr), 'qhd_exploration_decay': float(dec),
                   'qhd_discount': float(g), 'rff_gamma': float(rg),
                   'qhd_exploration_rate': 1.0, 'qhd_exploration_min': 0.001}
    m = re.match(r'(q[12])_K(\d+)$', v)
    if m:
        return m.group(1), {'qhd_agg_interval': int(m.group(2))}
    return None, None


def scores(suite_dir: Path):
    acc = defaultdict(list)
    cfgs = {}
    for vdir in sorted(suite_dir.iterdir()):
        q, cfg = parse_variant(vdir.name)
        if q is None:
            continue
        for f in glob.glob(str(vdir / 'seed*' / '*' / QDIR[q] / RES[q])):
            env = Path(f).parents[1].name
            if '.' in env:  # set-aside results, e.g. 'Acrobot.bugbounds'
                continue
            r = json.load(open(f))['reward_history']
            acc[(env, q, vdir.name)].append(float(np.mean(r[-100:])))
            cfgs[vdir.name] = cfg
    return acc, cfgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', type=int, default=1, choices=[1, 2])
    ap.add_argument('--tune_dir', type=Path, default=None)
    ap.add_argument('--stage1_best', type=Path, default=ROOT / 'results/revision/tune_best_stage1.json')
    ap.add_argument('--best_out', type=Path, default=None)
    ap.add_argument('--params_root', type=Path,
                    default=Path('/home/yuchen/projects/FedQHD/results/main_table_3seed'))
    ap.add_argument('--params_out', type=Path, default=ROOT / 'results/revision/tuned_params')
    ap.add_argument('--min_seeds', type=int, default=3)
    args = ap.parse_args()

    tune_dir = args.tune_dir or ROOT / 'results/revision' / ('tune' if args.stage == 1 else 'tune_K')
    best_out = args.best_out or ROOT / f'results/revision/tune_best_stage{args.stage}.json'
    acc, cfgs = scores(tune_dir)

    table = defaultdict(list)
    for (env, q, v), vals in acc.items():
        if len(vals) >= args.min_seeds:
            table[(env, q)].append((float(np.mean(vals)), float(np.std(vals)), len(vals), v))

    base = json.load(open(args.stage1_best)) if args.stage == 2 else {}
    best, report = {}, []
    for (env, q), rows in sorted(table.items()):
        rows.sort(reverse=True)
        top = rows[0]
        cfg = dict(base.get(f'{env}/{q}', {}), **cfgs[top[3]])
        best[f'{env}/{q}'] = cfg
        line = f'{env}/{q}: best {top[3]} = {top[0]:.1f} ± {top[1]:.1f} (n={top[2]})'
        if len(rows) > 1:
            line += f'; runner-up {rows[1][3]} = {rows[1][0]:.1f}'
        report.append(line)
    best_out.write_text(json.dumps(best, indent=1))
    print('\n'.join(report))
    print(f'-> {best_out}')

    if args.stage == 2:
        if args.params_out.exists():
            shutil.rmtree(args.params_out)
        shutil.copytree(args.params_root, args.params_out)
        for key, cfg in best.items():
            env, q = key.split('/')
            path = args.params_out / env / QDIR[q] / FILE[q]
            p = json.load(open(path))
            p.update(cfg)
            p['tuned'] = {'protocol': 'tuning seeds 0-2, stage1 agent grid + stage2 K sweep',
                          'source': str(tune_dir)}
            path.write_text(json.dumps(p, indent=2))
        print(f'tuned params root -> {args.params_out}')


if __name__ == '__main__':
    main()
