#!/usr/bin/env python3
"""v3 selections (all on tuning seeds 0-2; score = mean last-100 training return).

  --step tune   pick FedQHD-hetero compile knobs (results/revision/tune_compile) and the
                pooled-oracle configs (results/revision/tune_oracle); write
                results/revision/tuned_params_v3 and the probe spec (best oracle per cell).
  --step budget compute the per-environment episode budget E* from the probe runs with the
                pre-registered rule and write results/revision/episodes_map.json.

Budget rule (fixed before looking at any FedQHD result): for the selected oracle of each
(env, setting), with r = mean over tuning seeds of its smoothed learning curve,
    base = mean(r[0:100]),  best = max over episodes E of mean(r[E-100:E]),
    E*   = min{E in GRID : mean(r[E-100:E]) - base >= 0.95 * (best - base)}.
The environment budget is the max of E* over its two settings.
"""

import argparse
import glob
import json
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / 'results' / 'revision'
QDIR = {'q1': 'q1_homogeneous', 'q2': 'q2_heterogeneous'}
GRID = [300, 600, 1000, 1500]
ORACLES = {'q1': ['pooled_qhd', 'oracle_qhd', 'oracle_dqn'],
           'q2': ['oracle_qhd_hetero', 'oracle_dqn_hetero']}
RESULT = {'fedqhd_hetero': 'FedQHD_(Heterogeneous).json', 'pooled_qhd': 'Pooled_QHD.json',
          'oracle_qhd': 'Oracle_QHD_(Homogeneous).json', 'oracle_dqn': 'Oracle_DQN_(Homogeneous).json',
          'oracle_qhd_hetero': 'Oracle_QHD_(Heterogeneous).json',
          'oracle_dqn_hetero': 'Oracle_DQN_(Heterogeneous).json'}


def runs(suite: str, variant: str, env: str, q: str, method: str):
    return glob.glob(str(R / suite / variant / 'seed*' / env / QDIR[q] / RESULT[method]))


def score(files, min_seeds=3):
    if len(files) < min_seeds:
        return None
    return float(np.mean([np.mean(json.load(open(f))['reward_history'][-100:]) for f in files]))


def method_of(variant: str):
    """'oracle_qhd_hetero_lr0.1_dec0.99' -> 'oracle_qhd_hetero'; 'oracle_dqn_lrs1_eps5' -> 'oracle_dqn'."""
    m = re.match(r'(.+?)_(lrs|lr)\d', variant)
    return m.group(1) if m else None


def parse_overrides(variant: str) -> dict:
    out = {}
    for k, v in re.findall(r'(lrs|lr|dec|eps|lam)([\d.e+-]+)', variant):
        key = {'lr': 'qhd_lr', 'dec': 'qhd_exploration_decay', 'lrs': 'dqn_lr_scale',
               'eps': 'dqn_eps_steps_scale', 'lam': 'ridge_lambda'}[k]
        out[key] = float(v)
    if variant.endswith('_warmstart'):
        out['compile_mode'] = 'warmstart'
    elif variant.endswith('_overwrite'):
        out['compile_mode'] = 'overwrite'
    return out


def step_tune(envs):
    dst = R / 'tuned_params_v3'
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(R / 'tuned_params', dst)
    report, probe = [], {}
    for env in envs:
        # FedQHD-hetero compile knobs
        cand = {v.name: score(runs('tune_compile', v.name, env, 'q2', 'fedqhd_hetero'))
                for v in (R / 'tune_compile').iterdir()}
        cand = {k: s for k, s in cand.items() if s is not None}
        best = max(cand, key=cand.get)
        path = dst / env / QDIR['q2'] / 'params_fedqhd_hetero.json'
        p = json.load(open(path)); p.update(parse_overrides(best)); json.dump(p, open(path, 'w'), indent=2)
        report.append(f'{env}/q2 FedQHD compile: {best} = {cand[best]:.1f}  (all: '
                      + ', '.join(f'{k}={s:.0f}' for k, s in sorted(cand.items())) + ')')
        # oracles
        for q, methods in ORACLES.items():
            best_cell = None
            for method in methods:
                if 'dqn' in method:
                    # DQN oracles are not tuned (zoo parameters as-is). They still compete for
                    # the probe if their zoo-config tuning-seed runs are complete.
                    sc = score(runs('tune_oracle', f'{method}_lrs1_eps1', env, q, method))
                    report.append(f'{env}/{q} {method}: zoo params (not tuned), score '
                                  + (f'{sc:.1f}' if sc is not None else 'n/a'))
                    if sc is not None and (best_cell is None or sc > best_cell[1]):
                        best_cell = (method, sc, {})
                    continue
                vs = [v.name for v in (R / 'tune_oracle').iterdir() if method_of(v.name) == method]
                sc = {v: score(runs('tune_oracle', v, env, q, method)) for v in vs}
                sc = {k: s for k, s in sc.items() if s is not None}
                if not sc:
                    report.append(f'{env}/{q} {method}: NO complete tuning runs'); continue
                v = max(sc, key=sc.get)
                src = dst / env / QDIR[q] / f"params_{'fedqhd_homo' if q == 'q1' and 'qhd' in method else 'fedqhd_hetero' if 'qhd' in method else method}.json"
                if not src.exists():  # DQN oracles: start from the original params file
                    src = dst / env / QDIR[q] / f'params_{method}.json'
                p = json.load(open(src)); p.update(parse_overrides(v)); p['method_key'] = method
                p['tuned'] = {'protocol': 'tune_oracle, seeds 0-2', 'variant': v, 'score': sc[v]}
                json.dump(p, open(dst / env / QDIR[q] / f'params_{method}.json', 'w'), indent=2)
                report.append(f'{env}/{q} {method}: {v} = {sc[v]:.1f}')
                if best_cell is None or sc[v] > best_cell[1]:
                    best_cell = (method, sc[v], parse_overrides(v))
            if best_cell:
                probe[f'{env}/{q}'] = {'method': best_cell[0], 'score': best_cell[1],
                                       'overrides': {}}
    (R / 'probe_spec.json').write_text(json.dumps(probe, indent=1))
    print('\n'.join(report))
    print(f'probe spec: {probe}')


def step_budget(envs):
    spec = json.load(open(R / 'probe_spec.json'))
    budget, report = {}, []
    for env in envs:
        e_star = []
        for q in ['q1', 'q2']:
            item = spec.get(f'{env}/{q}')
            if not item:
                continue
            fs = runs('probe', f'{q}_{item["method"]}', env, q, item['method'])
            curves = [np.array(json.load(open(f))['reward_history'], dtype=float) for f in fs]
            n = min(len(c) for c in curves)
            r = np.mean([c[:n] for c in curves], axis=0)
            win = np.convolve(r, np.ones(100) / 100, 'valid')  # win[k] = mean r[k:k+100]
            base, best = win[0], win.max()
            ok = [E for E in GRID if E <= n and win[E - 100] - base >= 0.95 * (best - base)]
            E = ok[0] if ok else GRID[-1]
            e_star.append(E)
            report.append(f'{env}/{q} ({item["method"]}, {len(fs)} seeds): base {base:.1f}, best '
                          f'{best:.1f}, window at ' + ', '.join(f'{g}:{win[g-100]:.1f}' for g in GRID if g <= n)
                          + f' -> E* = {E}')
        budget[env] = max(e_star) if e_star else 600
    (R / 'episodes_map.json').write_text(json.dumps(budget, indent=1))
    print('\n'.join(report))
    print('episodes per environment:', budget)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--step', required=True, choices=['tune', 'budget'])
    ap.add_argument('--envs', nargs='+', default=['CartPole', 'Acrobot', 'LunarLander', 'MountainCar'])
    a = ap.parse_args()
    step_tune(a.envs) if a.step == 'tune' else step_budget(a.envs)
