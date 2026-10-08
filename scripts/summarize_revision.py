#!/usr/bin/env python3
"""Aggregate results/revision/<suite>/<variant>/seed*/<Env>/<q>/*.json into tables.

Writes results/revision/summary_<suite>.{json,md}. Final reward = mean of the last
100 episodes (last 10% for the budget suite, whose runs have different lengths).
"""

import argparse
import glob
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load(suite_dir: Path):
    """Yield (variant, env, q, method_name, seed, record)."""
    for path in glob.glob(str(suite_dir / '*' / 'seed*' / '*' / 'q*' / '*.json')):
        p = Path(path)
        if p.name in ('summary.json',) or p.name.startswith('params_'):
            continue
        q, env, seed, variant = p.parent.name, p.parent.parent.name, p.parents[2].name, p.parents[3].name
        if '.' in env:  # set-aside results
            continue
        with open(p) as f:
            rec = json.load(f)
        yield variant, env, q, rec['method_name'], int(seed[4:]), rec


# Standard Gymnasium "solved" thresholds (100-episode moving average of the return)
THRESHOLD = {'CartPole': 475.0, 'Acrobot': -100.0, 'LunarLander': 200.0, 'MountainCar': -110.0}


def auc(rec):
    """Normalized area under the learning curve: mean training return over all episodes."""
    return float(np.mean(rec['reward_history']))


def episodes_to_threshold(rec, env, window=100):
    """First episode E with mean(r[E-window:E]) >= threshold, or None if never reached."""
    r = np.asarray(rec['reward_history'], dtype=float)
    if len(r) < window or env not in THRESHOLD:
        return None
    ma = np.convolve(r, np.ones(window) / window, 'valid')   # ma[k] = mean r[k:k+window]
    hit = np.nonzero(ma >= THRESHOLD[env])[0]
    return int(hit[0] + window) if hit.size else None


def final_reward(rec, frac=None):
    r = np.asarray(rec['reward_history'], dtype=float)
    k = max(1, int(len(r) * frac)) if frac else 100
    return float(r[-k:].mean())


def ms(x):
    x = np.asarray(x, dtype=float)
    return float(x.mean()), float(x.std(ddof=1)) if len(x) > 1 else 0.0, len(x)


def summarize_main(rows):
    table = defaultdict(lambda: defaultdict(list))
    times = defaultdict(lambda: defaultdict(list))
    evals = defaultdict(lambda: defaultdict(list))
    aucs = defaultdict(lambda: defaultdict(list))
    hits = defaultdict(lambda: defaultdict(list))
    for variant, env, q, name, seed, rec in rows:
        key = name + (' [matched MSE]' if variant == 'distill_matched' else
                      '' if variant == 'default' else f' [{variant}]')
        table[(env, q)][key].append(final_reward(rec))
        times[(env, q)][key].append(rec['training_time'] / 60)
        if rec.get('evaluation'):
            evals[(env, q)][key].append(rec['evaluation']['eval_return'])
        aucs[(env, q)][key].append(auc(rec))
        hits[(env, q)][key].append(episodes_to_threshold(rec, env))
    out = {}
    for (env, q), d in sorted(table.items()):
        out[f'{env}/{q}'] = {}
        for k, v in sorted(d.items()):
            h = hits[(env, q)][k]
            reached = [x for x in h if x is not None]
            out[f'{env}/{q}'][k] = {
                'reward': ms(v), 'minutes': ms(times[(env, q)][k]),
                'eval': ms(evals[(env, q)][k]) if evals[(env, q)][k] else None,
                'auc': ms(aucs[(env, q)][k]),
                # episodes to threshold: median over the seeds that reached it, and how many did
                'ep_to_thr': {'median': float(np.median(reached)) if reached else None,
                              'reached': len(reached), 'n': len(h),
                              'threshold': THRESHOLD.get(env)},
            }
    return out


def summarize_anchors(rows):
    acc = defaultdict(lambda: defaultdict(list))
    for variant, env, q, name, seed, rec in rows:
        m = int(variant[1:])
        dg = rec.get('diagnostics', {})
        a = acc[(env, m)]
        a['reward'].append(final_reward(rec))
        a['minutes'].append(rec['training_time'] / 60)
        cl = dg.get('clients', [])
        if cl:
            a['eff_rank_frac'].append(np.mean([c['effective_rank_lambda'] / c['D'] for c in cl]))
            a['log10_gamma'].append(np.mean([np.log10(max(c['gamma_min'], 1e-300)) for c in cl]))
        rounds = dg.get('rounds', [])
        if rounds:
            last = rounds[-len(rounds) // 4 or -1:]  # last quarter of rounds
            a['rho_mean'].append(np.mean([np.mean(r['rho_mean']) for r in last if r['rho_mean']]))
            a['teacher_dis'].append(np.mean([np.mean(r['teacher_disagreement']) for r in last]))
    out = {}
    for (env, m), a in sorted(acc.items()):
        out[f'{env}/m{m}'] = {k: ms(v) for k, v in a.items()}
    return out


def summarize_budget(rows):
    acc = defaultdict(list)
    for variant, env, q, name, seed, rec in rows:
        acc[(env, name, int(variant[1:]))].append(final_reward(rec, frac=0.1))
    return {f'{env}/{name}/N{n}': ms(v) for (env, name, n), v in sorted(acc.items())}


def to_md(suite, res):
    lines = [f'# Revision suite: {suite}', '']
    if suite == 'main':
        for cell, d in res.items():
            thr = next(iter(d.values()))['ep_to_thr']['threshold']
            lines += [f'## {cell}', '',
                      f'| Method | Train return, last 100 (mean ± std, n) | Greedy eval | AUC (mean return) '
                      f'| Episodes to {thr:g} (median; seeds reached) | Minutes |',
                      '|---|---|---|---|---|---|']
            for k, v in sorted(d.items(), key=lambda kv: -kv[1]['reward'][0]):
                r, t, e, a, h = v['reward'], v['minutes'], v.get('eval'), v['auc'], v['ep_to_thr']
                es = f'{e[0]:.1f} ± {e[1]:.1f}' if e else '–'
                hs = (f"{h['median']:.0f} ({h['reached']}/{h['n']})" if h['reached']
                      else f"not reached (0/{h['n']})")
                lines.append(f'| {k} | {r[0]:.1f} ± {r[1]:.1f} (n={r[2]}) | {es} | '
                             f'{a[0]:.1f} ± {a[1]:.1f} | {hs} | {t[0]:.1f} |')
            lines.append('')
    elif suite == 'anchors':
        lines += ['| Env / m | Reward | eff. rank / D | log10 γ_min | ρ mean | teacher disagreement | Minutes |',
                  '|---|---|---|---|---|---|---|']
        for k, v in res.items():
            f = lambda n: f'{v[n][0]:.3g} ± {v[n][1]:.2g}' if n in v else '–'
            lines.append(f'| {k} | {f("reward")} | {f("eff_rank_frac")} | {f("log10_gamma")} | '
                         f'{f("rho_mean")} | {f("teacher_dis")} | {f("minutes")} |')
    else:
        lines += ['| Env / method / N | Reward last 10% (mean ± std, n) |', '|---|---|']
        for k, v in res.items():
            lines.append(f'| {k} | {v[0]:.1f} ± {v[1]:.1f} (n={v[2]}) |')
    return '\n'.join(lines) + '\n'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=ROOT / 'results' / 'revision')
    ap.add_argument('--suites', nargs='+', default=['main', 'anchors', 'budget'])
    ap.add_argument('--main_dirs', nargs='+', default=['main', 'main_v2'],
                    help='directories summarized with the main-table summarizer')
    args = ap.parse_args()
    for suite in args.suites:
        if not (args.root / suite).exists():
            continue
        rows = list(load(args.root / suite))
        kind = ('main' if suite in args.main_dirs else 'budget' if suite.startswith('budget')
                else 'anchors' if suite.startswith('anchors') else suite)
        res = {'main': summarize_main, 'anchors': summarize_anchors,
               'budget': summarize_budget}[kind](rows)
        with open(args.root / f'summary_{suite}.json', 'w') as f:
            json.dump(res, f, indent=1)
        with open(args.root / f'summary_{suite}.md', 'w') as f:
            f.write(to_md(kind, res))
        print(f'{suite}: {len(rows)} runs -> {args.root}/summary_{suite}.md')


if __name__ == '__main__':
    main()
