#!/usr/bin/env python3
"""Check that every run of a results suite follows one protocol.

For each <suite>/<variant>/seed<k>/<Env>/<q>/params_<method>.json it checks
  * seeds = the expected evaluation seeds, one run per (method, env, seed)
  * episodes, agent_num, anchor m (anchor-based methods), dqn preset, truncation handling,
    independent-QHD reset flag, greedy evaluation present
  * QHD-family methods use exactly the tuned FedQHD agent parameters of their setting
and prints every deviation plus a coverage matrix (number of seeds per method and cell).
"""

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QHD_KEYS = ['qhd_lr', 'qhd_discount', 'qhd_exploration_rate', 'qhd_exploration_decay',
            'qhd_exploration_min', 'rff_gamma', 'hyperdimension']
QHD_FAMILY = {
    'q1': ['independent_qhd', 'fedqhd_homo', 'oracle_qhd', 'pooled_qhd'],
    'q2': ['independent_qhd_hetero', 'fedqhd_hetero', 'oracle_qhd_hetero',
           'truncate_fedavg_qhd_hetero', 'fedqhd_gd_hetero', 'fedhql_qhd_hetero'],
}
# references tuned separately for pooled data (checked against their own tuned file)
OWN_TUNED = {'pooled_qhd', 'oracle_qhd', 'oracle_qhd_hetero'}
FEDERATED_QHD = {'fedqhd_homo', 'fedqhd_hetero', 'oracle_qhd_hetero', 'truncate_fedavg_qhd_hetero',
                 'fedqhd_gd_hetero'}
ANCHOR = {'fedqhd_hetero', 'oracle_qhd_hetero', 'distillation_dqn_hetero', 'fedqhd_gd_hetero',
          'fedhpd_hetero'}
DQN = {'oracle_dqn', 'fedavg_dqn', 'fedprox_dqn', 'distillation_dqn', 'oracle_dqn_hetero',
       'distillation_dqn_hetero', 'fedhql_dqn_hetero'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--suite_dir', type=Path, default=ROOT / 'results/revision/main_v2')
    ap.add_argument('--tuned', type=Path, default=ROOT / 'results/revision/tuned_params')
    ap.add_argument('--seeds', default='42-51')
    ap.add_argument('--episodes', type=int, default=600)
    ap.add_argument('--episodes_map', type=Path, default=None, help='json {env: episodes}')
    ap.add_argument('--agents', type=int, default=5)
    ap.add_argument('--m', type=int, default=10000)
    args = ap.parse_args()
    a, b = map(int, args.seeds.split('-'))
    seeds = set(range(a, b + 1))
    ep_map = json.load(open(args.episodes_map)) if args.episodes_map else None

    issues, cover = [], defaultdict(set)
    for f in sorted(glob.glob(str(args.suite_dir / '*' / 'seed*' / '*' / 'q*' / 'params_*.json'))):
        p = json.load(open(f))
        path = Path(f)
        variant, env, q = path.parents[3].name, path.parent.parent.name, p['question']
        key, seed, opt = p['method_key'], p['random_seed'], p.get('revision_options', {})
        # only count runs whose result file exists (params are written after the result)
        tag = f'{variant}/{env}/{q}/{key}/seed{seed}'
        cover[(variant, key, env, q)].add(seed)
        def bad(msg):
            issues.append(f'{tag}: {msg}')
        if seed not in seeds:
            bad(f'seed {seed} not in evaluation seeds')
        want = ep_map[env] if ep_map else args.episodes
        if p['episodes'] != want:
            bad(f"episodes={p['episodes']} (protocol: {want})")
        if p['agent_num'] != args.agents:
            bad(f"agent_num={p['agent_num']}")
        if not opt.get('bootstrap_on_truncation'):
            bad('truncation treated as terminal')
        if key in ANCHOR and p['anchor_set_size'] != args.m:
            bad(f"anchor m={p['anchor_set_size']}")
        if key in DQN and opt.get('dqn_preset') != 'zoo':
            bad(f"dqn_preset={opt.get('dqn_preset')}")
        if key in DQN and (float(opt.get('dqn_lr_scale') or 1) != 1 or
                           float(opt.get('dqn_eps_steps_scale') or 1) != 1):
            bad('DQN hyperparameters deviate from the zoo preset')
        if key in DQN and opt.get('dqn_device') != 'cpu':
            bad(f"dqn_device={opt.get('dqn_device')} (protocol: cpu)")
        if (key in {'fedavg_dqn', 'fedprox_dqn', 'distillation_dqn', 'fedhql_dqn_hetero'} or
                (key == 'distillation_dqn_hetero' and variant == 'default')) \
                and int(p.get('dqn_agg_interval') or 0) != 25:
            bad(f"DQN federation interval K={p.get('dqn_agg_interval')} (protocol: 25)")
        if key == 'fedprox_dqn' and float(opt.get('fedprox_mu') or 0) != 0.01:
            bad(f"FedProx mu={opt.get('fedprox_mu')} (protocol: 0.01, not tuned)")
        if key.startswith('independent_qhd') and not opt.get('indep_no_reset'):
            bad('independent QHD resets every K')
        if not opt.get('eval_episodes'):
            bad('no greedy evaluation')
        qdir = 'q1_homogeneous' if q == 'q1' else 'q2_heterogeneous'
        if key in OWN_TUNED:
            ref = json.load(open(args.tuned / env / qdir / f'params_{key}.json'))
            if 'tuned' not in ref:
                bad('oracle params are NOT tuned')
            ro = p.get('revision_options', {})
            got = {**p, **ro}
            keys = (QHD_KEYS if 'qhd' in key else []) + ['dqn_lr_scale', 'dqn_eps_steps_scale']
            diff = {k: (got.get(k), ref.get(k)) for k in keys if k in ref and got.get(k) != ref.get(k)}
            if diff:
                bad(f'oracle params differ from its tuned file (run, tuned): {diff}')
        elif key in QHD_FAMILY[q]:
            ref_file = args.tuned / env / qdir / f"params_fedqhd_{'homo' if q == 'q1' else 'hetero'}.json"
            ref = json.load(open(ref_file))
            diff = {k: (p.get(k), ref.get(k)) for k in QHD_KEYS if p.get(k) != ref.get(k)}
            ro = p.get('revision_options', {})
            for k in ['ridge_lambda', 'compile_mode']:
                if q == 'q2' and k in ref and key in FEDERATED_QHD and ro.get(k) != ref[k]:
                    diff[k] = (ro.get(k), ref[k])
            if key in FEDERATED_QHD and p.get('qhd_agg_interval') != ref.get('qhd_agg_interval'):
                diff['qhd_agg_interval'] = (p.get('qhd_agg_interval'), ref.get('qhd_agg_interval'))
            if 'tuned' not in ref:
                bad('reference FedQHD params are NOT tuned')
            if diff:
                bad(f'QHD params differ from tuned FedQHD (run, tuned): {diff}')

    print(f'{len(issues)} deviations')
    # documented exceptions: cells excluded from the suite and reported from a legacy run
    for name in ('exclusions.json', 'legacy_overrides.json'):
        f = args.suite_dir.parent / name
        if f.exists():
            for e in json.load(open(f)).get(args.suite_dir.name, []):
                print(f'  DOCUMENTED EXCEPTION ({name}): '
                      + ', '.join(f'{k}={v}' for k, v in e.items()
                                  if k in ('env', 'q', 'method', 'method_name', 'reason', 'protocol')))
    for line in issues[:200]:
        print('  ' + line)
    print('\ncoverage (number of evaluation seeds per method and cell):')
    cells = sorted({(env, q) for (_, _, env, q) in cover})
    for (variant, key) in sorted({(v, k) for (v, k, _, _) in cover}):
        row = [len(cover.get((variant, key, env, q), ())) for env, q in cells]
        print(f'  {variant[:16]:16s} {key:28s} ' + ' '.join(f'{n:2d}' for n in row))
    print('  cells: ' + ', '.join(f'{e[:4]}/{q}' for e, q in cells))


if __name__ == '__main__':
    main()
