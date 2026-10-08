#!/usr/bin/env python3
"""
Experiment suites for the post-review revision of the FedQHD paper.

Each job is one (method, env, setting, variant, seed) run of scripts/run_single_method.py,
written to  <out>/<suite>/<variant>/seed<k>/<Env>/<q1_homogeneous|q2_heterogeneous>/.
Finished jobs are skipped, so the driver can be re-launched to resume.

Suites
  main     More seeds for the main table, plus the new heterogeneous baselines
           (FedHQL with QHD / DQN clients, FedQHD-GD, communication-matched MSE distillation).
  anchors  Heterogeneous FedQHD with m in {200, ..., 10000}: under- vs over-determined compile
           (m >= D_i), with conditioning / coverage diagnostics.
  envhet   Per-client dynamics heterogeneity (levels 0.1 / 0.3 / 0.5).
  budget   Scalability in N under a fixed total budget of client-episodes.

Hyperparameters come from the tuned params files of the submitted runs
(--params_root/<Env>/<q>/params_<method>.json); new methods inherit those of their closest
existing counterpart.

Usage
  PYTHONPATH=. python scripts/run_revision_experiments.py --suite main --seeds 42-51 --jobs 40
  PYTHONPATH=. python scripts/run_revision_experiments.py --suite anchors --envs CartPole --dry_run
"""

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, as_completed, wait
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENVS = ['CartPole', 'Acrobot', 'LunarLander', 'MountainCar']

Q1 = ['independent_qhd', 'fedqhd_homo', 'oracle_qhd', 'oracle_dqn', 'fedavg_dqn',
      'distillation_dqn', 'truncate_fedavg_qhd', 'pooled_qhd', 'fedprox_dqn']
# Q1 methods of the main table. Truncate FedAvg-QHD is omitted: with a shared encoder it is
# the same algorithm as FedQHD-homo.
MAIN_Q1 = ['independent_qhd', 'fedqhd_homo', 'oracle_qhd', 'pooled_qhd', 'oracle_dqn',
           'fedavg_dqn', 'fedprox_dqn', 'distillation_dqn']
Q2 = ['independent_qhd_hetero', 'fedqhd_hetero', 'oracle_qhd_hetero', 'oracle_dqn_hetero',
      'truncate_fedavg_qhd_hetero', 'distillation_dqn_hetero']
NEW_Q2 = ['fedhql_qhd_hetero', 'fedhql_dqn_hetero', 'fedqhd_gd_hetero', 'fedhpd_hetero']

# method -> method whose tuned hyperparameters it inherits. QHD-client baselines that were
# never tuned on their own use FedQHD's tuned QHD parameters for the same setting (same agents,
# only the federation step differs). Oracle QHD keeps its own tuned parameters. With
# --dqn_preset zoo all DQN methods use the rl-baselines3-zoo parameters instead.
INHERIT = {
    'fedhql_qhd_hetero': 'fedqhd_hetero',
    'fedqhd_gd_hetero': 'fedqhd_hetero',
    'fedhql_dqn_hetero': 'distillation_dqn_hetero',
    'fedhpd_hetero': 'distillation_dqn_hetero',
    'fedprox_dqn': 'fedavg_dqn',
    'pooled_qhd': 'fedqhd_homo',
}
INHERIT_TUNED_QHD = {
    'oracle_qhd': 'fedqhd_homo',
    'oracle_qhd_hetero': 'fedqhd_hetero',
    'independent_qhd': 'fedqhd_homo',
    'truncate_fedavg_qhd': 'fedqhd_homo',
    'independent_qhd_hetero': 'fedqhd_hetero',
    'truncate_fedavg_qhd_hetero': 'fedqhd_hetero',
}
# methods that use the shared anchor set S_ref (overridden by --anchor_m)
ANCHOR_METHODS = {'fedqhd_hetero', 'oracle_qhd_hetero', 'distillation_dqn_hetero',
                  'fedqhd_gd_hetero', 'fedhpd_hetero'}  # FedHPD: public state set S_p

PARAM_KEYS = [
    'episodes', 'agent_num', 'qhd_lr', 'qhd_discount', 'qhd_exploration_rate',
    'qhd_exploration_decay', 'qhd_exploration_min', 'qhd_agg_interval', 'dqn_lr',
    'dqn_discount', 'dqn_exploration_rate', 'dqn_exploration_decay', 'dqn_exploration_min',
    'dqn_agg_interval', 'hyperdimension', 'rff_gamma', 'anchor_set_size', 'hetero_dims',
    'fedprox_mu', 'ridge_lambda', 'compile_mode', 'dqn_lr_scale', 'dqn_eps_steps_scale',
]
# DQN-family methods whose dqn_agg_interval is fixed to --dqn_k (oracle DQNs and FedHPD do not
# use it, but get the same value so the records are uniform)
DQN_FIXED_K = {'fedavg_dqn', 'fedprox_dqn', 'distillation_dqn', 'distillation_dqn_hetero',
               'fedhql_dqn_hetero', 'oracle_dqn', 'oracle_dqn_hetero', 'fedhpd_hetero'}
ORACLES = {'q1': ['pooled_qhd', 'oracle_qhd', 'oracle_dqn'],
           'q2': ['oracle_qhd_hetero', 'oracle_dqn_hetero']}


def question_of(method):
    return 'q1' if method in Q1 else 'q2'


def load_params(params_root: Path, env: str, method: str, tuned_qhd: bool = False) -> dict:
    q = 'q1_homogeneous' if question_of(method) == 'q1' else 'q2_heterogeneous'
    # a method's own params file wins over INHERIT (e.g. a tuned params_fedprox_dqn.json);
    # with --tuned_qhd the QHD-family methods always use FedQHD's tuned file
    own = params_root / env / q / f'params_{method}.json'
    src = method if own.exists() else INHERIT.get(method, method)
    own_tuned = own.exists() and 'tuned' in json.load(open(own))
    if tuned_qhd and not own_tuned:
        src = INHERIT_TUNED_QHD.get(method, src)
    path = params_root / env / q / f'params_{src}.json'
    if not path.exists():
        print(f'  [warn] no params file {path}; using CLI defaults')
        return {}
    with open(path) as f:
        p = json.load(f)
    return {k: p[k] for k in PARAM_KEYS if k in p and p[k] is not None}


def make_jobs(args):
    """Yield (suite, variant, method, env, seed, overrides) tuples."""
    seeds = args.seeds
    envs = args.envs
    if args.suite == 'main':
        for env, seed in itertools.product(envs, seeds):
            for method in MAIN_Q1 + Q2 + NEW_Q2:
                yield ('default', method, env, seed, {})
            # communication-matched distillation: same anchors, interval and target as FedQHD
            yield ('distill_matched', 'distillation_dqn_hetero', env, seed,
                   {'distill_loss': 'mse', 'distill_interval': 'MATCH_QHD'})
    elif args.suite == 'anchors':
        for env, seed, m in itertools.product(envs, seeds, args.m_grid):
            yield (f'm{m}', 'fedqhd_hetero', env, seed, {'anchor_set_size': m})
    elif args.suite == 'envhet':
        methods = ['independent_qhd', 'fedqhd_homo', 'fedavg_dqn', 'fedqhd_hetero',
                   'distillation_dqn_hetero', 'fedhql_dqn_hetero']
        for env, seed, lvl in itertools.product(envs, seeds, args.levels):
            base = {'env_hetero': 'dynamics', 'env_hetero_level': lvl}
            for method in methods:
                yield (f'level{lvl}', method, env, seed, dict(base))
            yield (f'level{lvl}_clientanchors', 'fedqhd_hetero', env, seed,
                   dict(base, anchor_source='clients'))
    elif args.suite == 'tune_compile':
        # FedQHD-hetero compile knobs on top of the stage-2 tuned parameters
        for env, seed, lam, mode in itertools.product(envs, seeds, [1e-4, 1e-2, 1.0, 10.0],
                                                      ['overwrite', 'warmstart']):
            yield (f'lam{lam:g}_{mode}', 'fedqhd_hetero', env, seed,
                   {'ridge_lambda': lam, 'compile_mode': mode})
    elif args.suite == 'tune_oracle':
        # pooled-data references, re-tuned for their N× data stream
        for env, seed in itertools.product(envs, seeds):
            for method in ['pooled_qhd', 'oracle_qhd', 'oracle_qhd_hetero']:
                for lr, dec in itertools.product([0.05, 0.1, 0.2, 0.5], [0.95, 0.99, 0.995]):
                    yield (f'{method}_lr{lr}_dec{dec}', method, env, seed,
                           {'qhd_lr': lr, 'qhd_exploration_decay': dec})
            # DQN oracles are NOT tuned: like every DQN-family method they use the
            # rl-baselines3-zoo parameters unchanged (user decision, 2026-10-07)
    elif args.suite == 'probe':
        # long runs of the selected oracle per (env, setting) to fix the episode budget
        with open(args.probe_spec) as f:
            spec = json.load(f)
        for key, item in spec.items():
            env, q = key.split('/')
            if env in envs:
                for seed in seeds:
                    yield (f'{q}_{item["method"]}', item['method'], env, seed,
                           dict(item.get('overrides', {}), episodes=args.probe_episodes))
    elif args.suite == 'anchor_ablation':
        # rollout (main table) vs uniform-box vs 50/50 mix anchors at the same m, FedQHD-hetero
        for env, seed, src in itertools.product(envs, seeds, ['nominal', 'uniform', 'mix']):
            yield (f'anchors_{src}', 'fedqhd_hetero', env, seed, {'anchor_source': src})
    elif args.suite == 'tune_prox':
        for env, seed, mu in itertools.product(envs, seeds, [0.001, 0.01, 0.1]):
            yield (f'mu{mu}', 'fedprox_dqn', env, seed, {'fedprox_mu': mu})
    elif args.suite == 'tune':
        # FedQHD re-tuning on tuning seeds (disjoint from evaluation seeds). Stage 1 grid over
        # QHD agent parameters at the params file's interval K; stage 2 (--tune_stage 2) sweeps
        # K for the stage-1 winner, read from --tune_best.
        methods = {'q1': 'fedqhd_homo', 'q2': 'fedqhd_hetero'}
        if args.tune_stage == 1:
            grid = itertools.product([0.1, 0.2, 0.5], [0.9, 0.95, 0.99, 0.995], [0.95, 0.99],
                                     [0.5, 1.0])
            for env, seed, (lr, dec, g, rg), q in itertools.product(
                    envs, seeds, list(grid), ['q1', 'q2']):
                ov = {'qhd_lr': lr, 'qhd_exploration_decay': dec, 'qhd_discount': g,
                      'rff_gamma': rg, 'qhd_exploration_rate': 1.0, 'qhd_exploration_min': 0.001}
                yield (f'{q}_lr{lr}_dec{dec}_g{g}_rg{rg}', methods[q], env, seed, ov)
        else:
            with open(args.tune_best) as f:
                best = json.load(f)
            for key, cfg in best.items():
                env, q = key.split('/')
                if env not in envs:
                    continue
                for K, seed in itertools.product([10, 25, 50, 100], seeds):
                    yield (f'{q}_K{K}', methods[q], env, seed, dict(cfg, qhd_agg_interval=K))
    elif args.suite == 'budget':
        # Fixed total interaction budget split over N clients. With --budget_factor F the total
        # is F x the env's main-table budget (episodes_map), so N = F reproduces the main table
        # (rule confirmed 2026-10-07: F = 5, N in {1, 2, 5, 10}).
        ep_map = json.load(open(args.episodes_map)) if args.episodes_map else {}
        for env, seed, n in itertools.product(envs, seeds, args.n_grid):
            total = (int(args.budget_factor * ep_map[env]) if args.budget_factor
                     else args.total_episodes)
            eps = total // n
            for method in ['independent_qhd', 'fedqhd_homo', 'fedqhd_hetero']:
                yield (f'N{n}', method, env, seed, {'agent_num': n, 'episodes': eps})
    else:
        raise ValueError(args.suite)


def build_command(args, variant, method, env, seed, overrides):
    params = load_params(args.params_root, env, method, tuned_qhd=args.tuned_qhd)
    if args.anchor_m and method in ANCHOR_METHODS:
        params['anchor_set_size'] = args.anchor_m
    if args.dqn_preset == 'zoo' and method in DQN_FIXED_K:
        # protocol: DQN-family federation interval is fixed (not tuned, not inherited from the
        # per-method params files, which carry old values 10/25/50)
        params['dqn_agg_interval'] = args.dqn_k
    params.update({k: v for k, v in overrides.items() if k in PARAM_KEYS})
    if args.episodes is not None and 'episodes' not in overrides:
        params['episodes'] = args.episodes
    if args.episodes_map and 'episodes' not in overrides:
        with open(args.episodes_map) as f:
            params['episodes'] = int(json.load(f)[env])
    extra = {k: v for k, v in overrides.items() if k not in PARAM_KEYS}
    if extra.get('distill_interval') == 'MATCH_QHD':
        qhd = load_params(args.params_root, env, 'fedqhd_hetero')
        extra['distill_interval'] = qhd.get('qhd_agg_interval', 25)
    out_dir = args.out / (args.tag or args.suite) / variant / f'seed{seed}'
    cmd = [sys.executable, str(args.code_root / 'scripts' / 'run_single_method.py'),
           '--method', method, '--env', env, '--question', question_of(method),
           '--random_seed', str(seed), '--runs', '1', '--output_dir', str(out_dir)]
    for k, v in {**params, **extra}.items():
        if isinstance(v, bool):
            if v:
                cmd.append(f'--{k}')
        else:
            cmd += [f'--{k}', str(v)]
    if args.bootstrap_on_truncation:
        cmd.append('--bootstrap_on_truncation')
    if args.indep_no_reset:
        cmd.append('--indep_no_reset')
    if args.dqn_preset != 'legacy':
        cmd += ['--dqn_preset', args.dqn_preset]
    if args.no_diagnostics:
        cmd.append('--no_diagnostics')
    if args.dqn_device != 'auto':
        cmd += ['--dqn_device', args.dqn_device]
    return cmd, out_dir


def result_file(out_dir: Path, env: str, method: str) -> Path:
    sys.path.insert(0, str(ROOT / 'scripts'))
    from run_single_method import METHODS, result_filename, output_dir_for
    name = METHODS[method][0]
    q = 'q1_homogeneous' if question_of(method) == 'q1' else 'q2_heterogeneous'
    return out_dir / env / q / result_filename(name)


def run_job(cmd, log_path, gpu, code_root):
    env = dict(os.environ, PYTHONPATH=str(code_root), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', CUDA_VISIBLE_DEVICES=str(gpu))
    t0 = time.time()
    pid_file = Path(str(log_path) + '.pid')
    with open(log_path, 'w') as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=code_root)
        pid_file.write_text(str(proc.pid))
        rc = proc.wait()
    pid_file.unlink(missing_ok=True)
    return rc, time.time() - t0


def is_running(log_path: Path) -> bool:
    """True if another driver is currently running this job (live PID in <log>.pid)."""
    pid_file = Path(str(log_path) + '.pid')
    if not pid_file.exists():
        return False
    try:
        os.kill(int(pid_file.read_text()), 0)
        return True
    except (OSError, ValueError):
        return False


def parse_seeds(s):
    if '-' in s:
        a, b = s.split('-')
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(',')]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--suite', required=True, choices=['main', 'anchors', 'envhet', 'budget', 'tune',
                                                    'tune_prox', 'tune_compile', 'tune_oracle',
                                                    'probe', 'anchor_ablation'])
    p.add_argument('--episodes_map', type=Path, default=None,
                   help='json {env: episodes}: one training budget per environment, all methods')
    p.add_argument('--probe_spec', type=Path, default=None)
    p.add_argument('--probe_episodes', type=int, default=2000)
    p.add_argument('--tune_stage', type=int, default=1, choices=[1, 2])
    p.add_argument('--tune_best', type=Path, default=None, help='stage-1 winners (json)')
    p.add_argument('--envs', nargs='+', default=ENVS)
    p.add_argument('--seeds', type=parse_seeds, default=parse_seeds('42-51'))
    p.add_argument('--methods', nargs='+', default=None, help='Only run these method keys')
    p.add_argument('--episodes', type=int, default=None, help='Override episodes (debug)')
    p.add_argument('--m_grid', type=int, nargs='+', default=[200, 500, 1000, 2000, 5000, 10000])
    p.add_argument('--levels', type=float, nargs='+', default=[0.1, 0.3, 0.5])
    p.add_argument('--n_grid', type=int, nargs='+', default=[1, 2, 5, 10, 20])
    p.add_argument('--budget_factor', type=float, default=None,
                   help='budget suite: total client-episodes = factor x episodes_map[env]')
    p.add_argument('--total_episodes', type=int, default=3000,
                   help='budget suite: total client-episodes, split evenly over N clients')
    p.add_argument('--bootstrap_on_truncation', action='store_true')
    p.add_argument('--indep_no_reset', action='store_true')
    p.add_argument('--dqn_preset', default='legacy', choices=['legacy', 'zoo'])
    p.add_argument('--dqn_device', default='auto', choices=['auto', 'cpu', 'cuda'])
    p.add_argument('--dqn_k', type=int, default=25,
                   help='fixed federation interval of the DQN family (zoo preset)')
    p.add_argument('--max_heavy', type=int, default=16,
                   help='Max concurrent GPU-heavy jobs (heterogeneous QHD with anchors), all drivers')
    p.add_argument('--no_diagnostics', action='store_true',
                   help='Skip compile diagnostics (tuning/probe runs; less GPU memory)')
    p.add_argument('--tuned_qhd', action='store_true',
                   help='Untuned QHD baselines inherit FedQHD tuned parameters')
    p.add_argument('--anchor_m', type=int, default=None,
                   help='Anchor-set size for all anchor-based methods (e.g. > max D_i)')
    p.add_argument('--tag', default=None, help='Output sub-directory (default: suite name)')
    p.add_argument('--params_root', type=Path,
                   default=Path('/home/yuchen/projects/FedQHD/results/main_table_3seed'))
    p.add_argument('--out', type=Path, default=ROOT / 'results' / 'revision')
    p.add_argument('--jobs', type=int, default=8)
    p.add_argument('--code_root', type=Path, default=ROOT,
                   help='Run a frozen copy of the code (so edits do not affect running jobs)')
    p.add_argument('--gpus', type=int, default=2)
    p.add_argument('--dry_run', action='store_true')
    args = p.parse_args()

    jobs, external = [], []
    for variant, method, env, seed, ov in make_jobs(args):
        if args.methods and method not in args.methods:
            continue
        cmd, out_dir = build_command(args, variant, method, env, seed, ov)
        if result_file(out_dir, env, method).exists():
            continue
        log = args.out / 'logs' / (args.tag or args.suite) / variant / f'{env}_{method}_seed{seed}.log'
        if is_running(log):
            external.append((log, result_file(out_dir, env, method)))
            continue  # launched by another driver that is still running it
        jobs.append((cmd, log, f'{variant}/{env}/{method}/seed{seed}'))

    # Cells excluded by a documented decision (results/revision/exclusions.json, read on every
    # invocation so that already-running pipelines pick it up): never launched or retried.
    excl_file = args.out / 'exclusions.json'
    if excl_file.exists():
        excl = json.load(open(excl_file)).get(args.tag or args.suite, [])
        keys = {(e['variant'], e['env'], e['method']) for e in excl}
        before = len(jobs)
        jobs = [j for j in jobs if tuple(j[2].split('/')[:3]) not in keys]
        if before != len(jobs):
            print(f'excluded {before - len(jobs)} jobs by {excl_file.name}: {sorted(keys)}')

    # Longest-processing-time-first: rough relative cost per job, so the slowest runs
    # (LunarLander DQN, pooled DQN oracles) start first and do not form a tail.
    env_w = {'LunarLander': 4.0, 'Acrobot': 2.0, 'MountainCar': 1.5, 'CartPole': 1.0}
    meth_w = {'oracle_dqn_hetero': 12.0, 'oracle_dqn': 5.0, 'fedavg_dqn': 4.0, 'fedprox_dqn': 4.0,
              'distillation_dqn': 4.0, 'distillation_dqn_hetero': 4.0, 'fedhql_dqn_hetero': 4.0,
              'oracle_qhd_hetero': 2.0}
    def cost(job):
        cmd, _, tag = job
        method, env = tag.split('/')[-2], tag.split('/')[-3]
        eps = int(cmd[cmd.index('--episodes') + 1]) if '--episodes' in cmd else 600
        return env_w.get(env, 1.0) * meth_w.get(method, 1.0) * eps
    jobs.sort(key=cost, reverse=True)

    print(f'{len(jobs)} pending jobs in suite "{args.suite}"')
    if args.dry_run:
        for cmd, _, tag in jobs[:20]:
            print(tag, '\n   ', ' '.join(cmd))
        return

    for _, log, _ in jobs:
        log.parent.mkdir(parents=True, exist_ok=True)
    failed = 0
    # GPU memory, not CPU, limits the anchor-heavy heterogeneous QHD runs (m x D_i feature
    # matrices and solves on the GPU). Schedule longest-first, but start a heavy job only while
    # fewer than --max_heavy are running (including heavy jobs of other drivers); otherwise
    # take the next CPU-light job so no slot idles. GPUs alternate per heavy job.
    HEAVY = {'fedqhd_hetero', 'oracle_qhd_hetero', 'fedqhd_gd_hetero'}
    is_heavy = lambda tag: tag.split('/')[-2] in HEAVY
    ext_heavy = lambda: sum(1 for log, _ in external if is_running(log)
                            and any(h in str(log) for h in HEAVY))
    pending = list(jobs)
    running = {}   # future -> (tag, log, heavy)
    gpu_next = {True: 0, False: 0}
    done_n = 0
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        while pending or running:
            n_heavy = sum(1 for v in running.values() if v[2]) + ext_heavy()
            while pending and len(running) < args.jobs:
                pick = None
                for idx, (_, _, tag) in enumerate(pending):
                    if not is_heavy(tag) or not args.max_heavy or n_heavy < args.max_heavy:
                        pick = idx
                        break
                if pick is None:
                    break
                cmd, log, tag = pending.pop(pick)
                heavy = is_heavy(tag)
                gpu = gpu_next[heavy] % args.gpus
                gpu_next[heavy] += 1
                running[ex.submit(run_job, cmd, log, gpu, args.code_root)] = (tag, log, heavy)
                n_heavy += heavy
            if not running:      # only heavy jobs left and the limit is held by other drivers
                time.sleep(30)
                continue
            finished, _ = wait(list(running), timeout=60, return_when=FIRST_COMPLETED)
            for fut in finished:
                tag, log, _ = running.pop(fut)
                rc, dt = fut.result()
                failed += rc != 0
                done_n += 1
                print(f'[{done_n}/{len(jobs)}] {"OK " if rc == 0 else "ERR"} {tag}  ({dt / 60:.1f} min)'
                      + ('' if rc == 0 else f'  see {log}'), flush=True)
    if external:  # wait for jobs run by another driver, so "done" means all results exist
        print(f'waiting for {len(external)} jobs run by another driver', flush=True)
        while any(is_running(log) for log, _ in external):
            time.sleep(60)
        missing = [str(res) for _, res in external if not res.exists()]
        failed += len(missing)
        for m_ in missing:
            print(f'ERR external job without result: {m_}', flush=True)
    print(f'done, {failed} failed')


if __name__ == '__main__':
    main()
