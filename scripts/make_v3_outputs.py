#!/usr/bin/env python3
"""Paper tables and figures from the main_v3 results, for a fixed set of seeds.

Outputs in --out (default results/revision/v3_<n>seed/):
  table_main.tex          final training return (last 100 eps), mean ± std
  table_eval.tex          greedy evaluation return
  table_auc.tex           normalized AUC and episodes-to-threshold
  table_time.tex          wall-clock minutes per run
  tables.md               all of the above in Markdown
  curves_homo.pdf/png     learning curves, homogeneous setting (1 x 4 envs)
  curves_hetero.pdf/png   learning curves, heterogeneous setting (1 x 4 envs)
  coverage.json           which (method, cell, seed) results were used / missing

Only cells where every listed seed has a result are filled; others show "--".
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ENVS = ['CartPole', 'Acrobot', 'LunarLander', 'MountainCar']
QDIR = {'q1': 'q1_homogeneous', 'q2': 'q2_heterogeneous'}
THRESHOLD = {'CartPole': 475.0, 'Acrobot': -100.0, 'LunarLander': 200.0, 'MountainCar': -110.0}

# (label, {setting: (variant, result method_name)}, is_reference)
ROWS = [
    ('Indep. QHD', {'q1': ('default', 'Independent QHD'),
                    'q2': ('default', 'Independent QHD (Heterogeneous)')}, False),
    ('Trunc. FedAvg-QHD', {'q2': ('default', 'Truncate FedAvg-QHD')}, False),
    ('FedAvg-DQN', {'q1': ('default', 'FedAvg-DQN')}, False),
    ('FedProx-DQN', {'q1': ('default', 'FedProx-DQN')}, False),
    ('Distill. FedDQN (KL)', {'q1': ('default', 'Distillation FedDQN'),
                              'q2': ('default', 'Distillation FedDQN (Hetero)')}, False),
    ('Distill. FedDQN (MSE, matched)', {'q2': ('distill_matched', 'Distillation FedDQN (Hetero)')}, False),
    ('FedHPD', {'q2': ('default', 'FedHPD')}, False),
    ('FedHQL (DQN clients)', {'q2': ('default', 'FedHQL (DQN clients)')}, False),
    ('FedHQL (QHD clients)', {'q2': ('default', 'FedHQL (QHD clients)')}, False),
    ('FedQHD-GD', {'q2': ('default', 'FedQHD-GD (10 steps)')}, False),
    ('FedQHD', {'q1': ('default', 'FedQHD (Homogeneous)'),
                'q2': ('default', 'FedQHD (Heterogeneous)')}, False),
    ('FedQHD ($K{=}1$)', {'q1': ('default', 'Oracle QHD')}, True),
    ('Pooled QHD$^\\dagger$', {'q1': ('default', 'Pooled QHD'),
                               'q2': ('default', 'Oracle QHD (Heterogeneous)')}, True),
    ('Pooled DQN$^\\dagger$', {'q1': ('default', 'Oracle DQN'),
                               'q2': ('default', 'Oracle DQN (Heterogeneous)')}, True),
]

# Learning-curve figures: fixed categorical slots (reference palette, light mode), color follows
# the method; pooled references are neutral gray (dashed / dotted).
SLOT = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']
FIG = {
    'q1': [('FedQHD', SLOT[0]), ('Indep. QHD', SLOT[1]), ('Distill. FedDQN (KL)', SLOT[2]),
           ('FedAvg-DQN', SLOT[3]), ('FedProx-DQN', SLOT[4])],
    'q2': [('FedQHD', SLOT[0]), ('Indep. QHD', SLOT[1]), ('Distill. FedDQN (KL)', SLOT[2]),
           ('FedQHD-GD', SLOT[5]), ('FedHQL (QHD clients)', SLOT[6]), ('FedHQL (DQN clients)', SLOT[7]),
           ('FedHPD', SLOT[3])],
}
REFS = [('Pooled QHD$^\\dagger$', '#52514e', '--'), ('Pooled DQN$^\\dagger$', '#8a8984', ':')]


def load(suite_dir: Path, seeds):
    data = {}   # (variant, env, q, method_name) -> {seed: record}
    for f in glob.glob(str(suite_dir / '*' / 'seed*' / '*' / 'q*' / '*.json')):
        p = Path(f)
        if p.name.startswith('params_') or p.name == 'summary.json':
            continue
        seed = int(p.parents[2].name[4:])
        env = p.parents[1].name
        if seed not in seeds or env not in ENVS:
            continue
        rec = json.load(open(f))
        data.setdefault((p.parents[3].name, env, p.parent.name, rec['method_name']), {})[seed] = rec
    return data


OVERRIDES = {}   # (env, q, method_name) -> legacy override entry (see legacy_overrides.json)


def load_overrides(suite_dir: Path):
    f = suite_dir.parent / 'legacy_overrides.json'
    if f.exists():
        for e in json.load(open(f)).get(suite_dir.name, []):
            OVERRIDES[(e['env'], e['q'], e['method_name'])] = e


def cell(data, spec, env, q, seeds):
    if q not in spec:
        return None
    variant, name = spec[q]
    recs = data.get((variant, env, QDIR[q], name), {})
    if not all(s in recs for s in seeds):
        ov = OVERRIDES.get((env, q, name)) if variant == 'default' else None
        return {'override': ov} if ov else 'missing'
    return [recs[s] for s in seeds]


def override_stats(ov, kind):
    """Legacy-protocol cell: only the final return and the time are available."""
    if kind == 'final':
        return (float(ov['final_mean']), float(ov['final_std']))
    if kind == 'time':
        t = ov['seed_training_minutes']
        return (float(np.mean(t)), float(np.std(t, ddof=1)))
    return None


def stats(recs, env, kind):
    if kind == 'final':
        x = [np.mean(r['reward_history'][-100:]) for r in recs]
    elif kind == 'eval':
        x = [r['evaluation']['eval_return'] for r in recs if r.get('evaluation')]
    elif kind == 'auc':
        x = [np.mean(r['reward_history']) for r in recs]
    elif kind == 'time':
        x = [r['training_time'] / 60 for r in recs]
    elif kind == 'thr':
        hit = []
        for r in recs:
            rr = np.asarray(r['reward_history'], float)
            ma = np.convolve(rr, np.ones(100) / 100, 'valid')
            k = np.nonzero(ma >= THRESHOLD[env])[0]
            hit.append(int(k[0] + 100) if k.size else None)
        return hit
    return (float(np.mean(x)), float(np.std(x, ddof=1)) if len(x) > 1 else 0.0) if x else None


def build_table(data, seeds, kind, caption, label):
    cols = [(e, q) for e in ENVS for q in ['q1', 'q2']]
    vals = {}
    for lab, spec, ref in ROWS:
        for env, q in cols:
            c = cell(data, spec, env, q, seeds)
            if isinstance(c, dict):
                vals[(lab, env, q)] = ('override', override_stats(c['override'], kind),
                                       c['override']['mark'])
            else:
                vals[(lab, env, q)] = c if c in (None, 'missing') else stats(c, env, kind)
    # bold: best non-reference per column (higher is better; time: lower is better)
    best = {}
    for env, q in cols:
        cand = [(v[0], lab) for (lab, e, qq), v in vals.items() if e == env and qq == q
                and isinstance(v, tuple) and v and v[0] != 'override'
                and not dict((r[0], r[2]) for r in ROWS)[lab]]
        if cand:  # every method within 0.05 of the best value is bold (ties included)
            top = (min if kind == 'time' else max)(c[0] for c in cand)
            best[(env, q)] = {lab for v, lab in cand if abs(v - top) < 0.05}
    tex, md = [], []
    tex += [r'\begin{table*}[t]', r'\centering',
            rf'\caption{{{caption} Mean $\pm$ std over {len(seeds)} seeds; \textbf{{bold}}: best '
            r'non-reference method; $^\dagger$: pooled-data reference; --: not applicable.'
            + ((r' $^\ddagger$: legacy-protocol cell (' + '; '.join(
                f"{o['env']} {'Homo' if o['q'] == 'q1' else 'Hetero'}: {o['protocol']}"
                for o in OVERRIDES.values()) + ').') if OVERRIDES else '') + '}',
            rf'\label{{{label}}}', r'\setlength{\tabcolsep}{3pt}', r'\small',
            r'\resizebox{\textwidth}{!}{%', r'\begin{tabular}{l rr rr rr rr}', r'\toprule',
            ' & ' + ' & '.join(rf'\multicolumn{{2}}{{c}}{{{e}}}' for e in ENVS) + r' \\',
            r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}',
            'Method & ' + ' & '.join(['Homo & Hetero'] * 4) + r' \\', r'\midrule']
    md += [f'| Method | ' + ' | '.join(f'{e} {q}' for e, q in cols) + ' |',
           '|---' * (len(cols) + 1) + '|']
    prev_ref = False
    for lab, spec, ref in ROWS:
        if ref and not prev_ref:
            tex.append(r'\midrule')
        prev_ref = ref
        tc, mc = [], []
        for env, q in cols:
            v = vals[(lab, env, q)]
            if isinstance(v, tuple) and v and v[0] == 'override':
                st, mark = v[1], v[2]
                tc.append(f'${st[0]:.1f}{{\\scriptstyle\\pm{st[1]:.1f}}}^{{{mark}}}$' if st
                          else f'{{n/a}}$^{{{mark}}}$')
                mc.append((f'{st[0]:.1f} ± {st[1]:.1f}' if st else 'n/a') + ' (legacy‡)')
                continue
            if v is None:
                tc.append('{--}'); mc.append('–')
            elif v == 'missing':
                tc.append('{\\it pending}'); mc.append('pending')
            else:
                s = f'{v[0]:.1f}{{\\scriptstyle\\pm{v[1]:.1f}}}'
                tc.append(f'$\\mathbf{{{s}}}$' if lab in best.get((env, q), ()) else f'${s}$')
                bb = '**' if lab in best.get((env, q), ()) else ''
                mc.append(f'{bb}{v[0]:.1f} ± {v[1]:.1f}{bb}')
        tex.append(lab + ' & ' + ' & '.join(tc) + r' \\')
        md.append(f'| {lab} | ' + ' | '.join(mc) + ' |')
    tex += [r'\bottomrule', r'\end{tabular}', r'}', r'\end{table*}']
    return '\n'.join(tex) + '\n', '\n'.join(md) + '\n'


def build_thr_md(data, seeds):
    cols = [(e, q) for e in ENVS for q in ['q1', 'q2']]
    md = ['| Method | ' + ' | '.join(f'{e} {q} (≥{THRESHOLD[e]:g})' for e, q in cols) + ' |',
          '|---' * (len(cols) + 1) + '|']
    tex_rows = []
    for lab, spec, ref in ROWS:
        mc, tc = [], []
        for env, q in cols:
            c = cell(data, spec, env, q, seeds)
            if c is None:
                mc.append('–'); tc.append('{--}')
            elif c == 'missing':
                mc.append('pending'); tc.append('{\\it pending}')
            elif isinstance(c, dict):
                mc.append('n/a (legacy‡)'); tc.append('{n/a}')
            else:
                h = stats(c, env, 'thr')
                r_ = [x for x in h if x is not None]
                txt = f'{np.median(r_):.0f} ({len(r_)}/{len(h)})' if r_ else f'n.r. (0/{len(h)})'
                mc.append(txt); tc.append(txt.replace('/', '/'))
        md.append(f'| {lab} | ' + ' | '.join(mc) + ' |')
        tex_rows.append(lab + ' & ' + ' & '.join(tc) + r' \\')
    return '\n'.join(md) + '\n', tex_rows


def smooth(x, w=20):
    x = np.asarray(x, float)
    if len(x) < w:
        return x
    c = np.cumsum(np.insert(x, 0, 0.0))
    out = np.empty_like(x)
    out[w - 1:] = (c[w:] - c[:-w]) / w
    out[:w - 1] = [x[:i + 1].mean() for i in range(w - 1)]
    return out


def figure(data, seeds, q, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'axes.edgecolor': '#8a8984', 'axes.linewidth': 0.6,
                         'xtick.color': '#52514e', 'ytick.color': '#52514e',
                         'axes.labelcolor': '#0b0b0b', 'text.color': '#0b0b0b'})
    spec = {lab: s for lab, s, _ in ROWS}
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 2.9))
    handles = {}
    for ax, env in zip(axes, ENVS):
        series = [(lab, col, '-', 1.6) for lab, col in FIG[q]] + \
                 [(lab, col, ls, 1.2) for lab, col, ls in REFS]
        for lab, col, ls, lw in series:
            c = cell(data, spec[lab], env, q, seeds)
            if c in (None, 'missing') or isinstance(c, dict):
                continue
            n = min(len(r['reward_history']) for r in c)
            Y = np.array([smooth(r['reward_history'][:n]) for r in c])
            m, sd = Y.mean(0), Y.std(0)
            x = np.arange(1, n + 1)
            (h,) = ax.plot(x, m, color=col, ls=ls, lw=lw, label=lab.replace('$^\\dagger$', '†'))
            if ls == '-':
                ax.fill_between(x, m - sd, m + sd, color=col, alpha=0.12, lw=0)
            handles[lab] = h
        if not ax.lines:
            ax.text(0.5, 0.5, 'pending', ha='center', va='center', transform=ax.transAxes,
                    color='#8a8984')
        ax.set_title(env, fontsize=10)
        ax.set_xlabel('Episode')
        ax.grid(True, color='#e6e5e0', lw=0.5)
        for side in ['top', 'right']:
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel('Training return (mean over clients)')
    order = [lab for lab, _ in FIG[q]] + [lab for lab, _, _ in REFS]
    hs = [handles[l] for l in order if l in handles]
    if hs:
        fig.legend(hs, [h.get_label() for h in hs], loc='lower center', ncol=len(hs), frameon=False,
                   bbox_to_anchor=(0.5, -0.08), fontsize=8.5)
    fig.suptitle(('Homogeneous' if q == 'q1' else 'Heterogeneous') + f' encoders ({len(seeds)} seeds, '
                 'mean ± std, 20-episode smoothing)', fontsize=10, y=1.02)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(f'{path}.{ext}', dpi=160, bbox_inches='tight')
    plt.close(fig)


def coverage(data, seeds):
    out = {}
    for lab, spec, _ in ROWS:
        for q, (variant, name) in spec.items():
            for env in ENVS:
                got = sorted(data.get((variant, env, QDIR[q], name), {}))
                missing = sorted(set(seeds) - set(got))
                if missing and variant == 'default' and (env, q, name) in OVERRIDES:
                    out[f'{lab} | {env}/{q}'] = {'have': got, 'missing': [], 'legacy_override': True}
                    continue
                out[f'{lab} | {env}/{q}'] = {'have': got, 'missing': missing}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--suite_dir', type=Path, default=ROOT / 'results/revision/main_v3')
    ap.add_argument('--seeds', default='42,43,44')
    ap.add_argument('--out', type=Path, default=None)
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(',')]
    out = a.out or ROOT / f'results/revision/v3_{len(seeds)}seed'
    out.mkdir(parents=True, exist_ok=True)
    data = load(a.suite_dir, set(seeds))
    load_overrides(a.suite_dir)
    md = [f'# main_v3 tables ({len(seeds)} seeds: {seeds})', '']
    for kind, cap, lab, fn in [
            ('final', 'Final training return (mean of the last 100 episodes).', 'tab:main_results', 'table_main'),
            ('eval', 'Greedy-policy evaluation return at the end of training (10 episodes per client).',
             'tab:eval_results', 'table_eval'),
            ('auc', 'Normalized area under the learning curve (mean training return over the budget).',
             'tab:auc_results', 'table_auc'),
            ('time', 'Wall-clock training time (minutes per run).', 'tab:computation_cost', 'table_time')]:
        tex, m = build_table(data, seeds, kind, cap, lab)
        (out / f'{fn}.tex').write_text(tex)
        md += [f'## {cap}', '', m]
    thr_md, _ = build_thr_md(data, seeds)
    md += ['## Episodes to the solved threshold (median over seeds that reach it; reached/n)', '', thr_md]
    (out / 'tables.md').write_text('\n'.join(md))
    figure(data, seeds, 'q1', out / 'curves_homo')
    figure(data, seeds, 'q2', out / 'curves_hetero')
    cov = coverage(data, seeds)
    (out / 'coverage.json').write_text(json.dumps(cov, indent=1))
    n_missing = sum(1 for v in cov.values() if v['missing'])
    print(f'written to {out}; cells with missing seeds: {n_missing}/{len(cov)}')


if __name__ == '__main__':
    main()
