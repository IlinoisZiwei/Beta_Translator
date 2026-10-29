"""Render measured held-out results and validation confidence tradeoffs."""
import argparse
import json
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/beta-translator-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    test = json.loads((args.package/'test_report.json').read_text())
    val = json.loads((args.package/'validation_report.json').read_text())
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.3), layout='constrained')
    colors = ['#8695a6', '#187c85']
    for offset, (name, color) in enumerate(zip(('baseline', 'model'), colors)):
        metrics = test[name]['aggregate']
        values = [metrics['accuracy']*100, metrics['event_f1']*100]
        bars = axes[0].bar(np.arange(2)+(offset-.5)*.32, values, .3, label=name.title(), color=color)
        axes[0].bar_label(bars, fmt='%.1f', padding=3)
    axes[0].set(xticks=[0, 1], xticklabels=['Frame accuracy', 'Event F1'], ylim=(0, 105), ylabel='Percent',
                title='Held-out people: p9 and p10')
    axes[0].legend(frameon=False, loc='lower right')
    curve = [p for p in val['coverage_curve'] if p['accepted_accuracy'] is not None]
    axes[1].plot([p['coverage']*100 for p in curve], [p['accepted_accuracy']*100 for p in curve],
                 'o-', color=colors[1], label='Validation thresholds')
    deployed = test['model']['aggregate']
    if deployed['accepted_accuracy'] is not None:
        axes[1].scatter([deployed['coverage']*100], [deployed['accepted_accuracy']*100], color='#df7e2a',
                        marker='*', s=150, label='Test at locked threshold', zorder=3)
    axes[1].axhline(97, color='#9ca3af', ls='--', lw=1)
    axes[1].set(xlabel='Accepted limb-frames (%)', ylabel='Accepted-label accuracy (%)',
                title='Confidence / review tradeoff', xlim=(0, 102))
    axes[1].legend(frameon=False, fontsize=8)
    for kind, color in [('arrival', '#187c85'), ('release', '#df7e2a')]:
        errors = np.array([e for clip in test['model']['clips'] for e in clip['event'][kind+'_errors_sec']])
        if len(errors):
            errors.sort()
            axes[2].step(errors, np.arange(1, len(errors)+1)/len(errors)*100, where='post', color=color,
                         label=f'{kind.title()} (n={len(errors)})')
    axes[2].set(xlabel='Absolute boundary error (seconds)', ylabel='Matched boundaries (%)',
                title='Timing, conditional on event match', xlim=(0, 1))
    axes[2].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.15); ax.set_axisbelow(True)
    fig.suptitle('Beta Translator | public-data contact model evaluation', fontsize=15)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)


if __name__ == '__main__': main()
