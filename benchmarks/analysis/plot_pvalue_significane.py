
#!/usr/bin/env python3
"""
compare_folds.py
Usage:
python compare_folds.py dir1 dir2 dir3 dir4 dir5
Generates:
- summary.csv          : statistics
- metrics_ci_plot.png  : 95 % CI plot
- pvalue_bar.png       : p-value bar
"""
import sys, json, os, glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
sns.set_theme(style="whitegrid")


dirs = ['/sugon_store/mahaohui/mahaohui/OpenChemIE-main/MolScribe-main/output/uspto/swin_base_char_aux_1m680k_custom_fold_0_epoch100_dataindigo5_traindata_regulization_newsplit_graph_MMD',
        '/sugon_store/mahaohui/mahaohui/OpenChemIE-main/MolScribe-main/output/uspto/swin_base_char_aux_1m680k_custom_fold_1_epoch100_dataindigo5_traindata_regulization_newsplit_graph_MMD',
        '/sugon_store/mahaohui/mahaohui/OpenChemIE-main/MolScribe-main/output/uspto/swin_base_char_aux_1m680k_custom_fold_2_epoch100_dataindigo5_traindata_regulization_newsplit_graph_MMD',
        '/sugon_store/mahaohui/mahaohui/OpenChemIE-main/MolScribe-main/output/uspto/swin_base_char_aux_1m680k_custom_fold_3_epoch100_dataindigo5_traindata_regulization_newsplit_graph_MMD',
        '/sugon_store/mahaohui/mahaohui/OpenChemIE-main/MolScribe-main/output/uspto/swin_base_char_aux_1m680k_custom_fold_4_epoch100_dataindigo5_traindata_regulization_newsplit_graph_MMD']
metrics = ['tanimoto', 'canon_smiles', 'graph', 'chiral',
           'post_smiles', 'post_graph', 'post_chiral', 'post_tanimoto',
           'graph_smiles', 'graph_graph', 'graph_chiral', 'graph_tanimoto']
data = {m: {'ep0': [], 'best': []} for m in metrics}

for fold_dir in dirs:
    ep0_json  = glob.glob(os.path.join(fold_dir, 'ep0',  'eval_scores_fold_*_valid_best.json'))[0]
    best_json = glob.glob(os.path.join(fold_dir, 'best', 'eval_scores_fold_*_valid_best.json'))[0]
    with open(ep0_json)  as f: ep0  = json.load(f)
    with open(best_json) as f: best = json.load(f)
    for m in metrics:
        data[m]['ep0'].append(ep0[m])
        data[m]['best'].append(best[m])

# ---------- 2. 计算统计 ----------
results = []
for m in metrics:
    best_arr = np.array(data[m]['best'])
    ep0_arr  = np.array(data[m]['ep0'])
    mean = best_arr.mean()
    ci   = stats.sem(best_arr) * stats.t.ppf((1 + 0.95)/2, len(best_arr)-1)
    t, p = stats.ttest_rel(best_arr, ep0_arr)
    results.append({'metric': m, 'mean_best': mean,
                    'ci_lower': mean-ci, 'ci_upper': mean+ci, 'p_value': p})
df_stats = pd.DataFrame(results)
df_stats.to_csv('summary.csv', index=False)
print('Saved summary.csv')

# ---------- 3. 95 % CI plot ----------
plt.figure(figsize=(6, 4))
sns.pointplot(data=df_stats, x='metric', y='mean_best', color='steelblue',
              markers='o', linestyles='', ci=95)
plt.errorbar(x=df_stats.index, y=df_stats['mean_best'],
             yerr=(df_stats['ci_upper'] - df_stats['ci_lower'])/2,
             fmt='none', color='steelblue', capsize=5)
plt.xticks(rotation=45, ha='right')
plt.ylabel('Mean Score (best epoch)')
plt.title('95 % Confidence Interval Over 5 Folds')
plt.tight_layout()
plt.savefig('metrics_ci_plot.png', dpi=300)
plt.savefig('metrics_ci_plot.svg')
print('Saved metrics_ci_plot.png')

# ---------- 4. p-value bar ----------
plt.figure(figsize=(6, 3))
colors = ['green' if p < 0.05 else 'gray' for p in df_stats['p_value']]
sns.barplot(x='metric', y='p_value', data=df_stats, palette=colors)
plt.axhline(0.05, color='red', linestyle='--', label='p = 0.05')
plt.yscale('log')
plt.xticks(rotation=45, ha='right')
plt.ylabel('p-value (paired t-test)')
plt.title('Significance of Improvement vs Pre-trained')
plt.legend()
plt.tight_layout()
plt.savefig('pvalue_bar.png', dpi=300)
plt.savefig('pvalue_bar.svg')
print('Saved pvalue_bar.png')


sig_metrics = df_stats[df_stats['p_value'] < 0.05]['metric'].tolist()
if sig_metrics:
    long = []
    for m in sig_metrics:
        for phase, label, vals in [('ep0',  'before finetuning', data[m]['ep0']),
                                   ('best', 'after finetuning',  data[m]['best'])]:
            for v in vals:
                long.append({'metric': m, 'phase': label, 'score': v})
    df_sig = pd.DataFrame(long)

    plt.figure(figsize=(6, 4))
    ax = sns.barplot(data=df_sig, x='metric', y='score', hue='phase',
                     palette={'before finetuning': '#aec6cf',
                              'after finetuning': '#ff7f0e'})
    plt.ylabel('Score')
    plt.title('Metrics with Significant Improvement (p < 0.05)')

    plt.legend(title='Phase', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig('significant_improvement.png', dpi=300, bbox_inches='tight')
    plt.savefig('significant_improvement.svg')
    print('Saved significant_improvement.png')


ns_metrics = df_stats[df_stats['p_value'] >= 0.05]['metric'].tolist()
if ns_metrics:
    long = []
    for m in ns_metrics:
        for phase, label, vals in [('ep0',  'before finetuning', data[m]['ep0']),
                                   ('best', 'after finetuning',  data[m]['best'])]:
            for v in vals:
                long.append({'metric': m, 'phase': label, 'score': v})
    df_ns = pd.DataFrame(long)

    plt.figure(figsize=(6, 3))
    sns.barplot(data=df_ns, x='metric', y='score', hue='phase',
                palette={'before finetuning': '#aec6cf',
                         'after finetuning': '#ff7f0e'})
    plt.ylabel('Score')
    plt.title('Metrics without Significant Change (p ≥ 0.05)')
    plt.legend(title='Phase')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('no_significant_change.png', dpi=300)
    plt.savefig('no_significant_change.svg')
    print('Saved no_significant_change.png')