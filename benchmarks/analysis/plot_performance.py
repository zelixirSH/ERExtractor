#!/usr/bin/env python3
"""
advanced_analysis.py

An advanced script to analyze 5-fold cross-validation results for a molecule
image-to-SMILES model. This script aims to demonstrate that fine-tuning on a small,
manually annotated dataset leads to significant in-domain adaptation with
controlled and minimal performance degradation on out-of-domain public benchmarks.

Generates:
- summary_statistics.csv
- 1_in_domain_improvement.png
- 2_cross_dataset_performance_change.png
- 3_fold4_learning_curves.png
"""
import sys
import json
import os
import glob
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats


sns.set_theme(style="whitegrid")


plt.rcParams['font.size'] = 16
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 16
plt.rcParams['figure.titlesize'] = 22


BASE_DIRS = [
    '/sugon_store/mahaohui/mahaohui/OpenChemIE-main/MolScribe-main/output/uspto/swin_base_char_aux_1m680k_custom_fold_0_epoch100_dataindigo5_traindata_regulization_newsplit_graph_MMD',
    '/sugon_store/mahaohui/mahaohui/OpenChemIE-main/MolScribe-main/output/uspto/swin_base_char_aux_1m680k_custom_fold_1_epoch100_dataindigo5_traindata_regulization_newsplit_graph_MMD',
    '/sugon_store/mahaohui/mahaohui/OpenChemIE-main/MolScribe-main/output/uspto/swin_base_char_aux_1m680k_custom_fold_2_epoch100_dataindigo5_traindata_regulization_newsplit_graph_MMD',
    '/sugon_store/mahaohui/mahaohui/OpenChemIE-main/MolScribe-main/output/uspto/swin_base_char_aux_1m680k_custom_fold_3_epoch100_dataindigo5_traindata_regulization_newsplit_graph_MMD',
    '/sugon_store/mahaohui/mahaohui/OpenChemIE-main/MolScribe-main/output/uspto/swin_base_char_aux_1m680k_custom_fold_4_epoch100_dataindigo5_traindata_regulization_newsplit_graph_MMD'
]


METRICS = [
    'tanimoto', 'canon_smiles', 'graph', 'chiral',
]


DATASETS = ['valid', 'acs', 'chemdraw', 'CLEF', 'indigo', 'UOB', 'USPTO']


KEY_METRICS_FOR_STORY = ['canon_smiles', 'tanimoto', 'graph']




def collect_5fold_data(base_dirs, datasets, metrics):

    print("Collecting 5-fold data for 'ep0' and 'best' phases...")
    data = {ds: {m: {'ep0': [], 'best': []} for m in metrics} for ds in datasets}

    for i, fold_dir in enumerate(base_dirs):
        for ds in datasets:
            for phase in ['ep0', 'best']:
                if ds == 'valid':

                    json_pattern = os.path.join(fold_dir, phase, f'eval_scores_fold_{i}_valid_best.json')
                else:

                    json_pattern = os.path.join(fold_dir, phase, f'eval_scores_{ds}_best.json')
                
                json_files = glob.glob(json_pattern)
                if not json_files:
                    print(f"Warning: Could not find JSON for fold {i}, dataset '{ds}', phase '{phase}'. Pattern: {json_pattern}")
 
                    for m in metrics:
                        data[ds][m][phase].append(None)
                    continue

                with open(json_files[0]) as f:
                    scores = json.load(f)
                
                for m in metrics:
                    data[ds][m][phase].append(scores.get(m, None))


    for ds in datasets:
        for m in metrics:
            for phase in ['ep0', 'best']:
                valid_scores = [s for s in data[ds][m][phase] if s is not None]
                if len(valid_scores) < len(base_dirs):
                    print(f"Warning: Missing data for {ds}/{m}/{phase}. Found {len(valid_scores)}/{len(base_dirs)} points.")
                data[ds][m][phase] = valid_scores

    print("Data collection complete.")
    return data

def collect_fold4_epoch_data(fold4_dir, datasets, metric):

    print(f"Collecting epoch-wise data for Fold 0, metric '{metric}'...")
    epoch_dirs = glob.glob(os.path.join(fold4_dir, 'ep*'))
    epoch_dirs.append(os.path.join(fold4_dir, 'best')) # 'best' 也可以看作一个时间点

    results = []
    for epoch_dir in epoch_dirs:
        dir_name = os.path.basename(epoch_dir)
        if dir_name == 'ep0':
            epoch = 0
        elif dir_name.startswith('ep'):
            epoch = int(re.search(r'ep(\d+)', dir_name).group(1))
        elif dir_name == 'best':
            epoch = 20

            continue
        
        for ds in datasets:
            if ds == 'valid':
                json_path = os.path.join(epoch_dir, 'eval_scores_fold_4_valid_best.json')
                print(json_path)
            else:
                json_path = os.path.join(epoch_dir, f'eval_scores_{ds}_best.json')

            if os.path.exists(json_path):
                with open(json_path) as f:
                    scores = json.load(f)
                score = scores.get(metric)
                if score is not None:
                    results.append({'epoch': epoch, 'dataset': ds, 'score': score})
                    print('epoch', epoch, 'dataset', ds, 'score', score)

    if not results:
        print("Warning: No epoch data found for Fold 0. Skipping learning curve plot.")
        return None
        
    df = pd.DataFrame(results).sort_values('epoch')
    print("Epoch data collection complete.")
    return df



def calculate_summary_stats(five_fold_data, datasets, metrics):

    results = []
    for ds in datasets:
        for m in metrics:
            best_arr = np.array(five_fold_data[ds][m]['best'])
            ep0_arr = np.array(five_fold_data[ds][m]['ep0'])

            if len(best_arr) < 2 or len(ep0_arr) < 2 or len(best_arr) != len(ep0_arr):
                continue

            mean_best = best_arr.mean()
            mean_ep0 = ep0_arr.mean()
            

            ci = stats.sem(best_arr) * stats.t.ppf((1 + 0.95) / 2., len(best_arr) - 1) if len(best_arr) > 1 else 0
            

            t, p = stats.ttest_rel(best_arr, ep0_arr)
            
            results.append({
                'dataset': ds,
                'metric': m,
                'mean_ep0': mean_ep0,
                'mean_best': mean_best,
                'ci_best': ci,
                'p_value': p,
                'change': mean_best - mean_ep0
            })
            
    df = pd.DataFrame(results)
    df['change_pct'] = (df['change'] / df['mean_ep0']) * 100
    df.to_csv('summary_statistics.csv', index=False)
    print("Saved summary_statistics.csv")
    return df



def plot_in_domain_improvement(summary_df):

    print("Generating plot 1: In-Domain Improvement...")
    df_valid = summary_df[summary_df['dataset'] == 'valid'].copy()
    

    metrics = df_valid['metric'].unique()
    x = np.arange(len(metrics))
    width = 0.35  
    

    ep0_means = [df_valid[df_valid['metric'] == m]['mean_ep0'].values[0] for m in metrics]
    best_means = [df_valid[df_valid['metric'] == m]['mean_best'].values[0] for m in metrics]
    p_values = [df_valid[df_valid['metric'] == m]['p_value'].values[0] for m in metrics]
    
    plt.figure(figsize=(12, 7))
    

    plt.bar(x - width/2, ep0_means, width, label='Pre-trained (ep0)', color='#aec6cf')
    plt.bar(x + width/2, best_means, width, label='Fine-tuned (best)', color='#ff7f0e')

    for i, p_val in enumerate(p_values):
        if p_val < 0.05:

            y_max = max(ep0_means[i], best_means[i])
            plt.text(i, y_max + 0.01, '*', ha='center', va='bottom', color='black', fontsize=20)

    plt.xticks(x, metrics, rotation=45, ha='right')
    plt.ylabel('Mean Score')
    plt.xlabel('Metric')
    plt.title('In-Domain Performance on Validation Set (5-Fold Avg)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('1_in_domain_improvement.png', dpi=300)
    plt.savefig('1_in_domain_improvement.svg')
    print("Saved 1_in_domain_improvement.png")
    plt.close()

def plot_cross_dataset_performance_change(summary_df, key_metrics):

    print("Generating plot 2: Cross-Dataset Performance Change...")
    df_plot = summary_df[summary_df['metric'].isin(key_metrics)]

    g = sns.catplot(
        data=df_plot,
        x='change_pct',
        y='dataset',
        col='metric',
        kind='bar',
        height=5,
        aspect=1.2,
        palette='viridis',
        sharex=False
    )
    g.fig.suptitle('Performance Change After Fine-tuning (Fine-tuned vs. Pre-trained)', y=1.03)
    g.set_axis_labels('Performance Change (%)', 'Dataset')
    g.set_titles("Metric: {col_name}")


    for ax in g.axes.flat:
        ax.axvline(0, color='red', linestyle='--', linewidth=1.5)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig('2_cross_dataset_performance_change.png', dpi=300)
    plt.savefig('2_cross_dataset_performance_change.svg')
    print("Saved 2_cross_dataset_performance_change.png")
    plt.close()
    
def plot_learning_curves(df_epoch, title_metric, datasets_to_plot):

    if df_epoch is None or df_epoch.empty:
        return
    print("Generating plot 3: Fold 4 Learning Curves (Relative Change)...")
    

    df_plot = df_epoch[df_epoch['dataset'].isin(datasets_to_plot)].copy()
    

    baseline_scores = {}
    for dataset in datasets_to_plot:
        ep0_score = df_plot[(df_plot['dataset'] == dataset) & (df_plot['epoch'] == 0)]['score']
        if len(ep0_score) > 0:
            baseline_scores[dataset] = ep0_score.values[0]
        else:
            print(f"Warning: No ep0 score found for {dataset}")
            baseline_scores[dataset] = None
    

    df_plot['relative_change'] = df_plot.apply(
        lambda row: ((row['score'] - baseline_scores[row['dataset']]) / baseline_scores[row['dataset']]) * 100 
        if baseline_scores[row['dataset']] is not None else None, 
        axis=1
    )
    

    df_plot = df_plot.dropna(subset=['relative_change'])
    

    all_epochs = sorted(df_plot['epoch'].unique())
    
    
    epoch_positions = {}
    position_counter = 0

    for epoch in all_epochs:
        epoch_positions[epoch] = position_counter
        position_counter += 1
    

    df_plot['x_position'] = df_plot['epoch'].apply(lambda x: epoch_positions[x])
    

    plt.figure(figsize=(16, 10))
    

    valid_style = {'marker': 'o', 'linestyle': '-', 'linewidth': 3, 'markersize': 10}
    test_style = {'marker': 's', 'linestyle': '--', 'linewidth': 2.5, 'markersize': 8}
    

    for dataset in datasets_to_plot:
        dataset_data = df_plot[df_plot['dataset'] == dataset]
        if dataset == 'valid':
            plt.plot(dataset_data['x_position'], dataset_data['relative_change'], 
                     label='Validation (In-domain)', **valid_style)
        else:
            plt.plot(dataset_data['x_position'], dataset_data['relative_change'], 
                     label=f'{dataset} (Out-of-domain)', **test_style)
    

    plt.axhline(y=0, color='gray',linestyle = '-', alpha=0.7, linewidth=2)
    

    plt.xlabel('Training Epoch')
    plt.ylabel('Relative Change (%)')
    plt.title(f'Fold 0 Learning Curves: Relative Performance Change\n(Metric: {title_metric})')
    

    x_ticks = []
    x_labels = []
    

    for epoch in all_epochs:
        x_ticks.append(epoch_positions[epoch])
        x_labels.append(str(epoch))
    
    plt.xticks(x_ticks, x_labels)
    

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    

    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    

    plt.xlim(min(x_ticks) - 0.5, max(x_ticks) + 0.5)
    
    plt.tight_layout()
    plt.savefig(f'3_fold4_learning_curves_{title_metric}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'3_fold4_learning_curves_{title_metric}.svg')
    print(f"Saved 3_fold4_learning_curves_{title_metric}.png")
    plt.close()



if __name__ == "__main__":

    five_fold_data = collect_5fold_data(BASE_DIRS, DATASETS, METRICS)


    summary_df = calculate_summary_stats(five_fold_data, DATASETS, METRICS)


    plot_in_domain_improvement(summary_df)
    plot_cross_dataset_performance_change(summary_df, KEY_METRICS_FOR_STORY)


    metrics_to_plot = ['tanimoto', 'canon_smiles', 'graph']
    metric_titles = {
        'canon_smiles': 'Canonical SMILES Accuracy',
        'tanimoto': 'Tanimoto Similarity',
        'graph': 'Graph Match',
        'graph tanimoto': 'Graph Tanimoto',
        'post tanimoto': 'Post Tanimoto'
    }

    for metric in metrics_to_plot:
        fold4_epoch_df = collect_fold4_epoch_data(
            BASE_DIRS[4],
            datasets=['valid', 'acs', 'CLEF', 'indigo', 'UOB', 'USPTO', 'chemdraw'],
            metric=metric
        )

        if fold4_epoch_df is not None and not fold4_epoch_df.empty:
            plot_learning_curves(
                fold4_epoch_df,
                title_metric=metric_titles[metric],
                datasets_to_plot=['valid', 'acs', 'CLEF', 'indigo', 'UOB', 'USPTO', 'chemdraw']
            )
        else:
            print(f"Skipping learning curve for {metric} due to missing data.")

    print("\nAnalysis complete. All files have been generated.")