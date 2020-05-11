# Specify variables
from collections import defaultdict

import matplotlib as mpl
from haven import haven_results as hr

mpl.rc('image', cmap='tab20c')
import pandas as pd

# please define the path to the experiments
savedir_base = "/mnt/projects/bayesian-active-learning/synbols_ckpt"
from benchmarks.scripts.old_active_learning import EXP_GROUPS


def get_report(exp_name, filterby_list=None):
    exp_list = EXP_GROUPS[exp_name]
    # group the experiments based on a hyperparameter, for example, ['dataset']
    groupby_list = None
    verbose = 0

    # plot vars
    y_metrics = 'test_cls_report_accuracy'
    avg_across = 'seed'

    # get experiments
    rm = hr.ResultManager(exp_list=exp_list,
                          savedir_base=savedir_base,
                          filterby_list=filterby_list,
                          verbose=verbose
                          )
    scores = rm.get_score_lists()
    d = defaultdict(list)
    for exp, score in zip(rm.exp_list, scores):
        exp.pop('dataset')
        dfs = pd.DataFrame.from_records([{**exp, **sco} for sco in score])
        d[exp[avg_across]].append(dfs)

    result = {}
    for exp_lst in zip(*d.values()):
        gb = pd.concat(exp_lst).groupby('num_samples')
        assert all(
            exp_lst[-1]['heuristic'][0] == exp_lst[i]['heuristic'][0] for i in range(len(exp_lst)))
        heuristic = exp_lst[-1]['heuristic'][0]
        x_axis = gb['num_samples'].mean()
        y_mean = gb[y_metrics].mean()
        y_std = gb[y_metrics].std()
        result[heuristic] = pd.DataFrame(
            {'x_axis': x_axis, 'y_mean': y_mean, 'y_std': y_std, 'heuristic': heuristic})

    five_percent = int(0.05 * 100000)
    ten_percent = int(0.10 * 100000)
    twenty_percent = int(0.20 * 100000)

    for p_s, p in zip(['5%', '10%', '20%'], [five_percent, ten_percent, twenty_percent]):
        print(f"----------------{p_s}-------------------")
        for heuristic in ['bald', 'entropy', 'random']:
            if heuristic in result:
                row = result[heuristic][result[heuristic]['x_axis'] == p]
                s = f"{row['heuristic'].item()} {row['y_mean'].item():.5f}±{row['y_std'].item():.5f}"
                print(s)
