# Specify variables
import pickle
from collections import defaultdict
from pickle import UnpicklingError

import imageio
import matplotlib as mpl
from haven import haven_results as hr
from scipy.optimize import curve_fit

mpl.rc('image', cmap='tab20c')
import pandas as pd
import numpy as np

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({'font.size': 14})
linestyles = {True: (0, (3, 10, 1, 10)), False: (0, (5, 1))}
color = {'C-BALD': 'coral',
         'BALD': 'firebrick',
         'Entropy': 'deepskyblue',
         'C-Entropy': 'turquoise',
         'Random': 'yellowgreen'}


def plot(x, y, y_err, color, ls, label):
    plt.plot(x, y, label=label, linestyle=ls, color=color)
    plt.fill_between(x, y - y_err, y + y_err,
                     color=color, alpha=0.2)


def label(h, calib):
    return f"{'C-' if calib else ''}{'BALD' if h == 'bald' else h.capitalize()}"


def go(h, calib, df):
    lbl = label(h, calib)
    plot(df['x_axis'], df['y_mean'], df['y_std'], color[lbl], ls=linestyles[calib], label=lbl)


def run(title):
    plt.clf()
    for calib, df in [(True, r1), (False, r2)]:
        for h in ['bald', 'entropy', 'random']:
            if h == 'random' and calib:
                continue
            go(h, calib, df[h])
    plt.xlim(2000, 20000)
    plt.legend()
    plt.xticks(np.arange(2000, 20000, 4000))
    plt.xlabel('Dataset size')
    plt.ylabel('Test NLL')
    plt.savefig(f'{title}.png', dpi=1000)


# please define the path to the experiments
savedir_base = "/mnt/projects/bayesian-active-learning/synbols_ckpt"
from benchmarks.scripts.active_learning import EXP_GROUPS


def get_report(exp_name, filterby_list=None, metric='test_loss'):
    exp_list = EXP_GROUPS[exp_name]
    # group the experiments based on a hyperparameter, for example, ['dataset']
    groupby_list = None
    verbose = 0

    # plot vars
    y_metrics = metric
    avg_across = 'seed'

    # get experiments
    rm = hr.ResultManager(exp_list=exp_list,
                          savedir_base=savedir_base,
                          filterby_list=filterby_list,
                          verbose=verbose
                          )
    for i in range(5):
        try:
            scores = rm.get_score_lists()
        except UnpicklingError:
            if i > 3:
                raise
            continue
        break

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
        try:
            print(f"----------------{p_s}-------------------")
            for heuristic in ['bald', 'entropy', 'random']:
                if heuristic in result:
                    row = result[heuristic][result[heuristic]['x_axis'] == p]
                    s = f"{row['heuristic'].item()} {row['y_mean'].item():.5f}±{row['y_std'].item():.5f}"
                    print(s)
        except ValueError:
            print("Something happened")
    return result


VIS = False
fitting_err = []


def my_fun(x, a, b,c):
    return 1 / ((a * np.log(x)**c + b))


def fit(x, y, y_err):
    try:
        popt, pcov = curve_fit(my_fun, x, y, sigma=y_err, method='lm', maxfev=10000)
        return lambda x: my_fun(x, *popt)
    except Exception as e:
        return None


def get_early_stop(values, threshold, patience):
    if len(values) < patience:
        return None
    idx = values.argsort()
    idx_a, idx_b = idx[:2]
    if idx_a - idx_b > patience:
        return idx_a
    return None



def fit_and_get_residual(df, title):
    x = df['x_axis'].values
    y = df['y_mean'].values
    y_err = df['y_std'].values
    y_max = 2 * np.max(y)
    res = {}
    images = []
    for spl in np.linspace(0.1, 0.95, 10):
        split = int(spl * len(x))
        x_train, y_train,y_err_train = x[:split], y[:split], y_err[:split]
        x_test, y_test, y_err_test = x[split:], y[split:], y_err[split:]
        popt_fn = fit(x_train, y_train, y_err_train)
        if popt_fn is None:
            continue
        mae = np.abs(y_test - popt_fn(x_test)).mean()
        y_pred = popt_fn(x)
        es_pred = get_early_stop(y_pred, 0.002, 2)
        es_true = get_early_stop(y, 0.002, 2)

        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(np.arange(len(x)), y_pred, color='red', label='Estimated')
        ax.vlines(split, 0, y_max, color='green', label='Train-Test Split')
        ax.plot(np.arange(len(x)), y, color='blue', label='True')
        ax.legend(loc='best')
        ax.set_title(f"Train ratio : {spl:.2f}%, MAE={mae:.2f}")
        ax.set_ylabel('NLL')
        ax.set_ylim(0, y_max)

        if es_pred is not None and es_true is not None:
            ax.vlines([es_pred, es_true],
                      0, y_max, ['coral', 'turquoise'], label=['ES Pred', 'ES True'])

        fig.canvas.draw()  # draw the canvas, cache the renderer
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        images.append(image)
        res[spl] = mae
        fig.clf()
        plt.close(fig)
        plt.clf()

    imageio.mimsave(f'./{title}.gif', images, fps=1)
    return res

all_data = []
for exp in ['active_char_default_calibrated',
            'active_char_calibrated',
            'active_char_pixel_noise_calibrated',
            'active_char_label_noise_calibrated',
            'active_char_large_trans_calibrated',
            'active_char_partly_occluded_calibrated'
            ]:
    print(f"--------------{exp}---------------------")
    print(f"Calibrate True")
    r1 = get_report(exp, filterby_list={'calibrate': True}, metric='test_loss')
    print(f"Calibrate False")
    r2 = get_report(exp, filterby_list={'calibrate': False}, metric='test_loss')
    print(f"--------------------------------------")
    continue
    if VIS:
        run(exp)
    else:
        fitting_err.append(fit_and_get_residual(r1['bald'], title=f'{exp}_Exp1'))
        fitting_err.append(fit_and_get_residual(r2['bald'], title=f'{exp}_Exp2'))
    all_data.append(r1['bald'])
    all_data.append(r2['bald'])

df = pd.DataFrame.from_records(fitting_err)
print(df)
pickle.dump(all_data, open('all-data.pkl', 'wb'))
