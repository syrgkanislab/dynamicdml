import joblib
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from collections import OrderedDict

def sanitize(str):
    return "".join([c for c in str if c.isalpha() or c.isdigit() or c==' ']).rstrip()

def helper(df, prop):
    return df[prop] if prop in df else 0

def mean_stderr(a):
    return np.nanmean(a), np.nanstd(a) / np.sqrt(np.sum(~np.isnan(a)))

def metrics(truth, point, std):
    metrics = OrderedDict()
    metrics['coverage90'] = mean_stderr((point - 1.645 * std <= truth) & (point + 1.645 * std >= truth))
    metrics['coverage95'] = mean_stderr((point - 1.96 * std <= truth) & (point + 1.96 * std >= truth))
    metrics['coverage99'] = mean_stderr((point - 2.58 * std <= truth) & (point + 2.58 * std >= truth))
    metrics['stderr'] = mean_stderr(std)
    metrics['bias'] = mean_stderr(point - truth)
    metrics['mse'] = mean_stderr((point - truth)**2)
    metrics['mae'] = mean_stderr(np.abs(point - truth))
    metrics['mape'] = mean_stderr(np.abs(point - truth) / np.abs(truth))
    return metrics

def summarize_samples(res, param_list, plot=False):
    n_instances = len(res)
    n_samples = len(res[0])
    n_periods = len(res[0][0]['true']['params'])
    summary = OrderedDict()
    for instance in range(n_instances):
        summary[instance] = OrderedDict()
        for feat in res[instance][0]['true']:
            if feat == 'params':
                continue
            summary[instance][feat] = OrderedDict()
            truth = res[instance][0]['true'][feat]
            summary[instance][feat]['true'] = truth
            for method in res[instance][0]:
                if method == 'true':
                    continue
                if feat in res[instance][0][method]:
                    point = np.array([res[instance][t][method][feat][0] for t in range(n_samples)])
                    std = np.array([res[instance][t][method][feat][1] for t in range(n_samples)])
                    summary[instance][feat][sanitize(method)] = metrics(truth, point, std)
                    if plot:
                        plt.hist(point, label=method, alpha=.4)
            if plot:
                plt.axvline(x = truth, label='true', color='red')
                plt.title(f'{feat}, instance {instance}')
                plt.legend()
                plt.savefig(sanitize(f'{instance}_{feat}.png'), dpi=600)
        
        summary[instance]['params'] = OrderedDict()
        for period in range(n_periods):
            summary[instance]['params'][period] = OrderedDict()
            for feat_it, feat in enumerate(res[instance][0]['true']['params'][period].keys()):
                if (res[instance][0]['true']['params'][period][feat] == 0) and (feat not in param_list):
                    continue
                summary[instance]['params'][period][feat] = OrderedDict()
                truth = res[instance][0]['true']['params'][period][feat]
                summary[instance]['params'][period][feat]['true'] = truth
                for method in res[instance][0]:
                    if method == 'true':
                        continue
                    point = np.array([helper(res[instance][t][method]['params'][period]['point_estimate'], feat)
                                    for t in range(n_samples)])
                    std = np.array([helper(res[instance][t][method]['params'][period]['stderr'], feat)
                                    for t in range(n_samples)])
                    summary[instance]['params'][period][feat][sanitize(method)] = metrics(truth, point, std)
                    if plot:
                        plt.hist(point, label=method, alpha=.4)
                if plot:
                    plt.axvline(x = truth, label='true', color='red')
                    plt.title(f'period {period}, feat {feat}, instance {instance}')
                    plt.legend()
                    plt.savefig(sanitize(f'{instance}_{period}_{feat}.png'), dpi=600)

    return summary

def summarize_instances(summary):
    n_instances = len(summary)
    allsummary = OrderedDict()
    for feat in summary[0]:
        if feat == 'params':
            continue
        allsummary[feat] = OrderedDict()
        for method in summary[0][feat]:
            if method =='true':
                continue
            allsummary[feat][method] = OrderedDict()
            for attr in summary[0][feat][method]:
                avg = np.nanmean([summary[t][feat][method][attr][0]
                            for t in range(n_instances)])
                std = np.sqrt(np.nanvar([summary[t][feat][method][attr][0]
                                    for t in range(n_instances)]) +
                            np.nanmean([summary[t][feat][method][attr][1]**2
                                    for t in range(n_instances)]))
                stderr = std / np.sqrt(n_instances)
                if attr == 'mse':
                    attr, avg, stderr = 'rmse', np.sqrt(avg), np.sqrt(stderr)
                if method == 'reg' and (attr.startswith('coverage') or attr == 'stderr'):
                    allsummary[feat][method][attr] = "NA"    
                else:
                    allsummary[feat][method][attr] = r"{:.3f} $\pm$ {:.3f}".format(avg, 1.96 * stderr)

    allsummary['params'] = OrderedDict()
    for period in summary[0]['params']:
        allsummary['params'][period] = OrderedDict()
        for feat in summary[0]['params'][period]:
            allsummary['params'][period][feat] = OrderedDict()
            for method in summary[0]['params'][period][feat]:
                if method =='true':
                    continue
                allsummary['params'][period][feat][method] = OrderedDict()
                for attr in summary[0]['params'][period][feat][method]:
                    avg = np.nanmean([summary[t]['params'][period][feat][method][attr][0]
                                for t in range(n_instances)])
                    std = np.sqrt(np.nanvar([summary[t]['params'][period][feat][method][attr][0]
                                        for t in range(n_instances)]) +
                                  np.nanmean([summary[t]['params'][period][feat][method][attr][1]**2
                                        for t in range(n_instances)]))
                    stderr = std / np.sqrt(n_instances)
                    if attr == 'mse':
                        attr, avg, stderr = 'rmse', np.sqrt(avg), np.sqrt(stderr)
                    if method == 'reg' and (attr.startswith('coverage') or attr == 'stderr'):
                        allsummary['params'][period][feat][method][attr] = "NA"    
                    else:
                        allsummary['params'][period][feat][method][attr] = r"{:.3f} $\pm$ {:.3f}".format(avg, 1.96 * stderr)

    return allsummary

def tables(allsummary, latex=False):
    tables = OrderedDict()
    for feat in allsummary:
        if feat == 'params':
            continue
        if latex:
            tables[feat] = pd.DataFrame(allsummary[feat]).loc[[k for k in next(iter(allsummary[feat].values()))]].to_latex()
        else:
            tables[feat] = pd.DataFrame(allsummary[feat]).loc[[k for k in next(iter(allsummary[feat].values()))]]

    for period, params in allsummary['params'].items():
        tables[period] = OrderedDict()
        for feat, metrics in params.items():
            if latex:
                tables[period][feat] = pd.DataFrame(metrics).loc[[k for k in next(iter(allsummary['params'][period][feat].values()))]].to_latex()
            else:
                tables[period][feat] = pd.DataFrame(metrics).loc[[k for k in next(iter(allsummary['params'][period][feat].values()))]]
    return tables

def main(res_folder, exp_name, n_unit_list, n_x_list, n_hetero_vars, param_list, plot=False, latex=False):
    n_instances = 1
    n_samples = 20
    n_periods = 2

    all_tables = OrderedDict()
    for n_x in n_x_list:
        all_tables[n_x] = OrderedDict()
        for n_units in n_unit_list:
            res = []
            for ins in range(10):
                res.append([])
                for it, sample in enumerate(np.arange(0, 100, 20)):
                    file = f'n_ins_{n_instances}_start_{ins}_n_sam_{n_samples}_start_{sample}_n_hetero_vars_{n_hetero_vars}_n_units_{n_units}_n_x_{n_x}.jbl'
                    folder = f'dyndml_{n_x}_{n_hetero_vars}_{n_units}_{ins}_{sample}'
                    full = os.path.join(res_folder, exp_name, folder, file)
                    res[ins] += joblib.load(full)[0]

            all_tables[n_x][n_units] = tables(summarize_instances(summarize_samples(res, param_list, plot=plot)), latex=latex)
    return all_tables

def print_tables(res_folder, exp_name, n_unit_list, n_x_list, param_list, n_hetero_vars):

    tables = main(res_folder, exp_name, n_unit_list, n_x_list, n_hetero_vars, param_list)

    feat = 'pival'
    table_dict = OrderedDict()
    for n_x in n_x_list:
        alldf = None
        for n_units in n_unit_list:
            df = tables[n_x][n_units][feat].copy()
            df = df.drop(['stderr', 'mae', 'mape'])
            df[r'$n_{\text{units}}$'] = n_units
            df = df.reset_index().rename({'index': 'metric'}, axis=1).set_index([r'$n_{\text{units}}$', 'metric'])
            alldf = df if alldf is None else pd.concat([alldf, df])
        table_dict[r"$n_x={}$".format(n_x)] = alldf
    table = pd.concat(table_dict, axis=1)

    print("Table for policy value")
    print(table.to_latex(bold_rows=True, multirow=True, escape=False))

    alltables = OrderedDict()
    for feat in param_list:
        table_dict = OrderedDict()
        for period in [0, 1]:
            alldf = None
            for n_x in n_x_list:
                for n_units in n_unit_list:
                    df = tables[n_x][n_units][period][feat].copy()
                    df = df.drop(['stderr', 'mae', 'mape'])
                    df['$n_x$'] = n_x
                    df[r'$n_{\text{units}}$'] = n_units
                    df = df.reset_index().rename({'index': 'metric'}, axis=1).set_index(['$n_x$', r'$n_{\text{units}}$', 'metric'])
                    alldf = df if alldf is None else pd.concat([alldf, df])
            table_dict[f'period={period}'] = alldf
        alltables[feat] = pd.concat(table_dict, axis=1)
    
    for feat in param_list:
        print(f"Table for parameter {feat}")
        print(alltables[feat].to_latex(bold_rows=True, multirow=True, escape=False))
