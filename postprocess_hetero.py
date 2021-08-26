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
    metrics['bias'] = mean_stderr(point - truth)
    metrics['mse'] = mean_stderr((point - truth)**2)
    metrics['mae'] = mean_stderr(np.abs(point - truth))
    metrics['mape'] = mean_stderr(np.abs(point - truth) / np.abs(truth))
    return metrics

def summarize_samples(res, plot=False):
    n_instances = len(res)
    n_samples = len(res[0])
    n_periods = len(res[0][0]['maxcv'])
    summary = OrderedDict()
    for instance in range(n_instances):
        summary[instance] = OrderedDict()
        for period in range(n_periods):
            summary[instance][period] = OrderedDict()
            for method in res[instance][0]:
                summary[instance][period][sanitize(method)] = OrderedDict() 
                rmse = np.array([res[instance][t][method][period]['rmse']
                                for t in range(n_samples)])
                mae = np.array([res[instance][t][method][period]['mae']
                                for t in range(n_samples)])
                summary[instance][period][sanitize(method)]['rmse'] = mean_stderr(rmse)
                # summary[instance][period][sanitize(method)]['mae'] = mean_stderr(mae)
    return summary

def summarize_instances(summary):
    n_instances = len(summary)
    allsummary = OrderedDict()
    for period in summary[0]:
        allsummary[period] = OrderedDict()
        for method in summary[0][period]:
            allsummary[period][method] = OrderedDict()
            for attr in summary[0][period][method]:
                avg = np.nanmean([summary[t][period][method][attr][0]
                            for t in range(n_instances)])
                std = np.sqrt(np.nanvar([summary[t][period][method][attr][0]
                                    for t in range(n_instances)]) +
                                np.nanmean([summary[t][period][method][attr][1]**2
                                    for t in range(n_instances)]))
                stderr = std / np.sqrt(n_instances)
                allsummary[period][method][attr] = (avg, 1.96 * stderr)
        best = {}
        for attr in allsummary[period][next(iter(allsummary[period]))]:
            best[attr] = np.min([allsummary[period][t][attr][0] for t in allsummary[period]])
        for method in allsummary[period]:
            for attr in allsummary[period][method]:
                if allsummary[period][method][attr][0] <= best[attr]:
                    allsummary[period][method][attr] = r'\textbf{' + r"{:.3f} $\pm$ {:.3f}".format(*allsummary[period][method][attr]) + r'}'
                else:
                    allsummary[period][method][attr] = r"{:.3f} $\pm$ {:.3f}".format(*allsummary[period][method][attr])
    return allsummary

def tables(allsummary, latex=False):
    tables = OrderedDict()
    for period, params in allsummary.items():
        tables[period] = pd.DataFrame(params).transpose()
    return tables

def main(res_folder, exp_name, n_unit_list, n_x_list, n_hetero_vars, nonlin_fn_list, plot=False, latex=False):
    n_instances = 1
    n_samples = 20
    n_periods = 2

    all_tables = OrderedDict()
    for nonlin_fn in nonlin_fn_list:
        all_tables[nonlin_fn] = OrderedDict()
        for n_x in n_x_list:
            all_tables[nonlin_fn][n_x] = OrderedDict()
            for n_units in n_unit_list:
                res = []
                for ins in range(10):
                    res.append([])
                    for it, sample in enumerate(np.arange(0, 100, 20)):
                        file = f'nonlin_fn_{nonlin_fn}_n_ins_{n_instances}_start_{ins}_n_sam_{n_samples}_start_{sample}_n_hetero_vars_{n_hetero_vars}_n_units_{n_units}_n_x_{n_x}.jbl'
                        folder = f'dyndml_{nonlin_fn}_{n_x}_{n_hetero_vars}_{n_units}_{ins}_{sample}'
                        full = os.path.join(res_folder, exp_name, folder, file)
                        res[ins] += joblib.load(full)[0]

                all_tables[nonlin_fn][n_x][n_units] = tables(summarize_instances(summarize_samples(res, plot=plot)), latex=latex)
    return all_tables

def print_tables(res_folder, exp_name, n_unit_list, n_x_list, nonlin_fn_list, n_hetero_vars):

    tables = main(res_folder, exp_name, n_unit_list, n_x_list, n_hetero_vars, nonlin_fn_list)

    alltables = OrderedDict()
    for nonlin_fn in nonlin_fn_list:
        table_dict = OrderedDict()
        for period in [0, 1]:
            alldf = None
            for n_x in n_x_list:
                for n_units in n_unit_list:
                    df = tables[nonlin_fn][n_x][n_units][period].copy()
                    df[r'$n_{\text{units}}$'] = n_units
                    df = df.reset_index().rename({'index': 'method'}, axis=1).set_index([r'$n_{\text{units}}$', 'method'])
                    alldf = df if alldf is None else pd.concat([alldf, df])
            table_dict[f'period={period}'] = alldf
        alltables[f'nonlinearity={nonlin_fn}'] = pd.concat(table_dict, axis=1)
    
    print(pd.concat(alltables, axis=1).to_latex(bold_rows=True, multirow=True, escape=False))

    for nonlin_fn in nonlin_fn_list:
        print(f"Table for function {nonlin_fn}")
        print(alltables[f'nonlinearity={nonlin_fn}'].to_latex(bold_rows=True, multirow=True, escape=False))
