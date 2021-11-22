import argparse
import os
import numpy as np
import pandas as pd
from utils import str2bool
from copy import deepcopy
import itertools
import time

def p_value(betas):
    # calculate p-value for two-sided difference from 0 test with a bootstrapped distribution of statistics, betas
    abs_mean_beta = np.abs(np.mean(betas))
    centered_betas = betas - np.mean(betas)
    outside_prop = np.mean(centered_betas < -abs_mean_beta) + np.mean(centered_betas > abs_mean_beta)  
    return outside_prop

def element_is_primitive(x):
    if type(x[0]) is np.ndarray:
        _x = x[0].item()
    elif type(x) is np.ndarray and (x.shape == (1,) or x.shape == (1)):
        _x = x.item()
    elif type(x) is np.ndarray:
        return False
    else:
        _x = x
    return type(_x) in [int, np.int, float, np.float, bool, np.bool]

def unroll_stats(x, z_index=False):
    if not z_index:
        if element_is_primitive(x[0]):
            return x
        elif type(x) is np.ndarray:
            return x.reshape(-1)
    else:
        return_list = []
        for many_accs in x.reshape(-1):
            if not is_nan(many_accs):
                for acc in many_accs.split():
                    k,v = acc.split('-')
                    return_list.append(float(v))
        return np.array(return_list)

def oth_accs_to_flat_tuples(x):
    return_list = []
    for acc in x.split():
        k,v = acc.split('-')
        return_list.append((int(k), int(v)))
    return return_list

def is_nan(elem):
    if type(elem) in [int, str]:
        return False
    elif type(elem) is np.ndarray:
        return is_nan(elem.item())
    else:
        return np.isnan(elem)

def bootstrap_sparse_tensor(args, df, bootstrap_row_idx, bootstrap_col_idx, resample_z=True, all_z_idx_unique=False):
    '''
    bootstrap a sparse tensor 
    - the elements are a sparse representation of a third "z" dimension
    - the elements can be sparse in x/y dim
    - format: elem is list [(id, statistic),...] where id is the 'z' id
    RETURNS the z_idx used, which depend on the row_idx and col_idx used at each step
    '''
    means = []
    bootstrap_z_idx = []
    if all_z_idx_unique:
        z_id_counter = 0
    for i in range(1000): #range(args.num_samples):
        row_idx = bootstrap_row_idx[i]
        col_idx = bootstrap_col_idx[i]
        # subset to rows
        x_sample = df[row_idx,:]
        x_sample = df[:, col_idx]
        # flatten the data and make the dictionary of data idx to stats
        flattened_list = []
        z_idx_to_samples = {}
        for row_num, row in enumerate(x_sample):
            for elem in row:
                if not is_nan(elem):
                    for id_stat in elem.split():
                        if not all_z_idx_unique:
                            data_id, stat = id_stat.split('-')
                            data_id, stat = int(data_id), float(stat)
                        else:
                            data_id, stat = z_id_counter, int(id_stat)
                            z_id_counter += 1
                        if data_id not in z_idx_to_samples:
                            z_idx_to_samples[data_id] = [stat]
                        else:
                            z_idx_to_samples[data_id].append(stat)
        z_idx_in_sample_rows = list(z_idx_to_samples.keys())
        if resample_z:
            z_idx = np.random.choice(np.array(z_idx_in_sample_rows), size=len(z_idx_in_sample_rows), replace=True)
        else:
            z_idx = z_idx_in_sample_rows
        x_sample = [stat for idx in z_idx for stat in z_idx_to_samples[idx]]
        mean = np.nanmean(x_sample, dtype='float32')
        means.append(mean)
        bootstrap_z_idx.append(z_idx)
    lb, ub = np.quantile(means, (.025, .975))
    CI = (ub - lb) / 2
    ovr_mean = np.mean(means)
    result_str = f"{100*ovr_mean:2.2f} ({100*CI:1.2f}; n={len(row_idx)}; s={len(col_idx)})"
    return result_str, means, bootstrap_z_idx

def bootstrap_grid(args, df, bootstrap_col_idx=None, bootstrap_row_idx=None):
    means = []
    for i in range(args.num_samples):
        if bootstrap_row_idx is None:
            row_idx = [np.random.choice(np.arange(n_rows), size=n_rows, replace=True) for i in range(args.num_samples)]
        else:
            row_idx = bootstrap_row_idx[i]
        if bootstrap_col_idx is None:
            col_idx = [np.random.choice(np.arange(n_cols), size=n_cols, replace=True) for i in range(args.num_samples)]
        else:
            col_idx = bootstrap_col_idx[i]
        x_sample = df[row_idx, :]
        x_sample = x_sample[:, col_idx]
        mean = np.nanmean(x_sample, dtype='float32')
        means.append(mean)
    lb, ub = np.quantile(means, (.025, .975))
    CI = (ub - lb) / 2
    ovr_mean = np.mean(means)
    result_str = f"{100*ovr_mean:2.2f} ({100*CI:1.2f}; n={len(row_idx)}; s={len(col_idx)})"
    return result_str, means

def bootstrap_grid_diff(args, df1, df2, bootstrap_col_idx=None, bootstrap_row_idx=None):
    means = []
    means2 = []
    n_rows, n_cols = df1.shape
    for i in range(args.num_samples):
        if bootstrap_row_idx is None:
            row_idx = [np.random.choice(np.arange(n_rows), size=n_rows, replace=True) for i in range(args.num_samples)]
        else:
            row_idx = bootstrap_row_idx[i]
        if col_idx is None:
            bootstrap_col_idx = [np.random.choice(np.arange(n_cols), size=n_cols, replace=True) for i in range(args.num_samples)]
        else:
            col_idx = bootstrap_col_idx[i]
        x_sample = df1[row_idx, :]
        x_sample = x_sample[:, col_idx]
        mean = np.nanmean(x_sample, dtype='float32')
        means.append(mean)
        x_sample = df1[row_idx, :]
        x_sample = x_sample[:, col_idx]
        mean = np.nanmean(x_sample, dtype='float32')
        means2.append(mean)
    means = np.array(means) - np.array(means2)
    lb, ub = np.quantile(means, (.025, .975))
    CI = (ub - lb) / 2
    ovr_mean = np.mean(means)
    result_str = f"{100*ovr_mean:2.2f} ({100*CI:1.2f}; n={n_rows}; s={n_cols})"
    return result_str, means, means2

def bootstrap_metrics(args, df, metrics, stats_dict={}, bootstrap_idx_dict=None):
    '''
    get bootstrap estimated CIs for the named statistics in metrics
    when calculating before_cons and after_cons, e.g., will use the same bootstrap idx for both since the data is paired
    - this is done via the bootstrap_idx_dict

    '''
    results = {}
    if bootstrap_idx_dict is None: 
        bootstrap_idx_dict = {}
    non_chg_metrics = [metric for metric in metrics if 'chg' not in metric]
    z_idx_metrics = ['oth_acc', 'oth_ret', 'oth_par_eq_mean', 'oth_odp_acc'] # (sparse representation of oth_id and acc-stat for after update accuracies. the z dim is what other points were randomly subsampled)
    for metric in non_chg_metrics:
        base_metric = metric.replace('before_','').replace('after_','')
        # if main in metric name, compute before_metric without storing the boot idx in the bootstrap_idx_dict. later, in after/before naming scheme, the before metric adopts row_idx from after metric for comparability. (only on updated points by default)
        if metric == 'main_acc': 
            metric = 'before_acc'
        if metric == 'main_dep_acc':
            metric = 'before_dep_acc'
        if metric == 'main_cons':
            metric = 'before_cons'
        if metric == 'main_contrapositive':
            metric = 'before_contrapositive'
        if metric.replace('incorrect_','').replace('correct_','') not in df:
            continue
        # subset the data if correct/incorrect in the metric
        if 'incorrect' in metric:
            where_incorrect = np.argwhere(1-df['before_acc'].to_numpy()).reshape(-1)
            use_df = df.iloc[where_incorrect,:]
        elif 'correct' in metric:
            where_correct = np.argwhere(df['before_acc'].to_numpy()).reshape(-1)
            use_df = df.iloc[where_correct,:]
        else:
            use_df = deepcopy(df)
        col_name = metric.replace('incorrect_','').replace('correct_','')
        # pivot the data
        use_df = use_df.drop_duplicates()
        x = use_df.pivot(index='id', columns='seed', values=col_name)
        x = x.to_numpy()
        n_rows, n_cols = x.shape
        eligible_cols = np.arange(n_cols)
        # case 1, all metrics except oth_acc
        if metric not in z_idx_metrics:
            rows_where_all_missing = np.argwhere([all([is_nan(elem) for elem in row]) for row in x]) 
            eligible_rows = np.setdiff1d(np.arange(n_rows), rows_where_all_missing.reshape(-1))
            sample_size = len(eligible_rows)
            metric_not_encountered = (base_metric not in bootstrap_idx_dict)
            if metric_not_encountered:
                if args.variance in ['seed', 'both']:
                    col_idx = [np.random.choice(eligible_cols, size=n_cols, replace=True) for i in range(args.num_samples)]
                else:
                    col_idx = [np.arange(n_cols) for i in range(args.num_samples)]
                if args.variance in ['sample', 'both']:
                    row_idx = [np.random.choice(eligible_rows, size=sample_size, replace=True) for i in range(args.num_samples)]
                else:
                    row_idx = [eligible_rows for i in range(arg.num_samples)]
                bootstrap_idx_dict[base_metric] = {
                    'col' : col_idx,
                    'row' : row_idx
                }
        # case 2, metric is oth_X (which is sparse representation of oth_id and the stat for after update statistics)
        if metric in z_idx_metrics:
            eligible_cols = np.arange(n_cols)
            nan_rep = np.array([is_nan(elem) for row in x for elem in row]).reshape(x.shape)
            eligible_rows = np.argwhere( (nan_rep).sum(-1) == 0 ).reshape(-1) # rows with at least one filled cell in them
            sample_size = len(eligible_rows) if not args.single_update_CI else 1
            if args.variance in ['seed', 'both']:
                col_idx = [np.random.choice(eligible_cols, size=n_cols, replace=True) for i in range(args.num_samples)]
            else:
                col_idx = [np.arange(n_cols) for i in range(args.num_samples)]
            if args.variance in ['sample', 'both']:
                ROW_SIZE = sample_size
                row_idx = [np.random.choice(eligible_rows, size=ROW_SIZE, replace=True) for i in range(args.num_samples)]
            else:
                row_idx = [eligible_rows for i in range(arg.num_samples)]
            bootstrap_idx_dict[base_metric] = {
                'col' : col_idx,
                'row' : row_idx
            }
        # continue here if there are no eligible rows, meaning variable is not represented in the data
        if sample_size == 0:            
            continue
        else:
            print(f" on metric: {metric}")
        # IF COMPUTING BEFORE_ACC, GET ROW_IDX FROM THE Z_IDX FROM oth_acc, if using oth_acc with bootstrap_z_idx
        if metric == 'before_acc' and 'oth_acc' in metrics:
            bootstrap_idx_dict['acc']['row'] = bootstrap_idx_dict['oth_acc']['z_idx'] 
        # if computing after acc, set each element to its mean
        if metric in ['after_acc', 'after_odp_acc', 'after_ret', 'after_odp_ret']:
            for row_num, row in enumerate(x):
                for col_num, elem in enumerate(row):
                    if not is_nan(elem):
                        x[row_num, col_num] = np.mean([float(val) for val in elem.split()])
        # PERFORM BOOTSTRAP HERE
        if metric in z_idx_metrics:
            result_str, means, bootstrap_z_idx = bootstrap_sparse_tensor(args, x, 
                bootstrap_col_idx=bootstrap_idx_dict[base_metric]['col'], 
                bootstrap_row_idx=bootstrap_idx_dict[base_metric]['row'],
            )
            # store z_idx
            bootstrap_idx_dict[metric]['z_idx'] = bootstrap_z_idx
        else:
            result_str, means = bootstrap_grid(args, x, 
                bootstrap_col_idx=bootstrap_idx_dict[base_metric]['col'], 
                bootstrap_row_idx=bootstrap_idx_dict[base_metric]['row'])
        # get ovr mean by seeing what were all the rows included in the bootstrap. 
        used_rows = np.array(list(set([item for row in bootstrap_idx_dict[base_metric]['row'] for item in row]))) # this takes a little while
        assert len(used_rows) > 0, "used_rows is len 0"
        unrolled_x = unroll_stats(x[used_rows], z_index = metric in z_idx_metrics) 
        ovr_mean = np.nanmean(unrolled_x, dtype='float32')
        error_thresh = .01
        if args.num_samples >= 100:
            print(f"    result: {result_str} -- exact mean is: {100*ovr_mean:.2f}")
        if np.abs(ovr_mean - np.mean(means)) / ovr_mean > error_thresh and args.num_samples >= 100: # 100 is debugging size
            print(f"    Warning: bootstrap too small, estimation error of greater than {100*error_thresh}% detected between bootstrap mean and sample mean")
        results[metric] = result_str
        stats_dict[f"{metric}_means"] = np.array(means)
    return results, bootstrap_idx_dict

def bootstrap_experiment(args, data_stats, metrics, condition_vars=None, condition_dict=None, bootstrap_idx_dict=None):
    # filter for some experiment conditons
    if 'r_ablation' in args.experiment:
        where_r_10 = np.argwhere(data_stats.r_test.to_numpy()==10).reshape(-1)
        data_stats = data_stats.iloc[where_r_10,:]
        data_stats.upd_suc = np.array([str2bool(elem) if type(elem) is str else 1*elem for elem in data_stats.upd_suc.to_numpy()])
        data_stats = data_stats.drop_duplicates(['dataset','seed','id', 'r_train', 'r_test'])
    if args.dataset is not None:
        where_data = np.argwhere(data_stats.dataset.to_numpy()==args.dataset).reshape(-1)
        data_stats = data_stats.iloc[where_data,:]
    # data_stats.upd_suc = data_stats.upd_suc.to_numpy().astype(np.bool).astype(np.float)
    # first get all experiment configs
    condition_lists = []
    if condition_dict is not None:
        for k,v in condition_dict.items():
            where_condition_holds = np.argwhere(data_stats[k].to_numpy()==v).reshape(-1)
            data_stats = data_stats.iloc[where_condition_holds,:]
    if condition_vars is None:
        condition_vars = condition_dict.keys()
    for var in condition_vars:
        conditions = set(data_stats[var])
        condition_lists.append(
            [(var, value) for value in conditions]
        )
    all_configs = sorted(list(itertools.product(*condition_lists)))
    all_results = {}
    assert len(all_configs) > 0, "missing saved data"
    for config in all_configs:
        # RESET STATS DICT and BOOTSTRAP dicts
        stats_dict = {} 
        bootstrap_idx_dict = {}
        exp_id = ' | '.join([f"{var:10s}: {str(value):13s}" for var, value in config])
        subset_df = deepcopy(data_stats)
        for var, value in config:
            subset_df = subset_df.loc[subset_df[var] == value]
        print(f"Starting bootstrap for experiment: {config}...")

        # make contrapositive column for LeapOfThought. this is 1 if contrapositive is correct, 0 if not, and nan if it doesnt apply
        if ("dataset", "LeapOfThought") in config:
            n = subset_df.shape[0]
            before_contrapositive = [(subset_df.before_acc.iloc[i]==0) if subset_df.before_dep_acc.iloc[i]==0 else np.nan for i in range(n)] # doesn't apply when B is true in A->B. only if not B.
            after_contrapositive = [(subset_df.upd_acc.iloc[i]==0) if subset_df.dep_acc.iloc[i]==0 else np.nan for i in range(n)] # doesn't apply when B is true in A->B. only if not B.
            subset_df['before_contrapositive'] = before_contrapositive
            subset_df['after_contrapositive'] = after_contrapositive
        
        # bootsTRAP HERE
        results, bootstrap_idx_dict = bootstrap_metrics(args, subset_df, metrics, stats_dict, bootstrap_idx_dict=bootstrap_idx_dict)
        for metric, result_str in results.items():
            exp_id_metric = exp_id + f' | {metric:12s}'
            all_results[exp_id_metric] = result_str
            if len(all_configs) == 1:
                all_results[metric] = result_str # used in hypothesis_testing
        
        # make metrics that require comparing two columns
        for metric in [metric for metric in metrics if 'chg' in metric]:

            name = metric.split('_')[0]
            # naming special cases
            if metric == 'dep_chg':
                before_name = f'incorrect_before_dep_acc_means'
                after_name = f'dep_acc_means'
            elif metric == 'acc_chg':
                before_name = f'before_acc_means'
                after_name = f'after_acc_means' if 'after_acc' in metrics else 'oth_acc_means'
            elif metric == 'ind_acc_chg':
                before_name = f'before_ind_acc_means'
                after_name = f'ind_acc_means'
            elif metric == 'cons_chg':
                before_name = f'before_cons_means'
                after_name = f'cons_means'
            elif metric == 'odp_chg':
                before_name = f'before_dep_acc_means'
                after_name = f'after_odp_acc_means'
            elif metric == 'combined_acc_chg':
                after_name = 'acc_chg_means'
                before_name = 'odp_chg_means'
            elif metric == 'combined_ret_chg':
                after_name = 'after_ret_means'
                before_name = 'after_odp_ret_means'
            else:
                before_name = f"before_{metric}_acc_means"
                after_name = f"after_{metric}_acc_means"
            operation = 'diff' if 'combined' not in metric else 'avg'
            # get arrays of means for both stats_dicts
            if before_name in stats_dict:
                print(f" on metric: {metric}")
                before_means = stats_dict[before_name]
                after_means = stats_dict[after_name]
                means = after_means - before_means if operation=='diff' else np.mean(np.concatenate((after_means.reshape(-1,1), before_means.reshape(-1,1)), axis=1), axis=1)
                lb, ub = np.quantile(means, (.025, .975))
                CI = (ub - lb) / 2
                ovr_mean = np.mean(means)
                result_str = f"{100*ovr_mean:2.2f} ({100*CI:1.2f}; p={p_value(means):.4f})"
                exp_id_metric = exp_id + f' | {metric:12s}'
                all_results[exp_id_metric] = result_str
                print(f"    result: {result_str} -- exact mean is: {100*ovr_mean:.2f}")
                stats_dict[f"{metric}_means"] = means # add means back in for comparing between two _chg metrics
                if len(all_configs) == 1:
                    all_results[metric] = result_str # used in hypothesis_testing
    return all_results, stats_dict, bootstrap_idx_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", '-e', default='', type=str, help='get statistics for this experiment') 
    parser.add_argument("--variance", '-v', default='both', choices=['seed', 'sample', 'both'], help='account for variance of this source of randomness') 
    parser.add_argument("--num_random_other", type=int, help='')
    parser.add_argument("--single_update_CI", type=str2bool, default=False, help='get CI on other statistics from updating a single point, rather than the CI on the dataset avg update') 
    parser.add_argument("--num_samples", '-n', type=int, default=10000, help='num bootstrap samples')
    parser.add_argument("--hypothesis_tests", '-t', type=str2bool, default=False, help='perform requested hypothesis tests') 
    parser.add_argument("--dataset", default=None, help='num bootstrap samples')
    args = parser.parse_args()

    results_dir = 'aggregated_results'        
    start = time.time()

    # define metrics to examine. "after_*" metrics MUST come before "before_" metrics for bootstrapping to work properly
    metrics = ['main_acc', 'main_dep_acc', 'main_cons', 'main_contrapositive', # get stats for whole dataset, before updating
            'upd_suc', 
            'par_eq_mean', 'before_par_eq_mean', # paraphrase update success
            'cons', 'before_cons', 'correct_before_cons', 'incorrect_before_cons', 'cons_chg',  # note before_cons will be for incorrectly predicted points
            'dep_acc', 'correct_before_dep_acc', 'incorrect_before_dep_acc', 'dep_chg', # entailed update success
            'after_contrapositive', 'before_contrapositive', 'contrapositive_chg',
            'after_ctp_acc', 'before_ctp_acc', 'ctp_chg', # contrapositive update success
            'after_acc',     'before_acc',     'acc_chg', # change in acc for other data            
            'after_odp_acc', 'odp_chg', # change in acc for other entailed data (entailed by other points)
            'combined_after_ret',
            'combined_acc_chg', # for leapofthought, avg the acc_chg and odp_chg            
            'ind_acc', 'before_ind_acc', 'ind_acc_chg', # change in acc for ind data
            'ind_ret', # retain prediction rate for neutral and other data
            'after_ret', 'after_odp_ret',
            'combined_ret_chg', # for leapofthought, avg the after_ret and odp_ret
        ]

    # confidence intervals for individual conditions
    if not args.hypothesis_tests:
        print(f"Starting bootstraps for experiment: {args.experiment}")
  
        # read files
        summary_path = os.path.join(results_dir, f'{args.experiment}_summary_stats.csv')
        data_stats_path = os.path.join(results_dir, f'{args.experiment}_data_stats.csv')
        summary_stats = pd.read_csv(summary_path)
        data_stats = pd.read_csv(data_stats_path)

        # define the variables that give each experiment condition
        condition_vars = ['dataset']
        if args.experiment == 'tune_base_optimizers':
            condition_vars.extend(['optimizer', 'lr', 'k_test'])
        if args.experiment == 'learned_opt_r_ablation':
            condition_vars.extend(['r_train', 'r_test'])
        if args.experiment == 'learned_opt_r_main':
            condition_vars.extend(['r_train'])
        if args.experiment == 'base_optimizers_r_ablation':
            condition_vars.extend(['r_train', 'r_test'])
        if args.experiment == 'base_optimizers_r_main':
            condition_vars.extend(['r_train', 'r_test'])
        if args.experiment == 'learned_opt_objective_ablation':
            condition_vars.extend(['obj'])
        if args.experiment == 'learned_opt_eval_ablation':
            condition_vars.extend(['eval_beam_search_alt_labels'])
        if args.experiment == 'learned_opt_label_ablation':
            condition_vars.extend(['alt_label'])

        results, _, bootstrap_idx_dict = bootstrap_experiment(args,
                                data_stats=data_stats, 
                                metrics=metrics,
                                condition_vars=condition_vars)
        print(f"Results for experiment {args.experiment}")
        with open(os.path.join('outputs', f'bootstrap_{args.experiment}_b{args.num_samples}.csv'), 'w') as f:
            f.write("config, estimate\n")
            for k,v in results.items():
                print(f"{k:50s} : {v}")
                f.write(f"{k}, {v}\n")

    # hypothesis testing
    else:
        hypothesis_tests = []
        if args.experiment == 'learned_opt_eval_ablation':
            hypothesis_tests.append(
                (("learned_opt_eval_ablation", {"dataset" : "ZSRE", "eval_beam_search_alt_labels" : True}),
                 ("learned_opt_eval_ablation", {"dataset" : "ZSRE", "eval_beam_search_alt_labels" : False}))
                )
        elif args.experiment == 'learned_opt_main':
            hypothesis_tests.append(
                (("learned_opt_main", {"dataset" : "Wikidata5m"}),
                 ("base_optimizers", {"dataset" : "Wikidata5m"})),
                )
            hypothesis_tests.append(
                (("learned_opt_main", {"dataset" : "ZSRE"}),
                 ("base_optimizers", {"dataset" : "ZSRE"})),
                )
            hypothesis_tests.append(
                (("learned_opt_main", {"dataset" : "LeapOfThought"}),
                 ("base_optimizers", {"dataset" : "LeapOfThought"})),
                )
            hypothesis_tests.append(
                (("learned_opt_main", {"dataset" : "FEVER"}),
                 ("base_optimizers", {"dataset" : "FEVER"})),
                )
        elif args.experiment == 'learned_opt_r_main':
            for dataset in ['ZSRE', 'Wikidata5m', 'FEVER', 'LeapOfThought']:
                hypothesis_tests.append(
                    (("learned_opt_r_main", {"dataset" : dataset, "r_train" : 10}),
                     ("base_optimizers_r_main", {"dataset" : dataset})),
                    )
        elif args.experiment == 'learned_opt_label_ablation':
            hypothesis_tests.append(
                (("learned_opt_label_ablation", {"dataset" : "ZSRE", "alt_label" : 'random'}),
                 ("learned_opt_label_ablation", {"dataset" : "ZSRE", "alt_label" : 'beam'}))
                )
        elif args.experiment == 'learned_opt_objective_ablation':
            hypothesis_tests.append(
                (("learned_opt_objective_ablation", {"dataset" : "LeapOfThought", "obj" : 'ce-kl'}),
                 ("learned_opt_objective_ablation", {"dataset" : "LeapOfThought", "obj" : 'ce-kl-dep'}))
                )
            hypothesis_tests.append(
                (("learned_opt_objective_ablation", {"dataset" : "ZSRE", "obj" : 'ce-kl'}),
                 ("learned_opt_objective_ablation", {"dataset" : "ZSRE", "obj" : 'ce-kl-par'}))
                )
            hypothesis_tests.append(
                (("learned_opt_objective_ablation", {"dataset" : "Wikidata5m", "obj" : 'ce-kl'}),
                 ("learned_opt_objective_ablation", {"dataset" : "Wikidata5m", "obj" : 'ce-kl-par'}))
                )
            hypothesis_tests.append(
                (("learned_opt_objective_ablation", {"dataset" : "Wikidata5m", "obj" : 'ce-kl-par'}),
                 ("learned_opt_objective_ablation", {"dataset" : "Wikidata5m", "obj" : 'ce-kl-ind-par'}))
                )
        elif args.experiment == 'learned_opt_label_ablation':
            hypothesis_tests.extend(['alt_label'])
        else:
            hypothesis_tests = [
                (("learned_opt_r_main", {"dataset" : "ZSRE", "r_train" : 10, "r_test" : 10}),
                 ("base_optimizers_r_ablation", {"dataset" : "ZSRE", "r_test" : 10}),  
                ),
                (("learned_opt_r_main", {"dataset" : "FEVER", "r_train" : 10, "r_test" : 10}),
                 ("base_optimizers_r_ablation", {"dataset" : "FEVER", "r_test" : 10}),  
                )
            ]        
        print("Beginning hypothesis testing...")
        for hypothesis_test in hypothesis_tests:
            condition_A = hypothesis_test[0]
            condition_B = hypothesis_test[1]
            print(f"  {condition_A} vs. {condition_B}")
            data_stats_path = os.path.join(results_dir, f'{condition_A[0]}_data_stats.csv')
            data_stats = pd.read_csv(data_stats_path)
            results_A, means_dict_A, bootstrap_idx_dict_A = bootstrap_experiment(args,
                            data_stats=data_stats, 
                            metrics=metrics,
                            condition_dict=condition_A[1])
            data_stats_path = os.path.join(results_dir, f'{condition_B[0]}_data_stats.csv')
            data_stats = pd.read_csv(data_stats_path)
            results_B, means_dict_B, _ = bootstrap_experiment(args,
                    data_stats=data_stats, 
                    metrics=metrics,
                    condition_dict=condition_B[1],
                    bootstrap_idx_dict=bootstrap_idx_dict_A)
            A_keys = list(means_dict_A.keys())
            B_keys = list(means_dict_B.keys())
            if not set(A_keys) == set(B_keys): print("metrics not the same between both experiments! may encounter error")
            for key in A_keys:
                metric = key.replace("_means", "")
                means = means_dict_A[key] - means_dict_B[key]
                lb, ub = np.quantile(means, (.025, .975))
                CI = (ub - lb) / 2
                ovr_mean = np.mean(means)
                result_str = f"{100*ovr_mean:2.2f} ({100*CI:1.2f}; p={p_value(means):.4f})"
                print(f"  metric: {metric}")
                print(f"    A_score: {results_A[metric]}")
                print(f"    B_score: {results_B[metric]}")
                print(f"    comparison: {result_str}")

    print(f"\n Runtime: {(time.time() - start) / 60:.2f} minutes")








