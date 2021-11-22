import submitit
import argparse
import os
import utils
from utils import str2bool
import pandas as pd

def job_function(job_command):
    os.system(job_command)

def accumulate_results_to_df(args, exp_names, df, is_update_epoch, split_name, prefix='data_stats'):
    for exp_name in exp_names:
        result_sheet_name = utils.get_result_sheet_name(args, exp_name, is_update_epoch, split_name)
        data_path = os.path.join('result_sheets/', f"{prefix}_{result_sheet_name}")
        try:
            job_finished = os.path.exists(data_path)
            new_data = pd.read_csv(data_path)
            # forgot to add these conditions to data frames
            if args.experiment == 'learned_opt_eval_ablation':
                new_data['eval_beam_search_alt_labels'] = args.eval_beam_search_alt_labels
            if args.experiment == 'learned_opt_label_ablation':
                new_data['alt_label'] = args.alt_label
            if args.experiment == 'learned_opt_sample_efficiency':
                new_data['num_train_points'] = args.num_train_points
            if job_finished:
                df = df.append(new_data)
            elif prefix == 'summary_stats':
                new_data = pd.read_csv(data_path)
                columns = new_data.columns
                for i in range(new_data.shape[-1]):
                    if 'train' in columns[i] or 'dev' in columns[i] or 'test' in columns[i]:
                        new_data.iloc[0,i] = "JOB NOT FINISHED"
                print(f"\nJOB NOT FINISHED: Incomplete result sheet: {prefix}_{result_sheet_name}\n")
                df = df.append(new_data)
        except:
            print(f"\nWARNING: Couldn't find result sheet: {prefix}_{result_sheet_name})\n")            
    return df

def run_across_seed_data_model(args, command, is_update_experiment=False, override_dataset=None, override_seeds=None, override_args=''):
    # runs a given experiment across datasets and seeds
    # returns this list of experiment_name's produced in main.py
    exp_names = []
    hparam_dicts = []
    if args.submit == 'slurm': 
        jobs_list = []
        commands_list = []
    _seeds = args.seeds if not override_seeds else override_seeds
    for seed in _seeds:
        _datasets = args.datasets if not override_dataset else [override_dataset]
        for data in _datasets:
            if args.num_train_epochs is not None:
                num_train_epochs = args.num_train_epochs
            else:
                epochs_key = 'num_train_epochs_learned_opt' if is_update_experiment else 'num_train_epochs'
                num_train_epochs = args.data_configs[data][epochs_key]
            eval_when_training_size = args.data_configs[data]['learned_opt_train_eval_size'] if is_update_experiment else -1
            learned_opt_objectives = args.data_configs[data]['learned_opt_default_objectives']
            job_params = args.job_params + \
                f" --num_train_epochs {num_train_epochs} "\
                f" --num_random_other {args.data_configs[data]['num_random_other']} "\
                f" --probing_style {args.data_configs[data]['probing_style']} "\
                f" --probe {args.data_configs[data]['probe']} "\
                f" --model {args.data_configs[data]['model']} "\
                f" --seed {seed} --dataset {data} "\
                f" --eval_n_points_when_training {eval_when_training_size} "\
                f" {learned_opt_objectives} "
            job_command = f" {command} {job_params} {override_args}"
            if args.submit=='slurm' and not args.collect_results:
                commands_list.append(job_command)
            elif args.submit=='slurm' and args.collect_results:
                print(f"\n\nCollecting results for experiment {args.experiment} | seed {seed} | data {data} | command:\n\t {job_command} \n\n")
            # launch os jobs here
            elif args.submit == 'os':
                print(f"\n\nStarting job | experiment {args.experiment} | seed {seed} | data {data} | command:\n\t {job_command} \n\n")
                job_function(job_command)
            exp_name = utils.get_experiment_name_from_command(job_command)
            exp_names.append(exp_name)
            hparam_dicts.append(utils.command_to_dict(job_command))
    # launch slurm jobs here
    if args.submit == 'slurm' and not args.collect_results:
        executor = submitit.AutoExecutor(folder="slurm_logs")
        executor.update_parameters(timeout_min=4000, 
                                    slurm_partition=args.server,
                                    gpus_per_node=1,
                                    tasks_per_node=1,
                                    cpus_per_task=10,
                                    nodes=1,
                                    slurm_constraint='volta32gb',
                                    slurm_array_parallelism=24)
        jobs_list = executor.map_array(job_function, commands_list)
        for job in jobs_list:
            print(f"Launched job {job.job_id}")
    return exp_names, hparam_dicts

def task_model(args):
    data_stats    = pd.DataFrame({'dataset' : [], 'seed' : [], 'id' : []})
    summary_stats = pd.DataFrame({'dataset' : [], 'seed' : []})
    command = f"python main.py --train_batch_size 32 --test_batch_size 64 --do_train true --write_preds_to_file true --beam_search_alt_labels true --beam_search_size 5 "
    exp_names, hparam_dicts = run_across_seed_data_model(args, command)
    if args.collect_results:
        for exp_name, hparam_dict in zip(exp_names, hparam_dicts):
            data_stats    = accumulate_results_to_df(args, [exp_name], data_stats,    is_update_epoch=False, split_name='test', prefix='data_stats')
            summary_stats = accumulate_results_to_df(args, [exp_name], summary_stats, is_update_epoch=False, split_name='test', prefix='summary_stats')
        for df, prefix in zip([data_stats, summary_stats], 
                              ['data_stats', 'summary_stats']):
            save_path = os.path.join('aggregated_results', args.experiment + '_' + prefix + '.csv')
            df.to_csv(save_path, index=False)

def write_graph(args):
    command = f"python main.py --test_batch_size 128 --do_train true --write_graph_to_file true "
    override = "--do_train false"
    exp_names, hparam_dicts = run_across_seed_data_model(args, command, override_args=override)

def write_LeapOfThought_preds(args):
    # write LeapOfThought preds to file for the implicit_rule
    data_stats    = pd.DataFrame({'dataset' : [], 'seed' : [], 'id' : []})
    summary_stats = pd.DataFrame({'dataset' : [], 'seed' : []})
    for main_input in ['implicit_rule', 'hypothesis']:
        command = f"python main.py --train_batch_size 32 --test_batch_size 64 --do_train false --leapofthought_main {main_input} --write_preds_to_file true "
        exp_names, hparam_dicts = run_across_seed_data_model(args, command)
        if args.collect_results:
            for exp_name, hparam_dict in zip(exp_names, hparam_dicts):
                data_stats    = accumulate_results_to_df(args, [exp_name], data_stats,    is_update_epoch=False, split_name='test', prefix='data_stats')
                summary_stats = accumulate_results_to_df(args, [exp_name], summary_stats, is_update_epoch=False, split_name='test', prefix='summary_stats')
            for df, prefix in zip([data_stats, summary_stats], 
                                  ['data_stats', 'summary_stats']):
                save_path = os.path.join('aggregated_results', args.experiment + '_' + prefix + '.csv')
                df.to_csv(save_path, index=False)

def write_alt_beam_preds(args):
    # write LeapOfThought preds to file for the implicit_rule
    data_stats    = pd.DataFrame({'dataset' : [], 'seed' : [], 'id' : []})
    summary_stats = pd.DataFrame({'dataset' : [], 'seed' : []})
    command = f"python main.py --train_batch_size 32 --test_batch_size 64 --do_train false --beam_search_alt_labels true --beam_search_size 5 --write_preds_to_file true "
    exp_names, hparam_dicts = run_across_seed_data_model(args, command, is_update_experiment=True)
    if args.collect_results:
        for exp_name, hparam_dict in zip(exp_names, hparam_dicts):
            data_stats    = accumulate_results_to_df(args, [exp_name], data_stats,    is_update_epoch=False, split_name='test', prefix='data_stats')
            summary_stats = accumulate_results_to_df(args, [exp_name], summary_stats, is_update_epoch=False, split_name='test', prefix='summary_stats')
        for df, prefix in zip([data_stats, summary_stats], 
                              ['data_stats', 'summary_stats']):
            save_path = os.path.join('aggregated_results', args.experiment + '_' + prefix + '.csv')
            df.to_csv(save_path, index=False)

def tune_base_optimizers(args):
    command = f"python main.py --test_batch_size 32 --update_beliefs true --num_eval_points 5000 --update_parameters all "\
              f" --leapofthought_main implicit_rule "
    max_num_updates = [5, 10, 100]
    args.job_params = args.job_params.replace("--do_train True", "--do_train false") # never train here
    data_stats = pd.DataFrame({'dataset' : [], 'seed' : [], 'id' : []})
    summary_stats = pd.DataFrame({'dataset' : [], 'seed' : []})
    for optimizer in ['adamw', 'sgd', 'rmsprop']:
        for max_num_update in max_num_updates:
            lrs = args.optimizer_to_sweep_lrs[optimizer]
            for lr in lrs:
                _command = command + f'--optimizer {optimizer} --update_steps {max_num_update} --lr {lr}'
                exp_names, hparam_dicts = run_across_seed_data_model(args, _command)
                if args.collect_results:
                    for exp_name, hparam_dict in zip(exp_names, hparam_dicts):
                        args.num_random_other = hparam_dict['num_random_other']
                        args.update_steps = max_num_update
                        args.num_successive_updates = 1
                        data_stats    = accumulate_results_to_df(args, [exp_name], data_stats,    is_update_epoch=True, split_name='dev', prefix='data_stats')
                        summary_stats = accumulate_results_to_df(args, [exp_name], summary_stats, is_update_epoch=True, split_name='dev', prefix='summary_stats')
                    for df, prefix in zip([data_stats, summary_stats], 
                                          ['data_stats', 'summary_stats']):
                        save_path = os.path.join('aggregated_results', args.experiment + '_' + prefix + '.csv')
                        df.to_csv(save_path, index=False)

def base_optimizers(args):
    command = f"python main.py --test_batch_size 32 --update_beliefs true --update_parameters all "\
              f" --leapofthought_main implicit_rule "
    args.job_params = args.job_params.replace("--do_train True", "--do_train false") # never train here
    data_stats    = pd.DataFrame({'dataset' : [], 'seed' : [], 'id' : []})
    summary_stats = pd.DataFrame({'dataset' : [], 'seed' : []})
    all_exp_names = []
    all_hparam_dicts = []
    for dataset in all_datasets:
        if dataset in args.datasets:
            override = f"{args.data_configs[dataset]['base_optimizer_config']} "
            exp_names, hparam_dicts = run_across_seed_data_model(args, command, is_update_experiment=True, override_dataset=dataset, override_args=override)
            all_exp_names.extend(exp_names)
            all_hparam_dicts.extend(hparam_dicts)
    if args.collect_results:
        for exp_name, hparam_dict in zip(all_exp_names, all_hparam_dicts):
            args.num_random_other = hparam_dict['num_random_other']
            args.update_steps = hparam_dict['update_steps']
            args.num_successive_updates = 1
            data_stats    = accumulate_results_to_df(args, [exp_name], data_stats,    is_update_epoch=True, split_name='test', prefix='data_stats')
            summary_stats = accumulate_results_to_df(args, [exp_name], summary_stats, is_update_epoch=True, split_name='test', prefix='summary_stats')
        for df, prefix in zip([data_stats, summary_stats], 
                              ['data_stats', 'summary_stats']):
            save_path = os.path.join('aggregated_results', args.experiment + '_' + prefix + '.csv')
            df.to_csv(save_path, index=False)

def base_optimizers_r_ablation(args):
    command = f"python main.py --test_batch_size 32 --update_beliefs true --update_parameters all "\
              f" --leapofthought_main implicit_rule "
    args.job_params = args.job_params.replace("--do_train True", "--do_train false") # never train here
    data_stats    = pd.DataFrame({'dataset' : [], 'seed' : [], 'id' : []})
    summary_stats = pd.DataFrame({'dataset' : [], 'seed' : []})
    num_successive_updates = [1, 2, 4, 6, 8, 10]
    for dataset in ['FEVER', 'ZSRE']:
        for num_successive_update in num_successive_updates:
            override = f"{args.data_configs[dataset]['base_optimizer_config']} --num_successive_updates {num_successive_update} "
            exp_names, hparam_dicts = run_across_seed_data_model(args, command, is_update_experiment=True, override_dataset=dataset, override_args=override)
            if args.collect_results:
                for exp_name, hparam_dict in zip(exp_names, hparam_dicts):
                    args.num_random_other = hparam_dict['num_random_other']
                    args.update_steps = hparam_dict['update_steps']
                    args.num_successive_updates = hparam_dict['num_successive_updates']
                    data_stats    = accumulate_results_to_df(args, [exp_name], data_stats,    is_update_epoch=True, split_name='test', prefix='data_stats')
                    summary_stats = accumulate_results_to_df(args, [exp_name], summary_stats, is_update_epoch=True, split_name='test', prefix='summary_stats')
                for df, prefix in zip([data_stats, summary_stats], 
                                      ['data_stats', 'summary_stats']):
                    save_path = os.path.join('aggregated_results', args.experiment + '_' + prefix + '.csv')
                    df.to_csv(save_path, index=False)

def base_optimizers_r_main(args):
    command = f"python main.py --test_batch_size 32 --update_beliefs true --update_parameters all "\
              f" --leapofthought_main implicit_rule "
    args.job_params = args.job_params.replace("--do_train True", "--do_train false") # never train here
    data_stats    = pd.DataFrame({'dataset' : [], 'seed' : [], 'id' : []})
    summary_stats = pd.DataFrame({'dataset' : [], 'seed' : []})
    num_successive_updates = [10]
    for dataset in args.datasets:
        for num_successive_update in num_successive_updates:
            override = f"{args.data_configs[dataset]['base_optimizer_config']} --num_successive_updates {num_successive_update} "
            exp_names, hparam_dicts = run_across_seed_data_model(args, command, is_update_experiment=True, override_dataset=dataset, override_args=override)
            if args.collect_results:
                for exp_name, hparam_dict in zip(exp_names, hparam_dicts):
                    args.num_random_other = hparam_dict['num_random_other']
                    args.update_steps = hparam_dict['update_steps']
                    args.num_successive_updates = hparam_dict['num_successive_updates']
                    data_stats    = accumulate_results_to_df(args, [exp_name], data_stats,    is_update_epoch=True, split_name='test', prefix='data_stats')
                    summary_stats = accumulate_results_to_df(args, [exp_name], summary_stats, is_update_epoch=True, split_name='test', prefix='summary_stats')
                for df, prefix in zip([data_stats, summary_stats], 
                                      ['data_stats', 'summary_stats']):
                    save_path = os.path.join('aggregated_results', args.experiment + '_' + prefix + '.csv')
                    df.to_csv(save_path, index=False)

def learned_opt_main(args):
    command = f"python main.py --train_batch_size 16 --test_batch_size 16 --load_finetuned_model true --use_learned_optimizer true --update_steps 5 --learned_opt_steps 5 "\
              f" --update_parameters optimizer --update_eval_truthfully true --implementation new --leapofthought_main implicit_rule --leapofthought_add_both_for_training true "\
              f" --lr 3e-4 --weight_decay 0 --fit_to_alt_labels true --divergences kl  "
    data_stats = pd.DataFrame({'dataset' : [], 'seed' : [], 'id' : []})
    summary_stats = pd.DataFrame({'dataset' : [], 'seed' : []})
    exp_names, hparam_dicts = run_across_seed_data_model(args, command, is_update_experiment=True)
    if args.collect_results:
        for exp_name, hparam_dict in zip(exp_names, hparam_dicts):
            args.num_random_other = hparam_dict['num_random_other']
            args.update_steps = hparam_dict['update_steps']
            data_stats    = accumulate_results_to_df(args, [exp_name], data_stats,    is_update_epoch=True, split_name='test', prefix='data_stats')
            summary_stats = accumulate_results_to_df(args, [exp_name], summary_stats, is_update_epoch=True, split_name='test', prefix='summary_stats')
        for df, prefix in zip([data_stats, summary_stats], 
                              ['data_stats', 'summary_stats']):
            save_path = os.path.join('aggregated_results', args.experiment + '_' + prefix + '.csv')
            df.to_csv(save_path, index=False)

def learned_opt_eval_ablation(args):
    command = f"python main.py --train_batch_size 16 --test_batch_size 16 --load_finetuned_model true --use_learned_optimizer true --update_steps 5 --learned_opt_steps 5 "\
              f" --update_parameters optimizer --update_eval_truthfully true --implementation new "\
              f" --lr 3e-4 --weight_decay 0 --fit_to_alt_labels true --divergences kl  "
    data_stats = pd.DataFrame({'dataset' : [], 'seed' : [], 'id' : []})
    summary_stats = pd.DataFrame({'dataset' : [], 'seed' : []})
    assert args.dataset in ['ZSRE', 'Wikidata5m']
     # never train here
    args.job_params = args.job_params.replace("--do_train True", "--do_train false")
    for condition in ['true_labels', 'beam_labels']:
        if condition == 'true_labels':
            _command = command + "--update_eval_truthfully true --update_all_points false --eval_beam_search_alt_labels false "
        if condition == 'beam_labels':
            _command = command + "--update_eval_truthfully false --update_all_points true --eval_beam_search_alt_labels true "
        exp_names, hparam_dicts = run_across_seed_data_model(args, _command, is_update_experiment=True)
        if args.collect_results:
            for exp_name, hparam_dict in zip(exp_names, hparam_dicts):
                args.num_random_other = hparam_dict['num_random_other']
                args.update_steps = hparam_dict['update_steps']
                args.eval_beam_search_alt_labels = str2bool(hparam_dict['eval_beam_search_alt_labels'])
                data_stats    = accumulate_results_to_df(args, [exp_name], data_stats,    is_update_epoch=True, split_name='test', prefix='data_stats')
                summary_stats = accumulate_results_to_df(args, [exp_name], summary_stats, is_update_epoch=True, split_name='test', prefix='summary_stats')
            for df, prefix in zip([data_stats, summary_stats], 
                                  ['data_stats', 'summary_stats']):
                save_path = os.path.join('aggregated_results', args.experiment + '_' + prefix + '.csv')
                df.to_csv(save_path, index=False)

def learned_opt_k_ablation(args):
    command = f"python main.py --train_batch_size 16 --test_batch_size 16 --load_finetuned_model true --use_learned_optimizer true "\
              f" --update_parameters optimizer --update_eval_truthfully true --implementation new --leapofthought_main implicit_rule "\
              f" --lr 3e-4 --weight_decay 0 --fit_to_alt_labels true --divergences kl "
    override = f" --divergences kl --fit_opt_to_paraphrases false --fit_opt_to_dependent_propositions false --fit_opt_to_independent_propositions false --lambda_kl 1 "
    max_num_updates = [1, 2, 4, 6, 8, 10]
    data_stats = pd.DataFrame({'dataset' : [], 'seed' : [], 'id' : []})
    summary_stats = pd.DataFrame({'dataset' : [], 'seed' : []})
    all_exp_names = []
    all_hparam_dicts = []
    for max_num_update in max_num_updates:
        _command = command + f' --update_steps {max_num_update} --learned_opt_steps {max_num_update} '
        exp_names, hparam_dicts = run_across_seed_data_model(args, _command, is_update_experiment=True, override_args=override)
        all_exp_names.extend(exp_names)
        all_hparam_dicts.extend(hparam_dicts)
    # now evaluate generalization with r_train=1
    args.job_params = args.job_params.replace("--do_train True", "--do_train false") # never train here
    for max_num_update in max_num_updates:
        _command = command + f' --update_steps {max_num_update} --learned_opt_steps 1 '
        exp_names, hparam_dicts = run_across_seed_data_model(args, _command, is_update_experiment=True, override_args=override)
        all_exp_names.extend(exp_names)
        all_hparam_dicts.extend(hparam_dicts)
    if args.collect_results:
        for exp_name, hparam_dict in zip(all_exp_names, all_hparam_dicts):
            args.num_random_other = hparam_dict['num_random_other']
            args.update_steps = hparam_dict['update_steps']
            data_stats    = accumulate_results_to_df(args, [exp_name], data_stats,    is_update_epoch=True, split_name='test', prefix='data_stats')
            summary_stats = accumulate_results_to_df(args, [exp_name], summary_stats, is_update_epoch=True, split_name='test', prefix='summary_stats')
        for df, prefix in zip([data_stats, summary_stats], 
                              ['data_stats', 'summary_stats']):
            save_path = os.path.join('aggregated_results', args.experiment + '_' + prefix + '.csv')
            df.to_csv(save_path, index=False)

def learned_opt_r_main(args):
    command = f"python main.py --train_batch_size 16 --test_batch_size 16 --load_finetuned_model true --use_learned_optimizer true "\
              f" --update_parameters optimizer --update_eval_truthfully true --implementation new --leapofthought_main implicit_rule "\
              f" --lr 3e-4 --weight_decay 0 --fit_to_alt_labels true --divergences kl "
    data_stats = pd.DataFrame({'dataset' : [], 'seed' : [], 'id' : []})
    summary_stats = pd.DataFrame({'dataset' : [], 'seed' : []})
    all_exp_names = []
    all_hparam_dicts = []
    # case 1: r_train = r_test
    conditions = [
        {'r_train' : 10, 'r_test' : 10},
        {'r_train' : 1, 'r_test' : 10},
    ] 
    for dataset in args.datasets:
        for condition in conditions:
            k = 1 if dataset in ['FEVER', 'LeapOfThought'] else 5
            override = f"--update_steps {k} --learned_opt_steps {k}"
            r_train = condition['r_train']
            r_test = condition['r_test']
            _command = command + f' --learned_successive_updates {r_train} --num_successive_updates {r_test} -gaf {r_train} '
            seeds = [0] if condition['r_train'] == 1 else None
            exp_names, hparam_dicts = run_across_seed_data_model(args, _command, is_update_experiment=True, override_args=override, override_dataset=dataset, override_seeds=seeds)
            all_exp_names.extend(exp_names)
            all_hparam_dicts.extend(hparam_dicts) 
            print(f"\n r_train: {r_train} | r_test: {r_test} \n")
    if args.collect_results:
        for exp_name, hparam_dict in zip(all_exp_names, all_hparam_dicts):
            args.num_random_other = hparam_dict['num_random_other']
            args.update_steps = hparam_dict['update_steps']
            args.num_successive_updates = hparam_dict['num_successive_updates']
            data_stats    = accumulate_results_to_df(args, [exp_name], data_stats,    is_update_epoch=True, split_name='test', prefix='data_stats')
            summary_stats = accumulate_results_to_df(args, [exp_name], summary_stats, is_update_epoch=True, split_name='test', prefix='summary_stats')
        for df, prefix in zip([data_stats, summary_stats], 
                              ['data_stats', 'summary_stats']):
            save_path = os.path.join('aggregated_results', args.experiment + '_' + prefix + '.csv')
            df.to_csv(save_path, index=False)

def learned_opt_r_ablation(args):
    command = f"python main.py --train_batch_size 16 --test_batch_size 16 --load_finetuned_model true --use_learned_optimizer true "\
              f" --update_parameters optimizer --update_eval_truthfully true --implementation new --leapofthought_main implicit_rule "\
              f" --lr 3e-4 --weight_decay 0 --fit_to_alt_labels true --divergences kl "
    override = f" --divergences kl --fit_opt_to_paraphrases false --fit_opt_to_dependent_propositions false --fit_opt_to_independent_propositions false --lambda_kl 1 "
    num_successive_updates = [1, 2, 4, 6, 8, 10]
    data_stats = pd.DataFrame({'dataset' : [], 'seed' : [], 'id' : []})
    summary_stats = pd.DataFrame({'dataset' : [], 'seed' : []})
    all_exp_names = []
    all_hparam_dicts = []
    # case 1: r_train = r_test
    for dataset in ['FEVER', 'ZSRE']:
        k = 1 if dataset in ['FEVER', 'LeapOfThought'] else 5
        _override = override + f"--update_steps {k} --learned_opt_steps {k}"
        for num_successive_update in num_successive_updates:
            _command = command + f' --learned_successive_updates {num_successive_update} --num_successive_updates {num_successive_update} -gaf {num_successive_update} '
            exp_names, hparam_dicts = run_across_seed_data_model(args, _command, is_update_experiment=True, override_dataset=dataset, override_args=_override)
            all_exp_names.extend(exp_names)
            all_hparam_dicts.extend(hparam_dicts)            
            print(f"\n r_train: {num_successive_update} | r_test: {num_successive_update} \n")
    # case 2: r_train is one, test r=1:10
    for dataset in ['FEVER', 'ZSRE']:
        k = 1 if dataset in ['FEVER', 'LeapOfThought'] else 5
        _override = override + f"--update_steps {k} --learned_opt_steps {k}"
        args.job_params = args.job_params.replace("--do_train True", "--do_train false") # never train here
        for r_train in [1]:
            for num_successive_update in num_successive_updates:
                _command = command + f' --learned_successive_updates {r_train} --num_successive_updates {num_successive_update} '
                exp_names, hparam_dicts = run_across_seed_data_model(args, _command, is_update_experiment=True, override_dataset=dataset, override_args=_override)
                all_exp_names.extend(exp_names)
                all_hparam_dicts.extend(hparam_dicts)
                print(f"\n r_train: {r_train} | r_test: {num_successive_update} \n")
    if args.collect_results:
        for exp_name, hparam_dict in zip(all_exp_names, all_hparam_dicts):
            args.num_random_other = hparam_dict['num_random_other']
            args.update_steps = hparam_dict['update_steps']
            args.num_successive_updates = hparam_dict['num_successive_updates']
            data_stats    = accumulate_results_to_df(args, [exp_name], data_stats,    is_update_epoch=True, split_name='test', prefix='data_stats')
            summary_stats = accumulate_results_to_df(args, [exp_name], summary_stats, is_update_epoch=True, split_name='test', prefix='summary_stats')
        for df, prefix in zip([data_stats, summary_stats], 
                              ['data_stats', 'summary_stats']):
            save_path = os.path.join('aggregated_results', args.experiment + '_' + prefix + '.csv')
            df.to_csv(save_path, index=False)

def learned_opt_objective_ablation(args):
    command = f"python main.py --train_batch_size 16 --test_batch_size 16 --load_finetuned_model true --use_learned_optimizer true --update_steps 5 --learned_opt_steps 5 "\
              f" --update_parameters optimizer --update_eval_truthfully true --implementation new --leapofthought_main implicit_rule --leapofthought_add_both_for_training true "\
              f" --lr 3e-4 --weight_decay 0 --fit_to_alt_labels true --divergences kl  "
    objective_conditions_to_datasets = {
            '--divergences none --fit_opt_to_paraphrases false --fit_opt_to_independent_propositions false --fit_opt_to_dependent_propositions false' : ['FEVER'],
            '--divergences kl   --fit_opt_to_paraphrases false --fit_opt_to_independent_propositions false --fit_opt_to_dependent_propositions false' : all_datasets,
            '--divergences kl   --fit_opt_to_paraphrases true  --fit_opt_to_independent_propositions false --fit_opt_to_dependent_propositions false' : ['ZSRE', 'Wikidata5m'],
            '--divergences kl   --fit_opt_to_paraphrases false --fit_opt_to_independent_propositions true  --fit_opt_to_dependent_propositions false'\
                ' --lambda_independents_updated .5 --lambda_kl .5'                                                                                     : ['Wikidata5m'],
            '--divergences kl   --fit_opt_to_paraphrases true  --fit_opt_to_independent_propositions true  --fit_opt_to_dependent_propositions false'\
                ' --lambda_independents_updated .5 --lambda_kl .5'                                                                                     : ['Wikidata5m'],
            '--divergences kl   --fit_opt_to_paraphrases false --fit_opt_to_independent_propositions false --fit_opt_to_dependent_propositions true ' : ['LeapOfThought']
    }
    data_stats = pd.DataFrame({'dataset' : [], 'seed' : [], 'id' : []})
    summary_stats = pd.DataFrame({'dataset' : [], 'seed' : []})
    for objective_condition, datasets in objective_conditions_to_datasets.items():
        for dataset in datasets:
            if dataset in args.datasets:
                override = f"{objective_condition} "
                print(f'\n{override}\n')
                exp_names, hparam_dicts = run_across_seed_data_model(args, command, is_update_experiment=True, override_dataset=dataset, override_args=override)
                if args.collect_results:
                    for exp_name, hparam_dict in zip(exp_names, hparam_dicts):
                        args.num_random_other = hparam_dict['num_random_other']
                        args.update_steps = hparam_dict['update_steps']
                        data_stats    = accumulate_results_to_df(args, [exp_name], data_stats,    is_update_epoch=True, split_name='test', prefix='data_stats')
                        summary_stats = accumulate_results_to_df(args, [exp_name], summary_stats, is_update_epoch=True, split_name='test', prefix='summary_stats')
                    for df, prefix in zip([data_stats, summary_stats], 
                                          ['data_stats', 'summary_stats']):
                        save_path = os.path.join('aggregated_results', args.experiment + '_' + prefix + '.csv')
                        df.to_csv(save_path, index=False)

def learned_opt_label_ablation(args):
    command = f"python main.py --train_batch_size 16 --test_batch_size 16 --load_finetuned_model true --use_learned_optimizer true --update_steps 5 --learned_opt_steps 5 "\
              f" --update_parameters optimizer --update_eval_truthfully true --implementation new "\
              f" --lr 3e-4 --weight_decay 0 --fit_to_alt_labels true --divergences kl  "
    data_stats = pd.DataFrame({'dataset' : [], 'seed' : [], 'id' : []})
    summary_stats = pd.DataFrame({'dataset' : [], 'seed' : []})
    alt_labels = ['beam', 'random']
    for alt_label in alt_labels:
        if alt_label == 'random':
            _command = command + f' --beam_search_alt_labels false '
        elif alt_label == 'beam':
            _command = command + f' --beam_search_alt_labels true '
        exp_names, hparam_dicts = run_across_seed_data_model(args, _command, is_update_experiment=True)
        if args.collect_results:
            for exp_name, hparam_dict in zip(exp_names, hparam_dicts):
                args.num_random_other = hparam_dict['num_random_other']
                args.update_steps = hparam_dict['update_steps']
                args.alt_label = alt_label
                data_stats    = accumulate_results_to_df(args, [exp_name], data_stats,    is_update_epoch=True, split_name='test', prefix='data_stats')
                summary_stats = accumulate_results_to_df(args, [exp_name], summary_stats, is_update_epoch=True, split_name='test', prefix='summary_stats')
            for df, prefix in zip([data_stats, summary_stats], 
                                  ['data_stats', 'summary_stats']):
                save_path = os.path.join('aggregated_results', args.experiment + '_' + prefix + '.csv')
                df.to_csv(save_path, index=False)

def learned_opt_de_cao(args):
    command = f"python main.py --train_batch_size 16 --test_batch_size 16 --load_finetuned_model true --use_learned_optimizer true  "\
              f" --update_parameters optimizer --update_eval_truthfully true --implementation new --leapofthought_main implicit_rule "\
              f" --lr 3e-4 --weight_decay 0 --fit_to_alt_labels true --divergences kl "
    data_stats = pd.DataFrame({'dataset' : [], 'seed' : [], 'id' : []})
    summary_stats = pd.DataFrame({'dataset' : [], 'seed' : []})
    override = '--fit_opt_to_paraphrases true'
    _command = command + f' --beam_search_alt_labels true  --learned_opt_steps 1 --update_steps 5 '
    exp_names, hparam_dicts = run_across_seed_data_model(args, _command, is_update_experiment=True, override_args=override)
    if args.collect_results:
        for exp_name, hparam_dict in zip(exp_names, hparam_dicts):
            args.num_random_other = hparam_dict['num_random_other']
            args.update_steps = hparam_dict['update_steps']
            data_stats    = accumulate_results_to_df(args, [exp_name], data_stats,    is_update_epoch=True, split_name='test', prefix='data_stats')
            summary_stats = accumulate_results_to_df(args, [exp_name], summary_stats, is_update_epoch=True, split_name='test', prefix='summary_stats')
        for df, prefix in zip([data_stats, summary_stats], 
                              ['data_stats', 'summary_stats']):
            save_path = os.path.join('aggregated_results', args.experiment + '_' + prefix + '.csv')
            df.to_csv(save_path, index=False)

def learned_opt_sample_efficiency(args):
    # doesn't save the models, currently. 
    command = f"python main.py --train_batch_size 16 --test_batch_size 16 --load_finetuned_model true --use_learned_optimizer true --update_steps 5 --learned_opt_steps 5 "\
              f" --update_parameters optimizer --update_eval_truthfully true --implementation new --leapofthought_main implicit_rule --leapofthought_add_both_for_training true "\
              f" --lr 3e-4 --weight_decay 0 --fit_to_alt_labels true --divergences kl  "
    data_stats = pd.DataFrame({'dataset' : [], 'seed' : [], 'id' : []})
    summary_stats = pd.DataFrame({'dataset' : [], 'seed' : []})
    for n in [1000, 5000, 10000, 50000, 100000]:
        override = f" --num_train_points {n} "
        if args.dataset in ['ZSRE', 'Wikidata5m']:
            override += " --eval_after 3 "
        exp_names, hparam_dicts = run_across_seed_data_model(args, command, is_update_experiment=True, override_args=override)
        if args.collect_results:
            for exp_name, hparam_dict in zip(exp_names, hparam_dicts):
                args.num_random_other = hparam_dict['num_random_other']
                args.update_steps = hparam_dict['update_steps']
                args.num_train_points = hparam_dict['num_train_points']
                data_stats    = accumulate_results_to_df(args, [exp_name], data_stats,    is_update_epoch=True, split_name='test', prefix='data_stats')
                summary_stats = accumulate_results_to_df(args, [exp_name], summary_stats, is_update_epoch=True, split_name='test', prefix='summary_stats')
            for df, prefix in zip([data_stats, summary_stats], 
                                  ['data_stats', 'summary_stats']):
                save_path = os.path.join('aggregated_results', args.experiment + '_' + prefix + '.csv')
                df.to_csv(save_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", '-e', type=str) 
    parser.add_argument("--submit", default='os', choices=['os', 'slurm']) 
    parser.add_argument("--gpu", default=-1, type=int) 
    parser.add_argument("--server", default='learnaccel', type=str) 
    parser.add_argument("--seeds", default=1, type=int) 
    parser.add_argument("--start", default=0, type=int, help='start seed') 
    parser.add_argument("--dataset", default='all', type=str, help='')
    parser.add_argument("--num_train_epochs", type=int, help='')
    parser.add_argument("--train_batch_size", type=int, help='')
    parser.add_argument("--test_batch_size", type=int, help='')
    parser.add_argument("--grad_accumulation_factor", '-gaf', default=-1, type=int, help='')
    parser.add_argument("--small_data", '-s', action='store_true')
    parser.add_argument("--do_train", default=True, type=str2bool)
    parser.add_argument("--overwrite_past_experiment", default=False, type=str2bool)
    parser.add_argument("--use_dev_not_test", default=False, type=str2bool)
    parser.add_argument("--collect_results", '-c', default=False, type=str2bool)
    parser.add_argument("--data_dir", default='/playpen3/home/peter/data', type=str, help='')
    parser.add_argument("--save_dir", default='/playpen3/home/peter/saved_models', type=str, help='')
    parser.add_argument("--cache_dir", default='/playpen3/home/peter/cached_models', type=str, help='')
    args = parser.parse_args()

    '''
    later arguments override earlier arguments in commands
    note that the final order of arguments is:
        f"{command} {job_params}"
    - command comes from the experiment-specific function
    - job_params defined in run_across_seed_data_model
    '''
    if not os.path.exists('aggregated_results'):        os.mkdir('aggregated_results')

    # arg checks
    if 'tune' in args.experiment:                       assert args.use_dev_not_test, "always use dev for tuning experiments"
    if 'tune' not in args.experiment:                   assert not args.use_dev_not_test, "always use test for non-tuning experiments"
    if args.submit == 'os':                             args.collect_results = True # always collect results when running everything serially
    if args.experiment == 'write_LeapOfThought_preds':  assert not args.do_train and args.dataset == 'LeapOfThought' # never train in this experiment
    if args.experiment == 'write_ZSRE_alt_preds':       assert not args.do_train and args.dataset == 'ZSRE'
    if args.experiment == 'learned_opt_across_k':       assert args.dataset == 'ZSRE'
    if args.experiment == 'learned_opt_label_ablation': assert args.dataset == 'ZSRE'

    # mkdirs
    if not os.path.exists('slurm_logs'):
        os.mkdir('slurm_logs')

    # slurm bug
    if args.submit == 'slurm':
        os.environ['MKL_THREADING_LAYER'] = 'GNU'
 
    # add args
    args.seeds = list(range(args.start, args.seeds+args.start))
    args.job_params = f"{'-s -us' if args.small_data else ''} "\
                   f"--do_train {args.do_train} "\
                   f"--overwrite_past_experiment {args.overwrite_past_experiment} "\
                   f"--use_dev_not_test {args.use_dev_not_test} "\
                   f"--data_dir {args.data_dir} "\
                   f"--save_dir {args.save_dir} "\
                   f"--cache_dir {args.cache_dir} "\
                   f"--server {args.server} "\
                   f"--write_statistics true "
    if args.train_batch_size is not None:       args.job_params += f" --train_batch_size {args.train_batch_size} "
    if args.test_batch_size is not None:        args.job_params += f" --test_batch_size {args.test_batch_size} "
    if args.grad_accumulation_factor > -1:      args.job_params += f" -gaf {args.grad_accumulation_factor} "
    if args.gpu >= 0:                           args.job_params += f" --gpu {args.gpu} "    
    all_datasets = ['FEVER', 'LeapOfThought', 'Wikidata5m', 'ZSRE']
    args.datasets = all_datasets if args.dataset == 'all' else [args.dataset]
    args.data_configs = {
        'FEVER' : {
            'num_train_epochs' : 10,
            'num_train_epochs_learned_opt' : 5,
            'num_random_other' : 200,
            'probing_style' : 'model',
            'probe' : 'linear',
            'model' : 'roberta-base',
            'learned_opt_train_eval_size' : -1,
            'learned_opt_default_objectives' : '',
            'base_optimizer_config' : '--optimizer adamw --update_steps 100 --lr 1e-6'
        },
        'LeapOfThought' : {
            'num_train_epochs' : 10,
            'num_train_epochs_learned_opt' : 10,
            'num_random_other' : 200,
            'probing_style' : 'model',
            'probe' : 'linear',
            'model' : 'roberta-base',
            'learned_opt_train_eval_size' : -1,
            'learned_opt_default_objectives' : " --fit_opt_to_dependent_propositions false ",
            'base_optimizer_config' : '--optimizer sgd --update_steps 100 --lr 1e-2'
        },
        'ZSRE' : {
            'num_train_epochs' : 20,
            'num_train_epochs_learned_opt' : 5,
            'num_random_other' : 30,
            'probing_style' : 'seq2seq',
            'probe' : None,
            'model' : 'facebook/bart-base',
            'learned_opt_train_eval_size' : 4000,
            'learned_opt_default_objectives' : "--fit_opt_to_paraphrases false ",
            'base_optimizer_config' : '--optimizer sgd --update_steps 10 --lr 1e-1'
        },
        'Wikidata5m' : {
            'num_train_epochs' : 20,
            'num_train_epochs_learned_opt' : 5,
            'num_random_other' : 30,
            'probing_style' : 'seq2seq',
            'probe' : None,
            'model' : 'facebook/bart-base',
            'learned_opt_train_eval_size' : 4000,
            'learned_opt_default_objectives' : "--fit_opt_to_paraphrases true --fit_opt_to_independent_propositions false ",
            'base_optimizer_config' : '--optimizer sgd --update_steps 10 --lr 1e-1'
        },
    }
    args.optimizer_to_sweep_lrs = {
        'adamw'   : [1e-4, 1e-5, 1e-6],
        'rmsprop' : [1e-4, 1e-5, 1e-6],
        'sgd'     : [1e-1, 1e-2, 1e-3],
    }
    # default values for some hparams needed for getting experiment names. these will be overridden depending on the experiment
    args.num_random_other = 1
    args.update_steps = 1
    args.num_successive_updates = 1

    # specify job to run
    if args.experiment == 'task_model':                     task_model(args)
    if args.experiment == 'tune_base_optimizers':           tune_base_optimizers(args)
    if args.experiment == 'base_optimizers':                base_optimizers(args)
    if args.experiment == 'base_optimizers_r_main':         base_optimizers_r_main(args)
    if args.experiment == 'base_optimizers_r_ablation':     base_optimizers_r_ablation(args)
    if args.experiment == 'write_LeapOfThought_preds':      write_LeapOfThought_preds(args)
    if args.experiment == 'write_alt_beam_preds':           write_alt_beam_preds(args)
    if args.experiment == 'learned_opt_main':               learned_opt_main(args)
    if args.experiment == 'learned_opt_r_main':             learned_opt_r_main(args)
    if args.experiment == 'learned_opt_k_ablation':         learned_opt_k_ablation(args)
    if args.experiment == 'learned_opt_r_ablation':         learned_opt_r_ablation(args)
    if args.experiment == 'learned_opt_objective_ablation': learned_opt_objective_ablation(args)
    if args.experiment == 'learned_opt_label_ablation':     learned_opt_label_ablation(args)
    if args.experiment == 'learned_opt_eval_ablation':      learned_opt_eval_ablation(args)
    if args.experiment == 'learned_opt_de_cao':             learned_opt_de_cao(args)
    if args.experiment == 'learned_opt_sample_efficiency':  learned_opt_sample_efficiency(args)