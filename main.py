from types import SimpleNamespace
import numpy as np
import torch
import pandas as pd
import argparse
import utils
import metrics
import time
import os
import sys
import jsonlines
from utils import str2bool, Report, safe_load_base_model, safe_load_final_model
from transformers import AutoModelForSeq2SeqLM, AutoModelWithLMHead, AutoModel, AutoTokenizer
from transformers import AdamW, get_scheduler, get_linear_schedule_with_warmup
from torch.optim import SGD, RMSprop
from torch.nn.parallel import DistributedDataParallel as DDP
from models.probe import Probe
from models.learned_optimizer import ModelWithLearnedOptimizer
from copy import deepcopy
import networkx as nx
import matplotlib.pyplot as plt
import graph_utils
from types import SimpleNamespace

def load_optimizer_and_scheduler(args, model, num_training_steps):
    if args.update_parameters == 'probe':
        named_parameters = model.probe.named_parameters()
    if args.update_parameters == 'all':
        named_parameters = model.named_parameters()
    if args.update_parameters == 'biases':
        named_parameters = [(n,p) for n,p in model.named_parameters() if 'bias' in n]
    if args.update_parameters == 'de_cao':
        avoid_list = ['norm', 'embeddings', 'classifier', 'pooler', 'shared', 'embed', 'positions', 'bias']
        named_parameters = [(n,p) for n,p in model.named_parameters() if all([e not in n.lower() for e in avoid_list])]
    if args.update_parameters == 'interior':
        avoid_list = ['norm', 'embeddings', 'classifier', 'pooler', 'shared', 'embed', 'positions']
        named_parameters = [(n,p) for n,p in model.named_parameters() if all([e not in n.lower() for e in avoid_list])]
    if args.update_parameters == 'optimizer':
        named_parameters = [(n,p) for n,p in model.named_parameters() if n.startswith('learner')]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in named_parameters if not any(nd in n for nd in ["bias", "LayerNorm.weight"])],
            "weight_decay": args.weight_decay,
            'lr' : args.lr},
        {"params": [p for n, p in named_parameters if any(nd in n for nd in ["bias", "LayerNorm.weight"])],
            "weight_decay": 0.0,
            'lr' : args.lr}
    ]
    # add alpha_kl lr
    if type(model) is ModelWithLearnedOptimizer:
        if args.implementation == 'de_cao':
            optimizer_grouped_parameters.append({
                'params' : [model.alpha_kl],
                'lr' : 1e-1,
                'weight_decay' : 0
            })
    optimizer_to_class = {'adamw' : AdamW, 'sgd' : SGD, 'rmsprop' : RMSprop}
    optimizer_class = optimizer_to_class[args.optimizer]
    if optimizer_class is RMSprop:
        optimizer = optimizer_class(optimizer_grouped_parameters, centered=True)
    else:
        optimizer = optimizer_class(optimizer_grouped_parameters)
    if args.implementation == 'de_cao' and args.use_learned_optimizer: # use this with use_learned_opt, not args.update_beliefs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=1000, num_training_steps=200000)
    else:
        scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
    return (optimizer, scheduler)


def update_model(args, model, datapoint_kwargs, tokenizer):
    '''
    deepcopies a model and updates it towards getting the right output for a single data point
    returns the binary 1/0 correctness of the final prediction and the resulting model
    '''
    model.eval()
    optimizer, scheduler = load_optimizer_and_scheduler(args, model, num_training_steps=2e8) 
    new_scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)
    label = datapoint_kwargs['labels']
    max_update_steps = args.update_steps if args.update_steps > 0 else int(2e8)
    for _ in range(max_update_steps):
        # forward pass on main input
        with torch.enable_grad() and torch.cuda.amp.autocast(enabled=args.fp16):
            outputs = model(**datapoint_kwargs)
        # step optimizer
        loss = outputs['loss']
        new_scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        new_scaler.step(optimizer)
        new_scaler.update()
        optimizer.zero_grad()
        scheduler.step() # this must be stepped or optimizer lr is always 0 (even though lr is constant)
        # get generative output if seq2seq
        if args.probing_style=='seq2seq':
            outputs = model(is_eval=True, **datapoint_kwargs)
        pred = outputs['preds']
        # break from updating when update is successful
        update_successful = metrics.compute_acc_sum(args.probing_style, pred, label, tokenizer) # either 0 or 1
        if update_successful:
            break
    update_correct = metrics.compute_acc_sum(args.probing_style, pred, datapoint_kwargs['orig_labels'], tokenizer)
    return outputs, update_successful, update_correct, model


def train_or_test(args, stats_dict, epoch, model, data_loader, tokenizer, optimizer=None, scheduler=None, scaler=None,
                  write_preds_to_file=False, break_after_n_points=-1, pre_eval=False):
    '''
    main train_or_test function that trains and evaluates base task models as well as learned optimizers for updating beliefs
    returns stats_dict 
    '''
    # local variables. stored in stat dict along with copies with 'before_' and 'after_' prefixes for tracking effects of model updates
    epoch_stats = utils.init_epoch_stats(stat_names = [
        'acc_sum',
        'n_data_points',
        'n_batches',
        'par_acc_sum',
        'par_eq_sum',
        'n_paraphrases',
        'n_paraphrase_pairs',
        'consistent_sum',
        'retain_sum',
        'n_random_other_points',
        'succ_updates_sum',
        'corrupted_sum',
        'n_updated',
        'dependent_acc_sum',
        'dependent_eq_sum',        
        'n_dependent_points',
        'other_dependent_acc_sum',
        'n_other_dependent_points',
        'independent_acc_sum',
        'independent_ret_sum',
        'n_independent_points',
        'entity_paraphrase_acc_sum',
        'entity_paraphrase_eq_sum',
        'n_entity_paraphrases',
        'relation_paraphrase_acc_sum',
        'relation_paraphrase_eq_sum',
        'n_relation_paraphrases'
    ])
    id_to_stats = {idx : {} for idx in range(len(data_loader.dataset))} # for individual data statistics
    epoch_stats['n_batches'] = n_batches = len(data_loader)
    start_time = time.time()
    split_name = data_loader.dataset.split_name
    data_names = data_loader.dataset.data_names
    is_train_epoch = (split_name.lower() == 'train' and args.do_train and optimizer is not None) # this means we're doing optimizer.step()
    is_eval_epoch = (not is_train_epoch)
    _update_all_points = (args.update_all_points if not (is_eval_epoch and args.update_eval_truthfully) else False)
    is_update_epoch = (split_name.lower() != 'train' and not pre_eval and (args.update_beliefs or args.use_learned_optimizer)) 
    main_grad_req = torch.enable_grad() if (is_train_epoch or args.use_learned_optimizer) else torch.no_grad()
    forward_time = 0
    updated_ids = []
    all_main_preds = []
    all_alt_preds = []
    all_main_correct = []
    all_update_correct = []
    all_independent_correct = []
    all_dependent_correct = []
    all_keep_ids = []
    all_entity_seen_in_training = []
    all_data_labels = []
    id_to_flipped_id_dict = {} # used with write_graph_to_file
    id_to_updated_pred = {} # used with write_graph_to_file
    eval_updated_model_counter = 0 # will eval every num_successive_updates points if use_learned_optimizer or update_beliefs
    # keep track of these stats across batches in order to always evaluate num_successive_updates data points
    running_batch_ids = []
    running_updated_ids = []
    # get counts used for equally distributing the num. times each data point is used in computing change in other data accuracy from updating (per_chg). see get_random_subset
    data_id_and_sampled_times = {id: 0 for id in range(len(data_loader.dataset))}
    odp_data_id_and_sampled_times = {id: 0 for id in range(len(data_loader.dataset))}
    # load the base model predictions for data if need them later
    if is_update_epoch:
        data_path = getattr(args, f"{split_name}_path")
        pred_name = f'{args.base_experiment_name}_pred' if args.dataset != 'LeapOfThought' else f'{args.base_experiment_name}_{args.leapofthought_main}_pred'
        all_before_preds = np.array([point[pred_name] for point in jsonlines.open(data_path)])
        if args.dataset == 'Wikidata5m':
            all_independent_preds = utils.flip_array_pairwise(all_before_preds) # only holds with our pairwise writing of the dev/test splits for Wikidata5m
        if args.dataset == 'LeapOfThought' and not args.write_graph_to_file:
            data_path = getattr(args, f"{split_name}_path")
            pred_name = f'{args.base_experiment_name}_hypothesis_pred'
            all_odp_preds = np.array([point[pred_name] for point in jsonlines.open(data_path)])
        else:
            all_odp_preds = None
    else:
        all_before_preds = None
        all_odp_preds = None
    if is_eval_epoch and args.update_beliefs:
        updated_model = deepcopy(model) # used to reset the updated model when updating with torch optimizer. need to start off with it, since grad updates applied to it

    if is_train_epoch:
        model.train() # note that in ModelWithLearnedOptimizer, model.model.eval() is always called on forward pass
    else:
        model.eval()
    
    for batch_num, batch in enumerate(data_loader):
        running_time = (time.time()-start_time)
        est_epoch_run_time = (running_time/(batch_num+1)*n_batches)
        batch_size = batch['main_input_ids'].size(0)
        args.report.print_training_prog(split_name, stats_dict, batch_num, n_batches, running_time, est_epoch_run_time, forward_time,
                                        stats=['acc', 'upd_suc', 'per_chg', 'ind_chg', 'par_eq', 'cons', 'dep_acc'] if not args.write_graph_to_file else ['acc', 'upd_suc', 'oth_ret'])

        # forward pass on main input. if using learned optimizer and this is the before-training eval or is_update_epoch, use the original model
        main_kwargs = {k.replace('main_', "") : v for k,v in batch.items() if any([name in k for name in ['main','paraphrases','orig_labels','dependent']]) or k=='id'}
        utils.move_kwargs_to_gpu(main_kwargs)
        if args.use_learned_optimizer and (pre_eval or is_update_epoch): main_kwargs['use_base_model'] = True
        with main_grad_req and torch.cuda.amp.autocast(enabled=args.fp16):
            begin = time.time()
            main_outputs = model(is_eval=is_eval_epoch, **main_kwargs)
            forward_time += (time.time() - begin)

        # step optimizer for training a model
        if is_train_epoch:
            loss = main_outputs['loss'] / args.grad_accumulation_factor
            if args.multi_gpu:
                loss = loss.mean()
            retain_graph = (args.num_successive_updates > 1 and not args.detach_prev_updates)
            scaler.scale(loss).backward(retain_graph=retain_graph)
            if (batch_num+1) % args.grad_accumulation_factor == 0 or (batch_num == n_batches-1):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

                # free up memory
                del loss, main_kwargs

        # update stats for main batch. note that for seq prediction problems we use eligible_labels for exact-match
        preds = main_outputs['preds']
        labels = batch['orig_labels'] if args.probing_style!='seq2seq' else [item['eligible_labels'] for item in batch['text_data']] 
        main_n_correct, main_binary_correct = metrics.compute_acc_sum(args.probing_style, preds, labels, tokenizer, return_where_correct=True)
        epoch_stats['acc_sum'] += main_n_correct
        epoch_stats['n_data_points'] += batch_size
        all_main_preds.append(preds.detach() if type(preds) is torch.Tensor else preds)
        all_alt_preds.append(main_outputs['alt_preds'] if 'alt_preds' in main_outputs else None)
        all_main_correct.extend(main_binary_correct.tolist())
        if args.dataset in ['FEVER', 'LeapOfThought'] and args.write_statistics:
            all_data_labels.extend(labels.tolist())
        # get metrics while training the learned optimizer
        if args.use_learned_optimizer and is_train_epoch and not is_update_epoch: # used during training the optimizer, not evaluating it
            if batch_size > 1:
                before_preds = main_outputs['all_before_preds'] # these are for the points in the batch not used to get the model grad
                after_preds = main_outputs['all_after_preds']
                all_labels = main_outputs['all_labels']
                epoch_stats['retain_sum'] += metrics.compute_acc_sum(args.probing_style, before_preds, after_preds, tokenizer)
                n_corrupted, n_right_before = metrics.get_num_corrupted(args.probing_style, before_preds, after_preds, all_labels, tokenizer)
                epoch_stats['corrupted_sum'] += n_corrupted
                epoch_stats['before_acc_sum'] += n_right_before
                epoch_stats['n_random_other_points'] += batch_size - 1 # one point is used for grad updated in model.forward
                epoch_stats['after_acc_sum'] += metrics.compute_acc_sum(args.probing_style, after_preds, all_labels, tokenizer)
            epoch_stats['succ_updates_sum'] += main_outputs['update_succ']
            epoch_stats['n_updated'] += 1 # one point is used for grad updated in model.forward
            if args.fit_opt_to_dependent_propositions:
                if 'all_dependent_preds' in main_outputs:
                    main_dep_preds = main_outputs['all_dependent_preds']
                    epoch_stats['dependent_eq_sum'] += metrics.compute_acc_sum(args.probing_style, main_dep_preds, main_outputs['updated_pred'], tokenizer)
                    epoch_stats['dependent_acc_sum'] += metrics.compute_acc_sum(args.probing_style, main_dep_preds, main_outputs['all_dependent_orig_labels'], tokenizer)
                    epoch_stats['n_dependent_points'] += 1 # one point at most is dependent on the updated point
            if args.fit_opt_to_paraphrases:
                all_paraphrase_preds = main_outputs['all_paraphrase_preds']
                update_idx = main_outputs['update_idx'].item()
                if len(all_paraphrase_preds) > 1:
                    orig_labels = [batch['orig_labels'][update_idx]] if args.probing_style!='seq2seq' else [batch['text_data'][update_idx]['eligible_labels']] 
                    orig_labels = list(orig_labels) * len(all_paraphrase_preds)
                    epoch_stats['par_acc_sum'] += metrics.compute_acc_sum(args.probing_style, all_paraphrase_preds, orig_labels, tokenizer) # these labels are updated_pred repeated as many times as there are paraphrases
                    epoch_stats['par_eq_sum'] += metrics.compute_acc_sum(args.probing_style, all_paraphrase_preds, main_outputs['all_paraphrase_eq_labels'], tokenizer)
                    epoch_stats['n_paraphrases'] += len(main_outputs['all_paraphrase_preds'])

        # two big eval cases: updating model and not updating model
        if is_eval_epoch:
            
            if not is_update_epoch:
                # pick the eval_model, and points to evaluate on
                eval_model = model
                eval_batch = batch
                keep_idx = np.arange(batch_size)
                if args.eval_subset == 'right':
                    keep_idx = np.intersect1d(keep_idx, np.argwhere(main_binary_correct).reshape(-1))
                if args.eval_subset == 'wrong':
                    keep_idx = np.intersect1d(keep_idx, np.argwhere(1-main_binary_correct).reshape(-1))
                all_keep_ids.extend(eval_batch['id'][keep_idx].tolist())
                # paraphrase metrics
                if args.eval_consistency and 'paraphrases' in data_names:
                    # only compute this if the requested keep_idx have associated paraphrases
                    paraphrase_metrics = metrics.get_metrics_on_named_data(args, main_grad_req, eval_model, tokenizer, eval_batch, 
                                                                        data_name='paraphrases', keep_idx=keep_idx, reference_preds=main_outputs['preds'])
                    epoch_stats['par_acc_sum'] += paraphrase_metrics['par_acc_sum']
                    epoch_stats['par_eq_sum'] += paraphrase_metrics['par_eq_sum']
                    epoch_stats['n_paraphrases'] += paraphrase_metrics['n_paraphrases']
                    epoch_stats['consistent_sum'] += paraphrase_metrics['n_consistent']
                    epoch_stats['n_paraphrase_pairs'] += paraphrase_metrics['n_paraphrase_pairs']
                    # None paraphrases were dropped inside get_metrics
                    where_not_none = np.argwhere([paraphrases is not None for paraphrases in eval_batch['paraphrases']]).reshape(-1)
                    where_eligible = np.intersect1d(where_not_none, keep_idx)
                    # add cons scores to id_to_stats. in this case, it's the position in keep_idx
                    for pos, idx in enumerate(where_eligible):
                        data_id = eval_batch['id'][idx].item()
                        if paraphrase_metrics['point_level_par_eqs'] is not None:
                            id_to_stats[data_id]['par_eq'] = ' '.join([str(1*item) for item in paraphrase_metrics['point_level_par_eqs'][pos]])
                            id_to_stats[data_id]['cons'] = paraphrase_metrics['point_level_cons'][pos]
                        else:
                            id_to_stats[data_id]['par_eq'] = id_to_stats[data_id]['cons'] = np.nan
                # eval: dependent propositions, independent propositions, entity paraphrases, and relation paraphrases
                if 'dependent_proposition' in data_names and \
                (args.leapofthought_main == 'implicit_rule' if args.dataset == 'LeapOfThought' else True):
                    output_metrics = metrics.get_metrics_on_named_data(args, main_grad_req, eval_model, tokenizer, eval_batch, 
                                                                        data_name='dependent_proposition', keep_idx=keep_idx, reference_preds=main_outputs['preds'])
                    epoch_stats['dependent_acc_sum'] += output_metrics['orig_acc_sum']
                    epoch_stats['dependent_eq_sum'] += output_metrics['ref_acc_sum']
                    epoch_stats['n_dependent_points'] += output_metrics['n_points']
                    all_dependent_correct.extend(output_metrics['orig_binary_correct'].tolist())
                    for pos, idx in enumerate(keep_idx):
                        data_id = eval_batch['id'][idx].item()
                        id_to_stats[data_id]['dep_eq'] = output_metrics['ref_binary_correct'][pos]
                        id_to_stats[data_id]['dep_acc'] = output_metrics['orig_binary_correct'][pos]
                if 'independent_proposition' in data_names:
                    output_metrics = metrics.get_metrics_on_named_data(args, main_grad_req, eval_model, tokenizer, eval_batch, 
                                                                        data_name='independent_proposition', keep_idx=keep_idx)
                    epoch_stats['independent_acc_sum'] += output_metrics['acc_sum']
                    epoch_stats['n_independent_points'] += output_metrics['n_points']
                    all_independent_correct.extend(output_metrics['binary_correct'].tolist())
                    for pos, idx in enumerate(keep_idx):
                        data_id = eval_batch['id'][idx].item()
                        id_to_stats[data_id]['ind_acc'] = output_metrics['binary_correct'][pos]
                if 'entity_paraphrase' in data_names and args.eval_paraphrase_types:
                    where_not_none = np.argwhere([item['entity_paraphrase'] is not None for item in eval_batch['text_data']]).reshape(-1)
                    entity_para_keep_idx = np.intersect1d(where_not_none, keep_idx)
                    eligible_paraphrases_exist = len(entity_para_keep_idx) > 0
                    if eligible_paraphrases_exist:
                        output_metrics = metrics.get_metrics_on_named_data(args, main_grad_req, eval_model, tokenizer, eval_batch, 
                                                                            data_name='entity_paraphrase', keep_idx=entity_para_keep_idx, reference_preds=main_outputs['preds'])
                        epoch_stats['entity_paraphrase_acc_sum'] += output_metrics['acc_sum']
                        epoch_stats['entity_paraphrase_eq_sum'] += output_metrics['ref_acc_sum']
                        epoch_stats['n_entity_paraphrases'] += output_metrics['n_points']
                        for pos, idx in enumerate(entity_para_keep_idx):
                            data_id = eval_batch['id'][idx].item()
                            id_to_stats[data_id]['ent_acc'] = output_metrics['binary_correct'][pos]
                if 'relation_paraphrase' in data_names and args.eval_paraphrase_types:
                    output_metrics = metrics.get_metrics_on_named_data(args, main_grad_req, eval_model, tokenizer, eval_batch, 
                                                                        data_name='relation_paraphrase', keep_idx=keep_idx, reference_preds=main_outputs['preds'])
                    epoch_stats['relation_paraphrase_acc_sum'] += output_metrics['acc_sum']
                    epoch_stats['relation_paraphrase_eq_sum'] += output_metrics['ref_acc_sum']
                    epoch_stats['n_relation_paraphrases'] += output_metrics['n_points']
                    for pos, idx in enumerate(keep_idx):
                        data_id = eval_batch['id'][idx].item()
                        id_to_stats[data_id]['rel_acc'] = output_metrics['binary_correct'][pos]
            
            # begin UPDATE MODEL case. this is simple if num_successive_updates=1. eval updated model after every update. otherwise, eval updated model after k successive updates (including across batches!)
            elif is_update_epoch:
                # pick update points for current batch. update wrong points only if args.update_eval_truthfully or if args.update_all_points set to false
                current_where_update = np.arange(batch_size) if _update_all_points else np.argwhere(1-main_binary_correct).reshape(-1)
                
                # combine last batch if needed. get update_blocks. prev_* vars made at end of for batch loop
                if eval_updated_model_counter > 0:          
                    current_ids = batch['id'].tolist()
                    get_these_ids = running_batch_ids + current_ids
                    eval_batch = data_loader.dataset.retrieve_ids(get_these_ids)
                    where_update = current_where_update + len(running_batch_ids) # shift new update idx over by length of running eval data
                    # if got cut off short of num_successive_updates in last batch, need the first update block to bring us up to exactly num_successive updates
                    num_short_by = args.num_successive_updates - eval_updated_model_counter
                    # check that we have enough data to complete the block
                    if len(where_update) >= num_short_by:
                        first_block_idx = where_update[:num_short_by]
                        remaining_idx = where_update[num_short_by:]
                        if len(remaining_idx) > 0:
                            remaining_update_blocks = utils.custom_chunk_array(remaining_idx, args.num_successive_updates)
                        else:
                            remaining_update_blocks = []
                        update_blocks = [first_block_idx] + remaining_update_blocks
                    # if not enough data in the batch to meet above criterion, update on everything in the batch anyway
                    else:
                        update_blocks = [where_update] 
                # otherwise, get update_blocks for current batch alone
                else:
                    where_update = current_where_update
                    update_blocks = utils.custom_chunk_array(where_update, args.num_successive_updates)
                    eval_batch = batch
                    running_updated_ids = []
                
                # get subset of other validation data. exclude data-to-be-updated from this validation data, unless writing graph to file (which needs data pairs considered)
                exclude_ids = eval_batch['id'][where_update].tolist() if not args.write_graph_to_file else []
                if args.dataset == 'Wikidata5m' and not args.write_graph_to_file:  # for Wikidata5m, also add independent statement idx as exclude_idx if building a graph 
                    exclude_ids.extend([item['independent_point_id'] for idx, item in enumerate(eval_batch['text_data']) if idx in where_update])
                random_subset, sample_ids = utils.get_random_subset(data_loader, size=args.num_random_other, exclude_ids=exclude_ids, batch_size=args.test_batch_size,
                                                                    data_sample_counts = data_id_and_sampled_times)
                
                # get PRE-UPDATE PREDICTIONS on other data
                if args.eval_before_cons or args.write_statistics and 'paraphrases' in data_names:
                    eval_model = model.model if args.use_learned_optimizer else model
                    paraphrase_metrics = metrics.get_metrics_on_named_data(args, main_grad_req, eval_model, tokenizer, batch, 
                                                                        data_name='paraphrases', reference_preds=main_outputs['preds'])
                    where_not_none = np.argwhere([paraphrases is not None for paraphrases in batch['paraphrases']]).reshape(-1)
                    for pos, idx in enumerate(where_not_none):
                        data_id = batch['id'][idx].item()
                        if paraphrase_metrics['point_level_par_eqs'] is not None:
                            id_to_stats[data_id]['before_par_eq'] = ' '.join([str(1*item) for item in paraphrase_metrics['point_level_par_eqs'][pos]])
                            id_to_stats[data_id]['before_par_eq_mean'] = np.mean(paraphrase_metrics['point_level_par_eqs'][pos])
                        else:
                            id_to_stats[data_id]['before_par_eq'] = id_to_stats[data_id]['before_par_eq_mean'] = np.nan                        
                        id_to_stats[data_id]['before_cons'] = paraphrase_metrics['point_level_cons'][pos]
                if args.eval_before_dep_acc or args.write_statistics and 'dependent_proposition' in data_names and \
                (args.leapofthought_main != 'hypothesis' if args.dataset == 'LeapOfThought' else True):
                    eval_model = model.model if args.use_learned_optimizer else model
                    output_metrics = metrics.get_metrics_on_named_data(args, main_grad_req, eval_model, tokenizer, batch, 
                                                                        data_name='dependent_proposition')
                    for idx in range(batch_size):
                        data_id = batch['id'][idx].item()
                        id_to_stats[data_id]['before_dep_acc'] = output_metrics['orig_binary_correct'][idx]
                if args.write_statistics and 'independent_proposition' in data_names:
                    eval_model = model.model if args.use_learned_optimizer else model
                    output_metrics = metrics.get_metrics_on_named_data(args, main_grad_req, eval_model, tokenizer, batch, 
                                                                        data_name='independent_proposition')
                    for idx in range(batch_size):
                        data_id = batch['id'][idx].item()
                        id_to_stats[data_id]['before_ind_acc'] = output_metrics['binary_correct'][idx]

                        
                # BEGIN UPDATE LOOP
                for update_block in update_blocks:
                    for update_idx in update_block: # convert back to numpy so update_idx is int
                        # main_kwargs here needs to be re-created with the eval_batch
                        main_kwargs = {k.replace('main_', "") : v for k,v in eval_batch.items() if any([name in k for name in ['main','paraphrases','orig_labels','dependent']]) or k=='id'}
                        utils.move_kwargs_to_gpu(main_kwargs)
                        # pick out single point
                        single_point = utils.slice_kwargs(main_kwargs, update_idx)
                        updated_id = single_point['id'].item()
                        updated_ids.append(updated_id)
                        # update model
                        with main_grad_req and torch.cuda.amp.autocast(enabled=args.fp16):
                            # split update into two cases. 
                            # (1) using a pytorch optimizer, use update_model. 
                            if not args.use_learned_optimizer:
                                update_outputs, update_successful, update_correct, updated_model = update_model(args, updated_model, single_point, tokenizer)
                            # (2) using a learned optimizer, apply the ModelWithLearnedOptimizer forward pass
                            elif args.use_learned_optimizer:
                                update_outputs = model(is_eval=True, return_new_model=True, update_steps=args.update_steps, **single_point)
                                update_successful = metrics.compute_acc_sum(args.probing_style, update_outputs['preds'], single_point['labels'], tokenizer)
                                update_correct = metrics.compute_acc_sum(args.probing_style, update_outputs['preds'], single_point['orig_labels'], tokenizer)
                                updated_model = update_outputs.updated_model
                        eval_updated_model_counter += 1 
                        running_updated_ids.append(updated_id)
                
                        # COMPUTE METRICS FOR UPDATED MODEL
                        # will eval every num_successive_updates points (this param defaults to 1)
                        if eval_updated_model_counter % args.num_successive_updates == 0: 

                            # set eval_model and points to evaluate on
                            eval_model = updated_model
                            eval_ids = eval_batch['id'].tolist()
                            running_updated_idx = [eval_ids.index(id) for id in running_updated_ids]
                            keep_idx = np.array(running_updated_idx)
                            keep_ids = running_updated_ids
                            all_keep_ids.extend(running_updated_ids)
                            
                            # FIRST get post-update predictions for all points
                            all_labels = []
                            all_after_preds = []
                            for other_batch in random_subset:
                                other_kwargs = {k.replace('main_', "") : v for k,v in other_batch.items() if 'main' in k}
                                other_batch_size = other_kwargs['input_ids'].size(0)
                                other_batch_orig_labels = other_batch['orig_labels']
                                all_labels.extend(other_batch_orig_labels.tolist())
                                utils.move_kwargs_to_gpu(other_kwargs)
                                with torch.no_grad() and torch.cuda.amp.autocast(enabled=args.fp16):
                                    outputs = updated_model(is_eval=True, **other_kwargs)
                                new_preds = outputs['preds']
                                epoch_stats['after_acc_sum'] += metrics.compute_acc_sum(args.probing_style, new_preds, other_batch_orig_labels, tokenizer)
                                all_after_preds.extend(new_preds.tolist())
                                epoch_stats['n_random_other_points'] += len(new_preds)
                            # remove updated point idx from all_before_preds and compute statistics for differences in predictions
                            before_preds = all_before_preds[sample_ids]
                            _, binary_retained = metrics.compute_acc_sum(args.probing_style, before_preds, all_after_preds, tokenizer, return_where_correct=True)
                            if args.write_graph_to_file or args.write_statistics:
                                retain_sum = np.sum(binary_retained) + (0 if not args.write_graph_to_file else update_successful) # do not count if the updated point was flipped, hence add 0/1 update_successful
                                epoch_stats['retain_sum'] += retain_sum 
                                where_flipped = np.argwhere(1-binary_retained).reshape(-1)
                                id_to_flipped_id_dict[updated_id] = sorted(np.setdiff1d(sample_ids[where_flipped], updated_id)) # don't include own id in a point's flipped_id list
                            # update id_to_stats
                            _, after_binary_correct = metrics.compute_acc_sum(args.probing_style, all_after_preds, all_labels, tokenizer, return_where_correct=True)                            
                            for idx, data_id in enumerate(sample_ids):
                                if 'after_acc' not in id_to_stats[data_id]:
                                    id_to_stats[data_id]['after_acc'] = str(1*after_binary_correct[idx])
                                else:
                                    id_to_stats[data_id]['after_acc'] += ' ' + str(1*after_binary_correct[idx])
                                    id_to_stats[data_id]['after_acc_mean'] = np.mean([float(x) for x in id_to_stats[data_id]['after_acc'].split()])
                                if 'after_ret' not in id_to_stats[data_id]:
                                    id_to_stats[data_id]['after_ret'] = str(1*binary_retained[idx])
                                else:
                                    id_to_stats[data_id]['after_ret'] += ' ' + str(1*binary_retained[idx])
                                    id_to_stats[data_id]['after_ret_mean'] = np.mean([float(x) for x in id_to_stats[data_id]['after_ret'].split()])
                            # put other_ids and their correctness with the most recent updated id, for bootstrap purposes later
                            id_to_stats[updated_id]['oth_acc'] = ' '.join([f"{other_id}-{int(1*correct)}" for other_id, correct in zip(sample_ids, after_binary_correct)])
                            if args.write_statistics:
                                id_to_stats[updated_id]['oth_ret'] = ' '.join([f"{other_id}-{int(1*correct)}" for other_id, correct in zip(sample_ids, binary_retained)])
                            # save updated ids in the block
                            id_to_stats[updated_id]['block_idx'] = ' '.join([f"{str(updated_id)}" for updated_id in running_updated_ids])
                            
                            # update success on main point. must be recomputed if num_successive_updates > 1
                            if args.num_successive_updates > 1:
                                output_metrics = metrics.get_metrics_on_named_data(args, main_grad_req, eval_model, tokenizer, eval_batch, 
                                                                                    data_name='main', keep_idx=keep_idx)
                                epoch_stats['succ_updates_sum'] += output_metrics['acc_sum']
                                epoch_stats['n_updated'] += output_metrics['n_points']
                                update_outputs = output_metrics['model_outputs'] # OVERWRITE UPDATE OUTPUTS RIGHT HERE, so that update_outputs['preds'] matches len of keep_idx
                                all_update_correct.extend(output_metrics['orig_binary_correct'].tolist())
                                for pos, idx in enumerate(keep_idx):
                                    data_id = eval_batch['id'][idx].item()
                                    id_to_updated_pred[updated_id] = output_metrics['preds'][pos]
                                    id_to_stats[data_id]['upd_suc'] = output_metrics['binary_correct'][pos]
                                    id_to_stats[data_id]['upd_acc'] = output_metrics['orig_binary_correct'][pos]
                            else:
                                epoch_stats['succ_updates_sum'] += update_successful
                                id_to_updated_pred[updated_id] = update_outputs['preds'].item()
                                all_update_correct.append(update_correct)
                                id_to_stats[updated_id]['upd_suc'] = update_successful 
                                id_to_stats[updated_id]['upd_acc'] = update_correct
                                epoch_stats['n_updated'] += 1
                            # paraphrase metrics
                            if args.eval_consistency and 'paraphrases' in data_names:
                                # only compute this if the requested keep_idx have associated paraphrases
                                where_not_none = np.argwhere([paraphrases is not None for paraphrases in eval_batch['paraphrases']]).reshape(-1)
                                where_eligible = np.intersect1d(where_not_none, keep_idx)
                                paraphrase_metrics = metrics.get_metrics_on_named_data(args, main_grad_req, eval_model, tokenizer, eval_batch, 
                                                                                    data_name='paraphrases', keep_idx=keep_idx, reference_preds=update_outputs['preds'])
                                epoch_stats['par_acc_sum'] += paraphrase_metrics['par_acc_sum']
                                epoch_stats['par_eq_sum'] += paraphrase_metrics['par_eq_sum']
                                epoch_stats['n_paraphrases'] += paraphrase_metrics['n_paraphrases']
                                epoch_stats['consistent_sum'] += paraphrase_metrics['n_consistent']
                                epoch_stats['n_paraphrase_pairs'] += paraphrase_metrics['n_paraphrase_pairs']
                                # add cons scores to id_to_stats. in this case, it's the position in keep_idx
                                for pos, idx in enumerate(where_eligible):
                                    data_id = eval_batch['id'][idx].item()
                                    if paraphrase_metrics['point_level_par_eqs'] is not None:
                                        id_to_stats[data_id]['par_eq'] = ' '.join([str(1*item) for item in paraphrase_metrics['point_level_par_eqs'][pos]])
                                        id_to_stats[data_id]['par_eq_mean'] = np.mean(paraphrase_metrics['point_level_par_eqs'][pos])
                                    else:
                                        id_to_stats[data_id]['par_eq_mean'] = id_to_stats[data_id]['par_eq'] = np.nan
                                    id_to_stats[data_id]['cons'] = paraphrase_metrics['point_level_cons'][pos]
                                # add par_eqs to last updated id for bootstrap
                                if paraphrase_metrics['point_level_par_eqs'] is not None:
                                    id_to_stats[updated_id]['oth_par_eq'] = ' '.join([f"{other_id}:{par_id}-{str(1*par_eq)}" for other_id, par_eqs in zip(sample_ids, paraphrase_metrics['point_level_par_eqs']) for par_id, par_eq in enumerate(par_eqs)])
                                    id_to_stats[updated_id]['oth_par_eq_mean'] = ' '.join([f"{other_id}-{np.mean(par_eqs)}" for other_id, par_eqs in zip(sample_ids, paraphrase_metrics['point_level_par_eqs'])])
                                else:
                                    id_to_stats[updated_id]['oth_par_eq'] = id_to_stats[updated_id]['oth_par_eq_mean'] = np.nan
                            # eval: dependent propositions, independent propositions, entity paraphrases, and relation paraphrases
                            if 'dependent_proposition' in data_names and \
                            (args.leapofthought_main != 'hypothesis' if args.dataset == 'LeapOfThought' else True):
                                output_metrics = metrics.get_metrics_on_named_data(args, main_grad_req, eval_model, tokenizer, eval_batch, 
                                                                                    data_name='dependent_proposition', keep_idx=keep_idx, reference_preds=update_outputs['preds'])
                                epoch_stats['dependent_acc_sum'] += output_metrics['orig_acc_sum']
                                epoch_stats['dependent_eq_sum'] += output_metrics['ref_acc_sum']
                                epoch_stats['n_dependent_points'] += output_metrics['n_points']
                                all_dependent_correct.extend(output_metrics['orig_binary_correct'].tolist())
                                for pos, idx in enumerate(keep_idx):
                                    data_id = eval_batch['id'][idx].item()
                                    id_to_stats[data_id]['dep_eq'] = output_metrics['ref_binary_correct'][pos]
                                    id_to_stats[data_id]['dep_acc'] = output_metrics['orig_binary_correct'][pos]
                                id_to_stats[updated_id]['oth_dep_eq'] = ' '.join([str(1*item) for item in output_metrics['ref_binary_correct']])
                                id_to_stats[updated_id]['oth_dep_acc'] = ' '.join([str(1*item) for item in output_metrics['orig_binary_correct']])
                                # when updating, also get acc on other dep data besides update point
                                odp_random_subset, odp_sample_ids = utils.get_random_subset(data_loader, size=args.num_random_other, exclude_ids=exclude_ids, batch_size=args.test_batch_size,
                                                                                            data_sample_counts=odp_data_id_and_sampled_times)
                                all_odp_accs = []
                                for odp_batch in odp_random_subset:
                                    output_metrics = metrics.get_metrics_on_named_data(args, main_grad_req, eval_model, tokenizer, odp_batch, 
                                                                                        data_name='dependent_proposition')
                                    epoch_stats['other_dependent_acc_sum'] += output_metrics['orig_acc_sum']
                                    epoch_stats['n_other_dependent_points'] += output_metrics['n_points']
                                    odp_batch_ids = odp_batch['id'].cpu().numpy()
                                    if all_odp_preds is not None:
                                        before_odp_preds = all_odp_preds[odp_batch_ids]
                                        after_odp_preds = output_metrics['preds']
                                        _, odp_binary_retained = metrics.compute_acc_sum(args.probing_style, before_odp_preds, after_odp_preds, tokenizer, return_where_correct=True)                                
                                    for pos, idx in enumerate(range(len(odp_batch_ids))):
                                        data_id = odp_batch['id'][idx].item()
                                        if 'after_odp_acc' not in id_to_stats[data_id]:
                                            id_to_stats[data_id]['after_odp_acc'] = str(1*output_metrics['orig_binary_correct'][pos])
                                        else:
                                            id_to_stats[data_id]['after_odp_acc'] += ' ' + str(1*output_metrics['orig_binary_correct'][pos])
                                            id_to_stats[data_id]['after_odp_acc_mean'] = np.mean([float(x) for x in id_to_stats[data_id]['after_odp_acc'].split()])
                                        if all_odp_preds is not None:
                                            if 'after_odp_ret' not in id_to_stats[data_id]:
                                                id_to_stats[data_id]['after_odp_ret'] = str(1*odp_binary_retained[idx])
                                            else:
                                                id_to_stats[data_id]['after_odp_ret'] += ' ' + str(1*odp_binary_retained[idx])
                                                id_to_stats[data_id]['after_odp_ret_mean'] = np.mean([float(x) for x in id_to_stats[data_id]['after_odp_ret'].split()])
                                    all_odp_accs.extend(output_metrics['orig_binary_correct'])
                                id_to_stats[updated_id]['oth_odp_acc'] = ' '.join([f"{other_id}-{str(1*correct)}" for other_id, correct in zip(odp_sample_ids, all_odp_accs)])
                            if 'independent_proposition' in data_names:
                                output_metrics = metrics.get_metrics_on_named_data(args, main_grad_req, eval_model, tokenizer, eval_batch, 
                                                                                    data_name='independent_proposition', keep_idx=keep_idx, reference_preds=all_independent_preds[keep_ids])
                                epoch_stats['independent_acc_sum'] += output_metrics['acc_sum']
                                epoch_stats['independent_ret_sum'] += output_metrics['ref_acc_sum']
                                epoch_stats['n_independent_points'] += output_metrics['n_points']
                                all_independent_correct.extend(output_metrics['binary_correct'].tolist())
                                for pos, idx in enumerate(keep_idx):
                                    data_id = eval_batch['id'][idx].item()
                                    id_to_stats[data_id]['ind_acc'] = output_metrics['binary_correct'][pos]
                                    id_to_stats[data_id]['ind_ret'] = output_metrics['ref_binary_correct'][pos]
                                id_to_stats[updated_id]['oth_ind_acc'] = ' '.join([f"{other_id}-{str(1*correct)}" for other_id, correct in zip(keep_ids, output_metrics['binary_correct'])])
                                id_to_stats[updated_id]['oth_ind_ret'] = ' '.join([f"{other_id}-{str(1*correct)}" for other_id, correct in zip(keep_ids, output_metrics['ref_binary_correct'])])
                            if 'entity_paraphrase' in data_names and args.eval_paraphrase_types:
                                where_not_none = np.argwhere([item['entity_paraphrase'] is not None for item in eval_batch['text_data']]).reshape(-1)
                                entity_para_keep_idx = np.intersect1d(where_not_none, keep_idx)
                                eligible_paraphrases_exist = len(entity_para_keep_idx) > 0
                                if eligible_paraphrases_exist:
                                    output_metrics = metrics.get_metrics_on_named_data(args, main_grad_req, eval_model, tokenizer, eval_batch, 
                                                                                        data_name='entity_paraphrase', keep_idx=entity_para_keep_idx, reference_preds=update_outputs['preds'])
                                    epoch_stats['entity_paraphrase_acc_sum'] += output_metrics['acc_sum']
                                    epoch_stats['entity_paraphrase_eq_sum'] += output_metrics['ref_acc_sum']
                                    epoch_stats['n_entity_paraphrases'] += output_metrics['n_points']
                                    for pos, idx in enumerate(entity_para_keep_idx):
                                        data_id = eval_batch['id'][idx].item()
                                        id_to_stats[data_id]['ent_acc'] = output_metrics['binary_correct'][pos]
                                    id_to_stats[updated_id]['oth_par_ent_eq'] = ' '.join([str(1*item) for item in output_metrics['ref_binary_correct']])
                            if 'relation_paraphrase' in data_names and args.eval_paraphrase_types:
                                output_metrics = metrics.get_metrics_on_named_data(args, main_grad_req, eval_model, tokenizer, eval_batch, 
                                                                                    data_name='relation_paraphrase', keep_idx=keep_idx, reference_preds=update_outputs['preds'])
                                epoch_stats['relation_paraphrase_acc_sum'] += output_metrics['acc_sum']
                                epoch_stats['relation_paraphrase_eq_sum'] += output_metrics['ref_acc_sum']
                                epoch_stats['n_relation_paraphrases'] += output_metrics['n_points']
                                for pos, idx in enumerate(keep_idx):
                                    data_id = eval_batch['id'][idx].item()
                                    id_to_stats[data_id]['rel_acc'] = output_metrics['binary_correct'][pos]
                                id_to_stats[updated_id]['oth_par_rel_eq'] = ' '.join([str(1*item) for item in output_metrics['ref_binary_correct']])

                            # clean up counters and running lists
                            eval_updated_model_counter = 0
                            running_batch_ids = []
                            running_updated_ids = []

                            # if eval was triggered for off-the-shelf optimizer, time to freshly copy the base model
                            if args.update_beliefs:
                                updated_model = deepcopy(model)

                # check for if there are any updated points remaining in this batch that will have to be evaluated with the next batch
                # need to accumulate these variables across batches, in case it takes many batches to get enough data
                if eval_updated_model_counter > 0:
                    running_batch_ids.extend(batch['id'].tolist())

        # record stats in stats_dict. first, do acc, which requires special calculation when doing update_beliefs
        n = SimpleNamespace(**epoch_stats)
        stats_dict[f'{split_name}_acc'] = 100 * n.acc_sum / n.n_data_points
        if n.n_updated > 0:                            stats_dict[f'{split_name}_upd_suc'] = 100 * n.succ_updates_sum / n.n_updated
        if n.n_paraphrase_pairs > 0:                   stats_dict[f'{split_name}_cons']    = 100 * n.consistent_sum / n.n_paraphrase_pairs
        if n.n_paraphrases > 0:                        stats_dict[f'{split_name}_par_acc'] = 100 * n.par_acc_sum / n.n_paraphrases
        if n.n_paraphrases > 0:                        stats_dict[f'{split_name}_par_eq']  = 100 * n.par_eq_sum / n.n_paraphrases
        if n.n_random_other_points > 0:                stats_dict[f'{split_name}_oth_ret'] = 100 * n.retain_sum / n.n_random_other_points
        if n.before_acc_sum > 0:                       stats_dict[f'{split_name}_crp_rte'] = 100 * n.corrupted_sum / n.before_acc_sum
        if n.n_dependent_points > 0:                   stats_dict[f'{split_name}_dep_acc'] = 100 * n.dependent_acc_sum / n.n_dependent_points 
        if n.n_other_dependent_points > 0:             stats_dict[f'{split_name}_odp_acc'] = 100 * n.other_dependent_acc_sum / n.n_other_dependent_points 
        if n.n_independent_points > 0:                 stats_dict[f'{split_name}_ind_acc'] = 100 * n.independent_acc_sum / n.n_independent_points
        if n.n_independent_points > 0:                 stats_dict[f'{split_name}_ind_ret'] = 100 * n.independent_ret_sum / n.n_independent_points
        if n.n_entity_paraphrases > 0:                 stats_dict[f'{split_name}_ent_acc'] = 100 * n.entity_paraphrase_acc_sum / n.n_entity_paraphrases
        if n.n_entity_paraphrases > 0:                 stats_dict[f'{split_name}_ent_ret'] = 100 * n.entity_paraphrase_eq_sum / n.n_entity_paraphrases
        if n.n_relation_paraphrases > 0:               stats_dict[f'{split_name}_rel_acc'] = 100 * n.relation_paraphrase_acc_sum / n.n_relation_paraphrases
        if n.n_relation_paraphrases > 0:               stats_dict[f'{split_name}_rel_ret'] = 100 * n.relation_paraphrase_eq_sum / n.n_relation_paraphrases
        if is_update_epoch  and n.n_random_other_points * n.before_acc_sum > 0:  
            stats_dict[f'{split_name}_per_det'] = 100 * (1 - (n.after_acc_sum/n.n_random_other_points) / (n.before_acc_sum/n.n_random_other_points))
        # get change in performance without computing before-update accuracies
        if n.n_random_other_points > 0:
            if args.update_eval_truthfully or not args.update_all_points:
                stats_dict[f'{split_name}_per_chg'] = 100 * n.after_acc_sum / n.n_random_other_points - 100 * (n.acc_sum / (n.n_data_points - eval_updated_model_counter)) # (this is exactly correct)
            else:
                stats_dict[f'{split_name}_per_chg'] = 100 * n.after_acc_sum / n.n_random_other_points - stats_dict[f'{split_name}_acc'] # very slight approximation
        # get change in independent performance without computing before-update accuracies -- only works for our unshuffled split of wikidata5m
        if n.n_independent_points > 0 and (args.use_learned_optimizer or is_update_epoch) and (args.update_eval_truthfully or not args.update_all_points):
            assert args.dataset == 'Wikidata5m', "this only works for our unshuffled eval splits of wikidata5m"
            assert len(all_main_correct) % 2 == 0, "eval size must be even"
            _all_independent_correct = utils.flip_array_pairwise(np.array(all_main_correct)) # switch every pair of points with one another, in order
            all_independent_correct_where_main_wrong = _all_independent_correct[np.argwhere(1-np.array(all_main_correct)).reshape(-1)]
            stats_dict[f'{split_name}_ind_chg'] = 100 * n.independent_acc_sum / n.n_independent_points - 100 * np.mean(all_independent_correct_where_main_wrong)
            
        # select best epoch based on these statistics when training learned optimizer. group by area, mean within area then mean overall
        if is_update_epoch and n.n_random_other_points > 0: 
            # upd succ and performance change
            upd_succ_stats = []
            upd_succ_stats.append(stats_dict[f'{split_name}_upd_suc'])
            if f'{split_name}_dep_acc' in stats_dict:
                upd_succ_stats.append(stats_dict[f'{split_name}_dep_acc'])
            if f'{split_name}_par_eq' in stats_dict:
                upd_succ_stats.append(stats_dict[f'{split_name}_par_eq'])
            # acc for other data
            acc_stats = []
            acc_stats.append(stats_dict[f'{split_name}_per_chg'])
            if f'{split_name}_dep_acc' in stats_dict:
                acc_stats.append(stats_dict[f'{split_name}_dep_acc'])
            if f'{split_name}_odp_acc' in stats_dict:
                acc_stats.append(stats_dict[f'{split_name}_odp_acc'])
            if f'{split_name}_ind_acc' in stats_dict:
                acc_stats.append(stats_dict[f'{split_name}_ind_acc'])
            # ret on independent data
            ret_stats = []
            if f'{split_name}_ind_ret' in stats_dict:
                ret_stats.append(stats_dict[f'{split_name}_ind_ret'])
            have_stats = [stats for stats in [upd_succ_stats, acc_stats, ret_stats] if len(stats) > 0]
            stats_dict[f'{split_name}_sel_for'] = np.mean([np.mean(stats) for stats in have_stats]).item()
        else:
            stats_dict[f'{split_name}_sel_for'] = stats_dict[f'{split_name}_acc']

        # BREAK EARLY IF REQUESTED -- used if only want to use a subset of dev points while training, due to slow seq2seq behavior e.g.
        if break_after_n_points > 0:
            if epoch_stats['n_data_points'] >= break_after_n_points: break

    # print last batch
    if args.print and split_name.lower() == 'dev':
        print("\n" + "-"*20 + f"\nPrinting {split_name.lower()} data:")
        print_data_names = [name for name in data_names if name not in ['main']]
        for i in range(min(args.num_print, batch_size)):
            text_data = eval_batch['text_data']
            print(f" Model input {i}: {text_data[i]['main']}")
            print(f"  Pred        : {preds[i]}")
            if is_update_epoch:
                print(f"  Updated Pred: {update_outputs['preds'][i]}")
            print(f"  Label (em)  : {labels[i]}")
            for data_name in print_data_names:
                print(f"  {data_name.capitalize():12s}: {text_data[i][data_name]}")
        print("-"*20 + '\n')
    # print dependent prop statistics
    if not is_train_epoch and epoch_stats['n_dependent_points'] > 0:
        all_knowledge_correct = all_main_correct if not is_update_epoch else all_update_correct
        if len(all_knowledge_correct) > len(all_dependent_correct): # happens when eval_subset != all and not using learned optimizer
            all_knowledge_correct = np.array(all_knowledge_correct)[np.array(all_keep_ids)]
        print(f" Dependent propositions statistics:")
        metrics.print_dependency_metrics(knowledge1=all_knowledge_correct, consequent1=all_dependent_correct)
        print(f" Dependent propositions statistics (reversed condition, to check for contrapositive):")
        metrics.print_dependency_metrics(knowledge1=all_dependent_correct, consequent1=all_knowledge_correct)
    if not is_train_epoch and epoch_stats['n_independent_points'] > 0:
        print(f" Independent propositions statistics:")
        all_knowledge_correct = all_main_correct if not is_update_epoch else all_update_correct
        if len(all_knowledge_correct) > len(all_independent_correct): # happens when eval_subset != all and not using learned optimizer
            all_knowledge_correct = np.array(all_knowledge_correct)[np.array(all_keep_ids)]
        metrics.print_dependency_metrics(knowledge1=all_knowledge_correct, consequent1=all_independent_correct)
        print(f" Control condition, randomly paired points:")
        np.random.shuffle(all_knowledge_correct)
        metrics.print_dependency_metrics(knowledge1=all_knowledge_correct, consequent1=all_independent_correct)

    # end-of-epoch model parameter reset/annealing
    if args.implementation == 'de_cao' and args.use_learned_optimizer and args.do_train and split_name=='dev' and epoch > 0: # hparam annealing for learned optimizer, at end of dev epochs during training
        model.reduce_constraint_margins(stats_dict['dev_upd_suc'])
        print(f"kl alpha: {model.alpha_kl:.6f}")
        print(f"kl margin: {model.margin_kl:.6f}")
    if args.use_learned_optimizer and args.num_successive_updates > 1 and epoch != -1: # i.e. not pre-eval
        model.reset_successive_update_vars()

    if write_preds_to_file: 
        # flatten a list of tensors/lists to a list
        all_main_preds = [item.item() if type(item) is torch.Tensor else item for iterable in all_main_preds for item in list(iterable)]
        if 'alt_preds' in main_outputs and args.beam_search_alt_labels:
            all_alt_preds = np.array([item for iterable in all_alt_preds for item in list(iterable)])
        data_path = getattr(args, f"{split_name}_path")
        new_data = []
        # all datapoints get added back into the original file the data loaded from
        with jsonlines.open(data_path) as file:
            point_counter = 0 # using a counter here because of the 'continue' used in a special case with LeapOfThought data
            for data_id, datapoint in enumerate(file):
                if data_id >= epoch_stats['n_data_points']: # meaning all_main_preds[point_counter] will lead to an index error
                    new_data.append(datapoint)
                    continue
                # make pred name
                if args.dataset != 'LeapOfThought':
                    pred_name = f'{args.base_experiment_name}_pred'
                elif args.dataset == 'LeapOfThought':
                    pred_name = f'{args.base_experiment_name}_{args.leapofthought_main}_pred'
                    if split_name=='train' and datapoint['context'] == '': # skip leapofthought train points if they are knowledge only / do not have a hypothesis
                        new_data.append(datapoint)
                        continue
                # add main point pred
                datapoint[pred_name] = all_main_preds[point_counter]
                # add alt preds for seq2seq tasks
                if 'alt_preds' in main_outputs and args.beam_search_alt_labels:
                    slice_idx = range((args.beam_search_size-1)*point_counter, (args.beam_search_size-1)*(point_counter+1))
                    datapoint[f'{args.base_experiment_name}_alt_preds'] = all_alt_preds[slice_idx].tolist()
                # add flipped points for forming belief graph
                if is_update_epoch:
                    datapoint[f'{args.experiment_name}_flips'] = [int(x) for x in id_to_flipped_id_dict[data_id]] # casting to avoid dtype error with JSON
                    datapoint[f'{args.experiment_name}_update_pred'] = id_to_updated_pred[data_id]
                # accumulate data point and counter
                new_data.append(datapoint)
                point_counter += 1
        with jsonlines.open(data_path, mode='w') as file:
            for point_num, point in enumerate(new_data):
                file.write(point)

    # write data-point level statistics and summary statistics to .csv's in ./results_sheets/, only for final eval!
    if args.write_statistics and split_name != 'train' and epoch==-1 and not pre_eval:
        result_sheet_name = utils.get_result_sheet_name(args, args.experiment_name, is_update_epoch, split_name)
        data_save_path = os.path.join('result_sheets/', f"data_stats_{result_sheet_name}")
        summary_save_path = os.path.join('result_sheets/',  f"summary_stats_{result_sheet_name}")
        # add before_acc to id dict, plus id itself
        for idx in range(len(all_main_correct)):
            acc_key = 'before_acc' if is_update_epoch else 'acc'
            id_to_stats[idx][acc_key] = int(all_main_correct[idx])
            id_to_stats[idx]['id'] = idx
            if args.dataset in ['FEVER', 'LeapOfThought']:
                id_to_stats[idx]['label'] = int(all_data_labels[idx])
        # save data stats
        data = pd.DataFrame.from_dict(id_to_stats, "index").sort_index() 
        data['seed'] = args.seed
        utils.add_df_cols_from_args(data, args, ["dataset", "optimizer", "lr"])
        if is_update_epoch:
            if args.use_learned_optimizer: 
                data['k_train'] = args.learned_opt_steps
            data['k_test'] = args.update_steps
            data['r_train'] = args.learned_successive_updates if args.learned_successive_updates > 0 else args.num_successive_updates
            data['r_test'] = args.num_successive_updates
            data['obj'] = utils.args_to_obj_name(args)
        data.to_csv(data_save_path, index=False)
        # save summary stats
        data = pd.DataFrame.from_dict({k : [v] for k,v in stats_dict.items()}, "columns")
        data['seed'] = args.seed
        data['exp_name'] = args.experiment_name
        utils.add_df_cols_from_args(data, args, ["dataset", "optimizer", "lr"])            
        if is_update_epoch:
            if args.use_learned_optimizer: 
                data['k_train'] = args.learned_opt_steps
            data['k_test'] = args.update_steps
            data['r_train'] = args.learned_successive_updates if args.learned_successive_updates > 0 else args.num_successive_updates
            data['r_test'] = args.num_successive_updates
            data['obj'] = utils.args_to_obj_name(args)
        data.to_csv(summary_save_path, index=False)
            
    return stats_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # misc. & debugging
    parser.add_argument('--gpu', type = int, default = 0, help = 'gpu id to use')
    parser.add_argument("--seed", default=0, type=int, help='')
    parser.add_argument("--fp16", default = False, type=str2bool, help = 'use fp16 as described here https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html')
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--small_data", '-s', action='store_true')
    parser.add_argument("--update_small_data", '-us', action='store_true')
    parser.add_argument("--print", default = True, type=str2bool, help = 'flag for printing things helpful for debugging / seeing whats happening')
    parser.add_argument("--num_print", '-np', default = 1, type=int, help = 'number of points to print in train_or_test.py')
    # model + model paths
    parser.add_argument("--model", default='bert-base-cased', type=str, help='name of pretrained model')
    parser.add_argument("--load_model_path", default=None, type=str, help='load the model at this path before training if not None')
    parser.add_argument("--load_opt_path", default=None, type=str, help='load the learned optimizer at this path before training if not None')
    parser.add_argument("--save_model_name", default=None, type=str, help='name for saving the model')
    # paths/directories for data, cache, etc.
    parser.add_argument("--dataset", default='FEVER', type=str, help='')
    parser.add_argument("--data_dir", default='', type=str, help='')
    parser.add_argument("--save_dir", default='', type=str, help='')
    parser.add_argument("--cache_dir", default='', type=str, help='')
    parser.add_argument("--report_dir", default='training_reports', type=str, help='')
    parser.add_argument("--server", default=None, type=str, help='')
    # probing args
    parser.add_argument("--probing_style", default='model', choices=['model', 'cloze', 'seq2seq', 'clm'], help='')
    parser.add_argument("--probe", default='linear', choices=['linear', 'transformer', 'None'], help='')
    # training hyperparams + conditions for a task model and the optimizer
    parser.add_argument("--num_train_points", "-n", default = -1, type=int, help='if set to >0, use this many points for training')
    parser.add_argument("--num_eval_points", "-ne", default = -1, type=int, help='if set to >0, use this many points for training')
    parser.add_argument("--eval_n_points_when_training", '-nt', default=-1, type=int, help='if > 0, only use this many points from dev set during training')
    parser.add_argument("--train_batch_size", default=32, type=int, help='')
    parser.add_argument("--test_batch_size", default=64, type=int, help='')
    parser.add_argument("--num_train_epochs", default=10, type=int, help='')
    parser.add_argument("--max_seq_len", default=200, type=int, help='')
    parser.add_argument("--max_grad_norm", default=1., type=float, help='')
    parser.add_argument("--grad_accumulation_factor", '-gaf', default=1, type=int, help='effective batch size = batch_size * grad_accumulation_factor')
    parser.add_argument("--lr", default=1e-5, type=float, help='')
    parser.add_argument("--weight_decay", default=1e-4, type=float, help='')
    parser.add_argument("--fit_model_to_paraphrases", default = False, type=str2bool, help = '')
    parser.add_argument("--paraphrases_to_unique_points", default = False, type=str2bool, help = 'if data point has e.g. three paraphrases, convert it to three unique data points')
    # training hyperparams for learned optimizer
    parser.add_argument("--update_parameters", default='all', choices=['probe', 'biases', 'ff_neurons', 'all', 'de_cao', 'interior', 'optimizer'], help='')
    parser.add_argument("--optimizer", default='adamw', choices=['sgd', 'adamw', 'rmsprop', 'learned', 'neuron_IG'])
    parser.add_argument("--implementation", default='new', choices=['new', 'de_cao'])
    parser.add_argument("--detach_prev_updates", default = True, type=str2bool, help = '')
    parser.add_argument("--fit_to_wrong_points", default = False, type=str2bool, help = 'fit learned optimizer to wrong points only')
    parser.add_argument("--learned_opt_steps", default=1, type=int, help='number of model update steps to apply learned opt for before doing optimizer.step()')
    parser.add_argument("--fit_opt_to_paraphrases", default = False, type=str2bool, help = '')
    parser.add_argument("--fit_opt_to_dependent_propositions", default = False, type=str2bool, help = '')
    parser.add_argument("--fit_opt_to_independent_propositions", default = False, type=str2bool, help = '')
    parser.add_argument("--divergences", default='none', choices=['kl', 'none'], help='')
    parser.add_argument("--min_corruption", default = False, type=str2bool, help = '')
    parser.add_argument("--lambda_main", default=1., type=float, help='weight on main loss (normalized to sum to 1 with others)')
    parser.add_argument("--lambda_corruption", default=1., type=float, help='weight on * term (normalized to sum to 1 with others)')
    parser.add_argument("--lambda_kl", default=1., type=float, help='weight on * term (normalized to sum to 1 with others)')
    parser.add_argument("--lambda_paraphrase", default=1., type=float, help='weight on * term (normalized to sum to 1 with others)')
    parser.add_argument("--lambda_dependents_updated", default=1., type=float, help='see models/learned_optimizer.py')
    parser.add_argument("--lambda_independents_updated", default=1., type=float, help='see models/learned_optimizer.py')
    parser.add_argument("--fit_to_alt_labels", default = False, type=str2bool, help = 'if true, use alt-labels for every data point to learn optimizer, not only using the true labels')
    parser.add_argument("--beam_search_alt_labels", default = False, type=str2bool, help = 'if true and seq2seq, get alt labels for correct points as other answers from the model beam search for use in training')
    parser.add_argument("--eval_beam_search_alt_labels", default = False, type=str2bool, help = 'if true and seq2seq, get alt labels for correct points as other answers from the model beam search for use in model eval')
    # eval hyperparams
    parser.add_argument("--num_random_other", default=40, type=int, help='num random other points to check perf degradation on')
    parser.add_argument("--wikidata_para_per_point", default=4, type=int, help='number of paraphrases to create per point for wikidata')
    parser.add_argument("--update_all_points", default = False, type=str2bool, help = 'if false, update only on wrong points')
    parser.add_argument("--beam_search_size", default=5, type=int, help='beam search size for generative tasks')
    parser.add_argument("--update_steps", default=1, type=int, help='use positive int to indicate that num of steps, or 0 to update until successful')
    # control flow + experiment conditions
    parser.add_argument("--do_train", default = False, type=str2bool, help = 'finetunes a model on a task')
    parser.add_argument("--do_eval", default = True, type=str2bool, help = 'finetunes a model on a task')
    parser.add_argument("--use_learned_optimizer", default = False, type=str2bool, help = 'learn an optimizer for the problem')
    parser.add_argument("--load_finetuned_model", default = True, type=str2bool, help = 'set experiment_name to load finetuned model even though --do_train is false')
    parser.add_argument("--load_alt_labels_model", default = False, type=str2bool, help = 'set experiment_name to load alt labels model even though using real labels')
    parser.add_argument("--load_seed", default = -1, type=int, help = 'seed used in base experiment that is being loaded form. used if >= 0')
    parser.add_argument("--orig_trained_parameters", default='all', choices=['probe', 'biases', 'ff_neurons', 'all', 'de_cao'], help='used for loading the right model with --update_beliefs set to true')
    parser.add_argument("--use_nonfinetuned_model", default = False, type=str2bool, help = 'evaluate an non-finetuned model')
    parser.add_argument("--update_beliefs", default = False, type=str2bool, help = 'use update_model function to update model beliefs (no learned opt)')
    parser.add_argument("--learned_successive_updates", default = -1, type=int, help = 'if > 0, used to load model with this hparam during training')
    parser.add_argument("--num_successive_updates", default = 1, type=int, help = 'evaluates model performance after this many serial updates')
    parser.add_argument("--update_eval_truthfully", default = False, type=str2bool, help = 'forces update_all_points to false and fit_to_alt_labels to false on eval splits')
    parser.add_argument("--eval_consistency", default = True, type=str2bool, help = '')
    parser.add_argument("--eval_paraphrase_types", default = False, type=str2bool, help = '')
    parser.add_argument("--eval_before_cons", default = False, type=str2bool, help = '')
    parser.add_argument("--eval_before_dep_acc", default = False, type=str2bool, help = '')
    parser.add_argument("--train_fever_base", default = False, type=str2bool, help = 'train on 83k rather than 93k points (exclude train-opt split)')
    parser.add_argument("--eval_subset", default = 'all', choices=['all','wrong','right'], type=str, help = 'eval cons on these points to compare with update beliefs setting. used in non-update setting')
    parser.add_argument("--paraphrase_labels", default = 'default', choices=['default', 'orig_label', 'new_pred', 'orig_pred'], 
                                                        help = 'which labels to use for paraphrase metrics. default is main label, which is new_label when updating or orig label when not')
    parser.add_argument("--preprocess_data_when_loading", '-p', default = False, type=str2bool, help = '')
    parser.add_argument("--fit_to_dev_data", default = False, type=str2bool, help = 'used with --use_learned_optimizer')
    parser.add_argument("--pre_eval", default = False, type=str2bool, help = '')
    parser.add_argument("--eval_after", default=0, type=int, help='eval using dev data at or after this epoch, not before')
    parser.add_argument("--write_preds_to_file", default = False, type=str2bool, help = '')
    parser.add_argument("--write_graph_to_file", default = False, type=str2bool, help = '')
    parser.add_argument("--leapofthought_main", default = 'hypothesis', choices=['implicit_rule', 'main', 'hypothesis'], help = 'see utils.load_data')
    parser.add_argument("--leapofthought_add_both_for_training", default = False, type=str2bool, help = 'see utils.load_data')
    parser.add_argument("--Wikidata5m_use_synonyms", default = False, type=str2bool, help = 'pick different synonym for entity/relation for every point')
    parser.add_argument("--write_statistics", default = False, type=str2bool, help = 'writes data-point level statistics to file')
    parser.add_argument("--use_dev_not_test", default = True, type=str2bool, help = 'writes data-point level statistics to file')
    parser.add_argument("--overwrite_past_experiment", default = False, type=str2bool, help = 'will overwrite past experiment even if the training report indicates a full past experiment has been run')
    parser.add_argument("--get_experiment_name_only", default = False, type=str2bool, help = 'if true, script sets the experiment name as an environment variable then does nothing else')
    # graph viz arguments
    parser.add_argument("--do_graph_analysis", default = False, type=str2bool, help = '')
    parser.add_argument("--graph_layout", default = 'dot', choices=['neato', 'dot'], help = '')
    
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    
    # parse + experiment checks
    args = parser.parse_args()
    if args.use_learned_optimizer and args.do_train: assert args.update_parameters=='optimizer', "only use learned optimizer parameters in adamw optimizer, not base model"
    if args.use_learned_optimizer and args.do_train: assert args.lr > 1e-5, "use a higher lr for training the optimizer"
    if args.fit_model_to_paraphrases: assert not args.use_learned_optimizer, "use fit_opt_to_paraphrases instead of fit_model_to_paraphrases here"
    if args.do_train and args.use_learned_optimizer: assert args.update_steps >= args.learned_opt_steps, "should use dev steps >= trains steps for opt"
    assert not (args.use_learned_optimizer and args.update_beliefs), "these are mutually exclusive arguments. update_beliefs is for off-the-shelf optimizers"
    if args.fit_to_alt_labels and args.do_train: assert not args.fit_to_wrong_points, "if training and fitting to alt labels, use all data points for training"
    if args.learned_successive_updates > 1 and args.do_train: assert args.grad_accumulation_factor >= args.learned_successive_updates, "grad accum factor must be at least learned_successive_updates"
    if args.use_learned_optimizer and args.num_successive_updates > 1 and args.do_train:
        print("\nNote that train update success is NOT calculated after num_successive_updates, but rather after every single update, so not comparable to dev upd_suc \n")

    # init experiment name, Report, stats_dict, and saving/loading paths
    experiment_name = utils.add_experiment_name_to_args(args) # note this both returns experiment_name and adds it AND base_experiment_name to args
    if args.get_experiment_name_only:
        with open('outputs/tmp_experiment_name.txt', 'w') as file: file.write(experiment_name)
        sys.exit()
    print(f"Starting experiment: {experiment_name} \n")
    report_name = f"report_{experiment_name}.txt"
    report_file = os.path.join(args.report_dir, report_name)
    score_names = args.dataset_config['stat_names']
    report = Report(args, report_file, experiment_name = experiment_name, score_names = score_names, overwrite_existing=args.do_train or args.update_beliefs)
    args.report = report # add report to args
    stats_dict = {name : 0 for name in score_names}
    if args.do_train or args.do_eval:
        before_training_load_path, before_eval_load_path, model_save_path = utils.get_model_save_and_load_paths(args)
        if before_training_load_path is not None: print(f"{'Before train load path: ':25s} {before_training_load_path.replace(args.save_dir, 'args.save_dir')}")
        if model_save_path is not None:           print(f"{'Save path: ':25s} {model_save_path.replace(args.save_dir, 'args.save_dir')}")
        print(f"{'Before eval load path: ':25s} {before_eval_load_path.replace(args.save_dir, 'args.save_dir')}")
        
    # make dirs if do not exist
    for dir in ['result_sheets', args.save_dir, args.cache_dir]:
        if not os.path.exists(dir): 
            os.mkdir(dir)

    # check if experiment already exists
    if not (args.overwrite_past_experiment or 'DEBUG' in experiment_name) and args.write_statistics:
        result_sheet_name = utils.get_result_sheet_name(args, args.experiment_name, args.use_learned_optimizer or args.update_beliefs, 'test')
        summary_save_path = os.path.join('result_sheets/', f"summary_stats_{result_sheet_name}")
        results_written = os.path.exists(summary_save_path)
        if results_written:
            print("Already wrote results for this experiment, and will not run again! This can be overridden with --overwrite_past_experiment")
            sys.exit()

    # GPU + SEED setup
    n_gpu = torch.cuda.device_count()
    args.multi_gpu = multi_gpu = (n_gpu > 1 and args.gpu == -1) # i.e. multiple gpus available and gpu choice not specified
    if multi_gpu: 
        device = torch.device("cuda") if args.gpu == -1 else torch.device(f'cuda:{args.gpu}')
        assert args.train_batch_size % n_gpu == 0, f"Train batch size will need to be allocated equally across {n_gpu} gpus, but {args.train_batch_size} cannot be"
        assert args.test_batch_size % n_gpu == 0, f"Eval batch size will need to be allocated equally across {n_gpu} gpus, but {args.test_batch_size} cannot be"
    else:
        device = torch.device(f"cuda:{args.gpu}")
        torch.cuda.set_device(device)
    args.device = device
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # load model and tokenizer. put model into Probe class
    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir)
    if args.do_train or args.do_eval:
        print("Loading model...")
        probing_style_to_model_class = {'cloze' : AutoModelWithLMHead, 'model' : AutoModel, 'seq2seq' : AutoModelForSeq2SeqLM, 'clm' : AutoModel}
        model_class = probing_style_to_model_class[args.probing_style]
        model = model_class.from_pretrained(args.model, cache_dir=args.cache_dir)
        model = Probe(args, model, tokenizer)
        if args.use_learned_optimizer:
            model = ModelWithLearnedOptimizer(args, model, tokenizer)
        model = model.to(args.device)
        if before_training_load_path is not None:
            print("Loading model from:", before_training_load_path)
            safe_load_base_model(model, state_dict=torch.load(before_training_load_path, map_location=lambda storage, loc: storage))
        if multi_gpu:
            assert False, "DataParallel not yet tested"
            model = torch.nn.DataParallel(model)
            
    # load data, optimizer, scheduler, and scaler
    print("Loading data...", end='\r')
    load_start = time.time()
    train_dataloader, dev_dataloader, test_dataloader = utils.load_data(args, tokenizer)
    print(f"Loading data...took {round((time.time() - load_start) / 60, 2)} minutes", end='\r')
    print(f"\nTrain size: {len(train_dataloader.dataset)} | Dev size: {len(dev_dataloader.dataset)} | Test size: {len(test_dataloader.dataset)} ")
    if args.dataset in ['FEVER', 'Leap-Of-Thought']:
        print(f"Prop. true: {[(dataloader.dataset.split_name.capitalize(), utils.get_prop_true(args, dataloader)) for dataloader in [train_dataloader, dev_dataloader, test_dataloader]]}")
    if args.do_train:
        num_train_optimization_steps = args.num_train_epochs * int(len(train_dataloader.dataset) / args.train_batch_size / args.grad_accumulation_factor)
        print("Num optimizer steps: ", num_train_optimization_steps)
        optimizer, scheduler = load_optimizer_and_scheduler(args, model, num_train_optimization_steps)
        num_opt_params = np.sum([np.prod(params.size()) for i in range(len(optimizer.param_groups)) for params in optimizer.param_groups[i]['params']])
        print(f"Num trainable parameters: {num_opt_params/1e6:.2f}m")
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

    # evaluate model before training 
    if args.pre_eval:
        if not multi_gpu:  eval_model = model if not args.use_learned_optimizer else model.model
        if multi_gpu:      eval_model = model
        print(f"Beginning eval on dev before training...")
        stats_dict = train_or_test(args,
                                stats_dict, 
                                epoch=-1, 
                                model=eval_model, 
                                data_loader=dev_dataloader, 
                                tokenizer=tokenizer,
                                pre_eval=True)
        report.print_epoch_scores(epoch = -1, scores = stats_dict)

    # train model
    best_epoch = 0
    best_score = -1
    start_time=time.time()
    if args.do_train:
        for e in range(1,args.num_train_epochs+1):
            # train and dev
            print(f"Epoch {e}")
            use_data_loaders = [train_dataloader, dev_dataloader] if e >= args.eval_after else [train_dataloader]
            for data_loader in use_data_loaders:
                stats_dict = train_or_test(args,
                                stats_dict, 
                                epoch=e, 
                                model=model, 
                                data_loader=data_loader, # see role of data_loader.dataset.split_name in train_or_test
                                tokenizer=tokenizer,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                scaler=scaler,
                                break_after_n_points = -1 if data_loader.dataset.split_name == 'train' else args.eval_n_points_when_training)
            report.write_epoch_scores(epoch = e, scores = stats_dict)
            report.print_epoch_scores(epoch = e, scores = stats_dict)
            if e >= args.eval_after:
                epoch_score = stats_dict['dev_acc'] if not args.use_learned_optimizer else stats_dict['dev_sel_for']
                if epoch_score > best_score:
                    print(f"\n    New best model with score: {epoch_score:.2f}! Saving model at {model_save_path}\n\n")
                    torch.save(model.state_dict(), model_save_path)
                    best_score, best_epoch = epoch_score, e
                else:
                    print(f"Current epoch score: {epoch_score:.2f} | Best epoch score: {best_score:.2f}")
        time_msg = utils.format_training_time(start=start_time, end=time.time())
        print(time_msg)
        
    # final model evaluation
    if args.do_eval:
        print(f"Beginning final eval for {experiment_name if args.load_opt_path is None else args.load_opt_path}...")
        # get un-shuffled (train) dataloaders if args.write_to_file
        if args.write_preds_to_file or args.write_graph_to_file:
            train_dataloader, dev_dataloader, test_dataloader = utils.load_data(args, tokenizer, shuffle_loaders=False)
        # reload from best checkpoint or load_model. should always be exact match for state_dict
        if not args.use_nonfinetuned_model:
            state_dict = torch.load(before_eval_load_path, map_location=lambda storage, loc: storage) 
            # if 'state_dict' in state_dict: state_dict = state_dict['state_dict'] # convert from pytorch lightning
            # model.load_state_dict(state_dict)
            safe_load_final_model(model, state_dict)
        eval_time = time.time()
        if args.write_preds_to_file:
            use_dataloaders = [train_dataloader, dev_dataloader, test_dataloader] 
        elif args.use_dev_not_test:
            use_dataloaders = [dev_dataloader]
        else:
            use_dataloaders = [test_dataloader]

        for data_loader in use_dataloaders:
            stats_dict = train_or_test(args,
                            stats_dict, 
                            epoch=-1, 
                            model=model, 
                            data_loader=data_loader, 
                            tokenizer=tokenizer,
                            scaler=scaler,
                            write_preds_to_file=args.write_preds_to_file or args.write_graph_to_file)
        eval_time_msg = utils.format_training_time(start=eval_time, end=time.time())

        train_sel_for = stats_dict['train_sel_for'] if 'train_sel_for' in stats_dict else -1
        dev_sel_for   = stats_dict['dev_sel_for']   if 'dev_sel_for'   in stats_dict else -1        
        test_sel_for  = stats_dict['test_sel_for']  if 'test_sel_for'  in stats_dict else -1

        final_msg = f"Best epoch: {best_epoch} | train score: {train_sel_for:.2f} | dev score: {dev_sel_for:.2f} | test score: {test_sel_for:.2f} "

        report.print_epoch_scores(epoch = best_epoch, scores = stats_dict)
        report.write_epoch_scores(epoch = -1, scores = stats_dict)
        report.write_final_score(args, final_score_str = final_msg, time_msg=time_msg if args.do_train else eval_time_msg)
        report.job_finished()
        print()
        print(final_msg)
        print(time_msg if args.do_train else eval_time_msg)

    # visualize graph
    if args.do_graph_analysis:
        print("Running graph analysis...")
        start = time.time()
        if not os.path.exists('outputs'): os.mkdir('outputs')
        # make graphs
        graph_df = graph_utils.make_graph_df(args, test_dataloader, tokenizer)
        graph_nx = graph_utils.nx_graph_from_pd_df(args, graph_df)
        # plot the entire graph
        if args.num_eval_points > 0 and args.num_eval_points < 100:
            graph_utils.plot_graph(args, graph_nx, plot_type=args.graph_layout)
        # print summary statistics for the overall graph
        graph_utils.print_graph_summary(args, graph_nx)
        # plot the subgraph centered on the most connected node
        if args.num_eval_points > 0 and args.num_eval_points <= 200:        
            most_connected_node_subgraph = graph_utils.get_most_connected_node_subgraph(graph_nx)
            graph_utils.plot_graph(args, most_connected_node_subgraph, save_suffix='most_connected_subgraph', plot_type=args.graph_layout)
        # get single chain graphs
        if args.num_eval_points > 0 and args.num_eval_points <= 200:
            chains_subgraphs = graph_utils.get_chains(graph_nx, top_k=10)
            for graph_num, graph in enumerate(chains_subgraphs):
                if graph.number_of_nodes() > 1:
                    graph_utils.plot_graph(args, graph, save_suffix=f'chain-{graph_num}', plot_type='neato')
        # get high betweenness subgraphs
        if args.num_eval_points > 0 and args.num_eval_points <= 200:
            between_subgraphs = graph_utils.get_betweenness_subgraphs(graph_nx)
            for graph_num, graph in enumerate(between_subgraphs):
                if graph.number_of_nodes() > 1:
                    graph_utils.plot_graph(args, graph, save_suffix=f'between-{graph_num}', plot_type='dot')
        # plot random subgraphs of 10 nodes
        for i in range(10):
            try:
                n_nodes = min(20, args.num_eval_points)
                random_subgraph_idx = np.random.choice(np.arange(graph_nx.number_of_nodes()), size=n_nodes, replace=False)
                random_subgraph = graph_nx.subgraph(random_subgraph_idx)
                graph_utils.plot_graph(args, random_subgraph, save_suffix=f'random_subgraph-{i}', plot_type = 'neato')
            except:
                continue
        # get time
        time_msg = utils.format_training_time(start=start_time, end=time.time())
        print(f'\n{time_msg}')




