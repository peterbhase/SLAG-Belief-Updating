'''
this file contains a bunch of random utilities as well as important data loading utilities and configs
'''
import numpy as np
import argparse
import pandas as pd
from copy import deepcopy
import jsonlines
import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader
import math
import metrics
import itertools
from models.probe import Probe
from models.learned_optimizer import ModelWithLearnedOptimizer

def add_df_cols_from_args(df, args, cols):
    for col in cols:
        df[col] = getattr(args, col)

def command_to_dict(command):
    command_dict = {}
    items = command.split()
    for idx, item in enumerate(items):
        if idx == len(items)-1:
            break
        if item[:2] == '--':
            k = item[2:]
            v = items[idx+1]
            command_dict[k] = v
        elif item[0] == '-':
            k = item[1:]
            v = items[idx+1]
            command_dict[k] = v        
    return command_dict

def get_result_sheet_name(args, experiment_name, is_update_epoch, split_name):
    if is_update_epoch:
        name_extension = f"oth{args.num_random_other}_k{args.update_steps}_r{args.num_successive_updates}_"
    else:
        name_extension = ""
    if hasattr(args, 'eval_beam_search_alt_labels'):
        if args.eval_beam_search_alt_labels:
            name_extension += "beam_"
    save_name = f"{name_extension}{split_name}_{experiment_name}.csv"
    return save_name

def get_experiment_name_from_command(job_command):
    os.system(job_command +  " --get_experiment_name_only true")
    exp_name = ''.join([line for line in open('outputs/tmp_experiment_name.txt', 'r')])
    return exp_name

def custom_chunk_array(array, size):
    # return chunks from the array of size=size, in left to right order
    # if array % size != 0, then last components of the array are also added, but will not be of size=size
    if len(array) <= size:
        return [array]
    start_idx = 0
    chunks = []
    for end_idx in range(1, len(array)+1):
        if end_idx % size == 0 or end_idx == len(array):
            chunks.append(array[start_idx:end_idx])
            start_idx = end_idx
    return chunks

def flip_array_pairwise(array: np.ndarray) -> np.ndarray:
    # flip the pairs of elements in an array, in order. for example: 0,1,2,3 -> 1,0,3,2
    flip_pairwise_idx = [i+1 if i % 2 == 0 else i-1 for i in range(len(array))] # switch every pair of points with one another, in order
    return array[flip_pairwise_idx]

def arraydiff1d(iter1, iter2):
    # returns all the items in iter1 that are not in iter2
    if not (type(iter2) is list or type(iter2) is np.ndarray or type(iter2) is torch.Tensor):
        iter2 = [iter2]
    return np.array([item for item in iter1 if item not in iter2])

def init_epoch_stats(stat_names):
    return_dict = {}
    for stat_name in stat_names:
        return_dict[stat_name] = 0
        return_dict[f"before_{stat_name}"] = 0
        return_dict[f"after_{stat_name}"] = 0
    return return_dict

def safe_load_base_model(model, state_dict):
    # handles some loading cases, while requiring that source state dict keys exactly match destination keys
    load_from_de_cao_codebase = 'state_dict' in state_dict
    if load_from_de_cao_codebase: 
        state_dict = state_dict['state_dict'] # convert from pytorch lightning
        load_from_huggingface_classifier = 'model.classifier.weight' in state_dict
        if load_from_huggingface_classifier:
            from_to = {
                'model.model.pooler.dense.weight' : 'model.probe.classifier.1.weight',
                'model.model.pooler.dense.bias' : 'model.probe.classifier.1.bias',
                'model.classifier.weight' : 'model.probe.classifier.4.weight',
                'model.classifier.bias' : 'model.probe.classifier.4.bias',
            }
            for k_from, k_to in from_to.items():
                state_dict[k_to] = state_dict[k_from]
                if 'classifier.weight' in k_from:
                    state_dict[k_to] = torch.cat((-state_dict[k_to], state_dict[k_to]), dim=0)
                if 'classifier.bias' in k_from:
                    state_dict[k_to] = torch.cat((torch.tensor([0]), state_dict[k_to]), dim=0)
                if 'pooler' not in k_from:
                    state_dict.pop(k_from)
            for k in list(state_dict.keys()):
                old_k = k
                k = k.replace("model.probe.", "probe.")
                k = k.replace("model.model.", "model.")
                state_dict[k] = state_dict[old_k]
                state_dict.pop(old_k)
    if type(model) is ModelWithLearnedOptimizer:
        load_module = model.model
    elif type(model) is Probe:
        load_module = model
    assert set(state_dict.keys()) == set(load_module.state_dict().keys()), "keys in load_module do not match keys from before_training_load_path"
    load_module.load_state_dict(state_dict)

def safe_load_final_model(model, state_dict):
    # handles some loading cases, while requiring that source state dict keys exactly match destination keys
    if 'state_dict' in state_dict: # this means loading from de Cao pytorch-lightning code
        state_dict = state_dict['state_dict']
        # need to remove some keys to load models from de Cao codebase, as well as rename others
        sd_keys = sorted(list(state_dict.keys()))
        for k in sd_keys:
            if any([name in k for name in ['_lp']]):
                state_dict.pop(k)
            else:
                old_k = k
                k = k.replace("model_", "model_model_") # hacky renaming
                k = k.replace("model.", "model.model.")
                k = k.replace("model.model.model.model.", "model.model.model.")
                k = k.replace("model_model_model_model_", "model_model_model_")
                if k != old_k:
                    state_dict[k] = state_dict[old_k]
                    state_dict.pop(old_k)
    assert set(state_dict.keys()) == set(model.state_dict().keys()), "keys in load_module do not match keys from before_training_load_path"
    model.load_state_dict(state_dict)

def negate_first_sentence(sentence):
    sentences = sentence.split('.')
    sentences = [negate_sentence(sentences[0])] + sentences[1:]
    sentences = '.'.join(sentences)
    return sentences

def negate_sentence(sentence):
    # heuristic used in LeapOfThought data
    assert any([key_word in sentence for key_word in ['is', 'does', 'has', 'desires']]), \
        f"sentence '{sentence}' breaks LeapOfThought template for negateable sentences"
    if 'not' in sentence:
        sentence = sentence.replace("does not have", "has")
        sentence = sentence.replace("does not desire", "desires")
        sentence = sentence.replace("does not", "does")
        if 'opposite' in sentence:
            sentence = sentence.replace("opposite of", "same as")
        elif 'same as' in sentence:
            sentence = sentence.replace("same as", "opposite of")
        sentence = sentence.replace("is not", "is")
    else:
        sentence = sentence.replace("does", "does not")
        sentence = sentence.replace("has", "does not have")
        sentence = sentence.replace("desires", "does not desire")
        if 'opposite' in sentence:
            sentence = sentence.replace("opposite of", "same as")
        elif 'same as' in sentence:
            sentence = sentence.replace("same as", "opposite of")
        else:
            sentence = sentence.replace("is", "is not")
    return sentence

def pad_kwargs_to_multiple_of_8(kwargs, tokenizer):
    for k,v in kwargs.items():
        if type(v) is torch.Tensor and 'label' not in k:
            len = v.size(-1)
            if len % 8 == 0:
                continue
            else:
                make_len = len + (8 - len % 8)
            if 'ids' in k:
                pad_id = tokenizer.pad_token_id
            elif 'attention' in k:
                pad_id = 0
            else:
                print(k)
                import pdb; pdb.set_trace()
                raise ValueError(f"what is the kind of tensor we are padding? {k}?")
            kwargs[k] = torch.nn.ConstantPad2d((0, make_len-len, 0, 0), pad_id)(v) 

def move_kwargs_to_gpu(kwargs):
    for k,v in kwargs.items():
        if type(v) is torch.Tensor:
            kwargs[k] = v.cuda(non_blocking=True)

def get_model_save_and_load_paths(args):
    # returns: save path for saving new model from an experiment, the before_training_load_path and the before_eval_load_path
    if args.update_beliefs and not args.use_learned_optimizer:
        experiment_load_name = get_experiment_name(args)
        experiment_save_name = get_experiment_name_update_beliefs(args)
        before_eval_load_path = os.path.join(args.save_dir, experiment_load_name + '.pt')
        before_training_load_path = None if not args.pre_eval else before_eval_load_path
        save_path = os.path.join(args.save_dir, experiment_save_name + '.pt')
    elif args.update_beliefs and args.use_learned_optimizer:
        experiment_before_train_name = get_experiment_name_learned_opt(args)
        experiment_load_name = get_experiment_name_learned_opt(args)
        experiment_save_name = get_experiment_name_update_beliefs(args, experiment_name = experiment_load_name)
        before_training_load_path = os.path.join(args.save_dir, experiment_before_train_name + '.pt') 
        before_eval_load_path = os.path.join(args.save_dir, experiment_load_name + '.pt')
        save_path = os.path.join(args.save_dir, experiment_save_name + '_updated.pt')
    elif args.use_learned_optimizer and args.do_train:
        experiment_load_name = get_experiment_name(args)
        experiment_save_name = get_experiment_name_learned_opt(args)
        before_training_load_path = os.path.join(args.save_dir, experiment_load_name + '.pt')
        save_path = os.path.join(args.save_dir, experiment_save_name + '.pt')
        before_eval_load_path = save_path
    elif args.use_learned_optimizer and not args.do_train:
        experiment_before_train_name = get_experiment_name(args)
        experiment_load_name = get_experiment_name_learned_opt(args)
        before_training_load_path = os.path.join(args.save_dir, experiment_before_train_name + '.pt') 
        before_eval_load_path = os.path.join(args.save_dir, experiment_load_name + '.pt')
        save_path = None
    elif args.do_train:
        experiment_save_name = get_experiment_name(args)
        before_training_load_path = None
        save_path = os.path.join(args.save_dir, experiment_save_name + '.pt')
        before_eval_load_path = save_path
    elif not args.do_train:
        experiment_save_name = get_experiment_name(args)
        before_training_load_path = None
        save_path = None
        before_eval_load_path = os.path.join(args.save_dir, experiment_save_name + '.pt')
    else:
        raise NotImplementedError("missing a condition on experiment naming and save/load pathing")
    # overwrite before training load paths if load_model_path is specified, as well as before_eval_path if not training
    if args.load_model_path is not None:
        before_training_load_path = args.load_model_path
        if not args.use_learned_optimizer:
            before_eval_load_path = before_training_load_path
    if args.load_opt_path is not None:
        before_eval_load_path = args.load_opt_path
    if args.save_model_name is not None:
        save_path = os.path.join(args.save_dir, args.save_model_name + '.pt')
        before_eval_load_path = save_path
    return before_training_load_path, before_eval_load_path, save_path

def slice_list_of_kwargs(list_kwargs, idx):
    return_list = []
    for single_point in list_kwargs:
        if single_point is None:
            return_list.append(None)
        else:
            return_list.append({k : v[idx,...] for k,v in single_point.items()})
    return return_list

def select_point_from_kwargs(kwargs, idx):
    # handles indexing without losing the first dimension for inputs, and losing where appropriate for lables
    # if any of kwargs are lists, allow for indexing of these as well
    if type(idx) is torch.Tensor or type(idx) is np.ndarray: idx = idx.item()
    return_kwargs = {k: v[idx].view(1,-1) for k,v in kwargs.items() if 'labels' not in k and type(v) is torch.Tensor}
    add_kwargs = {k: v[idx] for k,v in kwargs.items() if type(v) is list}
    return_kwargs.update(add_kwargs)
    for k in list(kwargs.keys()):
        if 'labels' in k:
            label_dim = kwargs[k].dim()
            if label_dim == 1:
                return_kwargs[k] = kwargs[k][idx].view(1)
            if label_dim == 2: # i.e., seq2seq
                return_kwargs[k] = kwargs[k][idx].view(1,-1)
    return return_kwargs

def slice_kwargs(kwargs, idx):
    if not (type(idx) is torch.Tensor or type(idx) is np.ndarray or type(idx) is list): 
        idx = np.array([idx])
    return_kwargs = {}
    for k in list(kwargs.keys()):
        if type(kwargs[k]) is torch.Tensor or type(kwargs[k]) is np.ndarray:
            return_kwargs[k] = kwargs[k][idx,...]  
        elif type(kwargs[k]) is list:
            return_kwargs[k] = [item for keep_idx, item in enumerate(kwargs[k]) if keep_idx in idx]
        else:
            return_kwargs[k] = None
    return return_kwargs

def get_prop_true(args, dataloader):
    dataset=dataloader.dataset
    if args.preprocess_data_when_loading:
        orig_labels = [datapoint['text_data'][0]["orig_label"] for datapoint in dataset]
    else:
        orig_labels = [datapoint["orig_label"] for datapoint in dataset]
    labels = [orig_label == args.dataset_config['var_for_true'] for orig_label in orig_labels]
    label_prop = np.mean(labels)
    return 100*round(label_prop,2)
    
def get_random_subset(dataloader, size, exclude_ids, batch_size, data_sample_counts=None):
    if data_sample_counts is not None:
        _data_sample_counts = [(k,v) for k,v in data_sample_counts.items()]
        np.random.shuffle(_data_sample_counts) # used to make sure we use a 'random' order, starting from a dict of all 0s
        _data_sample_counts = sorted(_data_sample_counts, key = lambda x : x[1])
        _data_sample_counts = [data_sample_count for data_sample_count in _data_sample_counts if data_sample_count[0] not in exclude_ids]
        sample_idx = np.array([data_sample_count[0] for data_sample_count in _data_sample_counts[:size]])
        for idx in sample_idx:
            data_sample_counts[idx] += 1
    else:
        n_points = len(dataloader.dataset)
        eligible_idx = np.setdiff1d(np.arange(n_points), np.array(exclude_ids).reshape(-1))
        sample_idx = np.random.choice(eligible_idx, min(size, len(eligible_idx)), replace=False)
    subset = torch.utils.data.Subset(dataloader.dataset, sample_idx.tolist())
    subset_dataloader = torch.utils.data.DataLoader(subset, batch_size=batch_size, collate_fn=dataloader.collate_fn, shuffle=False)
    return subset_dataloader, sample_idx

def tensor_into_batches(tensor, batch_size):
    tensor_size = tensor.size(0)
    num_batches = max(1, math.ceil(tensor_size / batch_size)) # num_restarts is num parallel runs
    return tensor.chunk(num_batches)

def batch_list_of_kwargs(list_of_kwargs, batch_size):
    # convert list of kwargs of varying batch size into list of kwargs of batch size
    # requires tensors of the same key to have the same max seq len
    # allows for Nones in list_of_kwargs
    not_none_kwargs = [kwargs for kwargs in list_of_kwargs if kwargs is not None]
    keys = [k for k in list(not_none_kwargs[0].keys()) if type(not_none_kwargs[0][k]) is torch.Tensor]
    keys_to_stacked_tensors = {k : torch.cat([kwargs[k] for kwargs in not_none_kwargs], dim=0) for k in keys}
    keys_to_batched_tensors = {k : v.chunk(batch_size) for k,v in keys_to_stacked_tensors.items()}
    num_batches = len(keys_to_batched_tensors[keys[0]])
    return_batches = []
    for batch_num in range(num_batches):
        batch = {k : keys_to_batched_tensors[k][batch_num] for k in keys}
        return_batches.append(batch)
    return return_batches

def kwargs_into_batches(kwargs, batch_size):
    batches = {}
    for key in list(kwargs.keys()):
        split_kwargs = tensor_into_batches(kwargs[key], batch_size)
        num_batches = len(split_kwargs)
        for batch_num in range(num_batches):
            if batch_num not in batches:
                batches[batch_num] = {}
            batches[batch_num][key] = split_kwargs[batch_num]
    return list(batches.values())

def decode_until_eos_token(tokenizer, tensor):
    assert tensor.dim() == 2, "tensor should be batch_size x max_seq_len"
    eos_id = tokenizer.eos_token_id
    seqs = tensor.tolist()
    eos_first_idx = [seq.index(eos_id) if eos_id in seq else len(seq) for seq in seqs]
    return [tokenizer.decode(seq[:eos_first_idx[i]], skip_special_tokens=True) for i, seq in enumerate(seqs)]

def first_sentences_from_samples(samples):
    # samples is list of lists of strs
    new_samples = []
    for list_samples in samples:
        add_new_samples = []
        for sample in list_samples:
            sample = sample.split('\n')[0]
            if '?' in sample:
                sample = sample.split('?')[0] + '?'
            elif '.' in sample:
                sample = sample.split('.')[0] + '.'
            add_new_samples.append(sample)
        new_samples.append(add_new_samples)
    return new_samples

def format_training_time(start, end):
    training_time = (end-start) / 60
    unit = 'minutes' if training_time < 60 else 'hours'
    training_time = training_time if training_time < 60 else training_time / 60
    time_msg = f"\nTotal runtime: {training_time:.2f} {unit}"
    return time_msg

def lower_case_first_letter(x):
    return x[0].lower() + x[1:]

def prepend_token_to_tensor(token_id, tensor):
    # prepend a column of tokens to the last dim of a tensor
    prepend_tensor = torch.ones_like(tensor)[...,0].unsqueeze(-1).fill_(token_id).long()
    return torch.cat([prepend_tensor, tensor], dim=-1)

def append_token_to_tensor(token_id, tensor):
    # append a column of tokens to the last dim of a tensor
    append_tensor = torch.ones_like(tensor)[...,0].unsqueeze(-1).fill_(token_id).long()
    return torch.cat([tensor, append_tensor], dim=-1)

def add_experiment_name_to_args(args):
    e_name = get_experiment_name(args)
    # overwrite e_name if save_model_name is not None
    if args.save_model_name is not None:
        e_name = args.save_model_name
    args.base_experiment_name = e_name
    if args.use_learned_optimizer:
        e_name = get_experiment_name_learned_opt(args)
    if args.update_beliefs:
        e_name = get_experiment_name_update_beliefs(args, e_name)
    args.experiment_name = e_name
    return e_name

def get_experiment_name(args):
    # get experiment name for training base model
    # if load_model_path provided, that model name becomes the base_experiment_name
    if args.load_model_path:
        return args.load_model_path.split('/')[-1].replace(".pt", "").replace(".ckpt", "")
    if args.do_train:
        trained_or_updated = f'finetuned-{args.update_parameters}'
    else:
        trained_or_updated = 'pretrained'
    if args.load_finetuned_model:
        trained_or_updated = f'finetuned-{args.orig_trained_parameters}'
    model_name = args.model.replace('facebook/', '')
    probe = args.probe if args.probing_style == 'model' else args.probing_style
    # additional objectives
    paraphrases = '_paraphrases' if args.fit_model_to_paraphrases else ''
    if args.small_data and not args.use_learned_optimizer:
        data_addin = "_DEBUG" 
    elif args.num_train_points > 0 and not args.use_learned_optimizer:
        data_addin = f"_n{args.num_train_points}"
    else:
        data_addin = ''
    seed = args.seed if args.load_seed < 0 else args.load_seed
    return f"experiment_{model_name}_{args.dataset}_{probe}-probe_{trained_or_updated}{paraphrases}_sd{seed}{data_addin}"

def get_experiment_name_learned_opt(args):
    # get experiment name for training learned optimizer
    experiment_name = get_experiment_name(args)
    learned_opt_or_update_cond = "_learned-opt"
    if args.implementation == 'de_cao': learned_opt_or_update_cond += '-de-cao'
    if args.implementation == 'new'   : learned_opt_or_update_cond += '-ours'
    learned_opt_or_update_cond += f"_k{args.learned_opt_steps}"
    learned_opt_or_update_cond += f'_r{args.learned_successive_updates}' if args.learned_successive_updates > -1 else f"_r1"
    # give objective
    objective = '_obj-ce'
    if args.min_corruption:         objective += '-crp'
    if args.divergences == 'kl':    objective += '-kl'
    if args.fit_opt_to_dependent_propositions:  
        objective += '-dep'
        if args.leapofthought_add_both_for_training:
            objective += 'KL' # dependent data also used in KL term
    if args.fit_opt_to_independent_propositions:  
        objective += '-ind'
    if args.fit_opt_to_paraphrases: objective += '-par'
    if not args.detach_prev_updates: objective += '-nodetach'
    if args.update_small_data:
        data_addin = "_DEBUG" 
    elif args.num_train_points > 0:
        data_addin = f"_n{args.num_train_points}"
    else:
        data_addin = ''
    if args.fit_to_alt_labels or args.load_alt_labels_model:
        alt_labels = '_alt-labels' 
    else:
        alt_labels = ''
    if args.beam_search_alt_labels:
        alt_labels += '-beam'
    experiment_name = f"{experiment_name}{learned_opt_or_update_cond}{objective}{alt_labels}_opt-{args.optimizer}_sd{args.seed}{data_addin}"
    return experiment_name

def args_to_obj_name(args):
     # give objective
    objective = 'ce'
    if args.min_corruption:         objective += '-crp'
    if args.divergences == 'kl':    objective += '-kl'
    if args.fit_opt_to_dependent_propositions:  objective += '-dep'
    if args.fit_opt_to_independent_propositions:  objective += '-ind'
    if args.fit_opt_to_paraphrases: objective += '-par'
    if not args.detach_prev_updates: objective += '-nodetach'
    return objective

def get_experiment_name_update_beliefs(args, experiment_name = None):
    # get experiment name for updating beliefs. can be applied to base model, or model with learned optimizer by supplying experiment_name
    if experiment_name is None:
        experiment_name = get_experiment_name(args)
    learned_opt_or_update_cond = f'_r{args.num_successive_updates}'
    learned_opt_or_update_cond += f"_steps-{args.update_steps}"
    # paraphrases = '_paraphrase-fit' if args.fit_opt_to_paraphrases else '' # oracle condition not currently implemented
    optimizer = f"_opt-{args.optimizer}" if not args.use_learned_optimizer else ''
    optimizer += f'_lr{args.lr}'
    data_addin = "_DEBUG" if (args.update_small_data or args.num_train_points > 0) else ''
    experiment_name = f"{experiment_name}{learned_opt_or_update_cond}{optimizer}_sd{args.seed}{data_addin}"
    return experiment_name
        
def add_dataset_config_to_args(args, dataset_name):
    dataset_name_to_config = {
        'FEVER' : {
            'var_for_true' : 'SUPPORTS',
            'data_names' : ['proposition', 'main_opt_context'],
            'candidate_set' : ['true', 'false'],
            'stat_names' :  ['train_acc', 
                             'dev_acc', 'dev_upd_suc', 'dev_oth_ret', 'dev_per_chg', 'dev_crp_rte',
                             'test_acc', 'test_upd_suc', 'test_oth_ret', 'test_per_chg', 'test_crp_rte']
        },
        'ZSRE' : {
            'data_names' : ['proposition', 'main_opt_context', 'paraphrases'],
            'candidate_set' : None,
            'stat_names' :  ['train_acc', 
                             'dev_acc', 'dev_cons', 'dev_par_acc', 'dev_par_eq', 'dev_upd_suc', 'dev_oth_ret', 'dev_per_chg', 'dev_crp_rte',
                             'test_acc', 'test_cons', 'test_par_acc', 'test_par_eq',  'test_upd_suc', 'test_oth_ret', 'test_per_chg', 'test_crp_rte']
        },
        'LeapOfThought' : {
            'var_for_true' : 1,
            'data_names' : ['proposition', 'main_opt_context', 'dependent_proposition'],#, 'independent_proposition'],
            'candidate_set' : ['true', 'false'],
            'stat_names' :  ['train_acc', 
                             'dev_acc', 'dev_upd_suc', 'dev_per_chg', 'dev_crp_rte', 'dev_dep_acc', 'dev_odp_acc',
                             'test_acc', 'test_upd_suc', 'test_per_chg', 'test_crp_rte', 'test_dep_acc', 'test_odp_acc']
        },
        'Wikidata5m' : {
            'data_names' : ['proposition', 'main_opt_context', 'independent_proposition', 'paraphrases', 'entity_paraphrase', 'relation_paraphrase'],
            'candidate_set' : None,
            'stat_names' :  ['train_acc', 
                            'dev_acc', 'dev_upd_suc', 'dev_per_chg', 'dev_crp_rte', 'dev_ind_acc', 'dev_ind_chg', 'dev_ind_ret', 'dev_ent_acc', 'dev_ent_eq', 'dev_rel_acc', 'dev_rel_eq', 'dev_cons', 'dev_par_acc', 'dev_par_eq',
                            'test_acc', 'test_upd_suc', 'test_per_chg', 'test_crp_rte', 'test_ind_acc', 'test_ind_chg', 'test_ind_ret', 'test_ent_acc', 'test_ent_eq', 'test_rel_acc', 'test_rel_eq', 'test_cons', 'test_par_acc',  'test_par_eq']
        },
        'ParaRel' : {
            'data_names' : ['proposition', 'paraphrases'],
            'candidate_set' : None,
        }
    }
    args.dataset_config = dataset_name_to_config[dataset_name]
    return

class PropositionDataset(Dataset):
    def __init__(
        self,
        args,
        tokenizer,
        data, # list of dictionaries, one dict per data point
        split_name=None,
    ):
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.data = data
        self.data_names = args.dataset_config['data_names']
        self.split_name = split_name
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def retrieve_ids(self, ids):
        data = [self.data[id] for id in ids]
        batch = self.collate_fn(data)
        return batch

    def add_labels_to_batch(self, batch, orig_batch, padding, from_name, to_name):
        # add label tensors to batch, drawing from from_name key in orig_batch and adding as to_name key in batch
        if self.args.probing_style == 'model':
            labels = torch.tensor([item[from_name + '_label'] == self.args.dataset_config['var_for_true'] for item in orig_batch])
            batch[to_name + '_labels'] = labels.long()
            batch['nan_' + to_name + '_labels'] = torch.tensor([item[from_name + '_label'] is None for item in orig_batch])
        elif self.args.probing_style=='cloze':
            self.tokenizer_output = self.tokenizer(
                [item[from_name + '_label'] for item in orig_batch],
                return_tensors="pt",
                add_special_tokens=False
            )
            batch[to_name + '_labels'] = self.tokenizer_output['input_ids'].squeeze(dim=1)
        elif self.args.probing_style=='seq2seq':
            tokenize_list = [item[from_name + '_label'] for item in orig_batch]
            tokenizer_output = self.tokenizer(
                tokenize_list,
                return_tensors="pt",
                padding=padding,
                max_length=self.args.max_seq_len,
                truncation=True,
            )
            batch[f'{to_name}_decoder_input_ids'] = tokenizer_output['input_ids']
            # set first elem in main_decoder_input_ids to eos
            batch[f'{to_name}_decoder_input_ids'][:,0] = self.tokenizer.eos_token_id
            # for labels, replace pad tokens with -100, which is ignore index on labels
            labels = tokenizer_output['input_ids']
            labels = labels.masked_fill(labels==self.tokenizer.pad_token_id, -100)
            batch[to_name + '_labels'] = labels

    def add_paraphrases_to_batch(self, batch, orig_batch, data_names, padding):
        # get paraphrases. note these labels are for training purposes but not used in evaluation. 
        if 'paraphrases' in data_names:
            batch['paraphrases'] = []
            name_to_list_of_data = {'paraphrases' : [], 'labels' : []} # for obtaining batched_paraphrases
            for data_id, item in enumerate(orig_batch):
                num_paraphrases=len(item['paraphrases'])
                if num_paraphrases > 0:
                    paraphrase_set = {k : v for k,v in self.tokenizer(item['paraphrases'], return_tensors='pt', padding=padding, max_length=self.args.max_seq_len, truncation=True).items()}
                    paraphrase_set['decoder_input_ids'] = paraphrase_set['labels'] = batch['main_labels'][data_id].expand(num_paraphrases, -1).clone()
                    # now do the reverse of the replace for main_labels and fill the -100 ignore index value with pad token ids
                    paraphrase_set['decoder_input_ids'].masked_fill_(paraphrase_set['decoder_input_ids']==-100, self.tokenizer.pad_token_id)
                    batch['paraphrases'].append(paraphrase_set)
                    name_to_list_of_data['paraphrases'].extend(item['paraphrases'])
                    name_to_list_of_data['labels'].extend([item['orig_label'] for _ in range(num_paraphrases)])
                else:
                    batch['paraphrases'].append(None)
            # get all_paraphrases for more efficient training (note uses orig_labels)
            if len(name_to_list_of_data['paraphrases']) > 0:
                all_paraphrases_batch = {
                    f"{data_name}_{tensor_name}" : tensor
                    for data_name in ['paraphrases', 'labels']
                    for tensor_name, tensor in self.tokenizer(
                        name_to_list_of_data[data_name],
                        return_tensors="pt",
                        padding=padding,
                        max_length=self.args.max_seq_len,
                        truncation=True,
                    ).items()
                }
                # for labels, replace pad tokens with -100, which is ignore index on labels
                labels = all_paraphrases_batch['labels_input_ids']
                labels = labels.masked_fill(labels==self.tokenizer.pad_token_id, -100)
                batch['concatenated_paraphrases'] = {
                    'input_ids' : all_paraphrases_batch['paraphrases_input_ids'],
                    'attention_mask' : all_paraphrases_batch['paraphrases_attention_mask'],
                    'decoder_input_ids' : all_paraphrases_batch['labels_input_ids'],
                    'labels' : labels
                }
            else:
                batch['concatenated_paraphrases'] = None

    def collate_fn(self, orig_batch):
        '''
        this is where most of the data preprocessing, e.g. tokenization, actually occurs
        this can either be passed to DataLoader for on-load preprocessing, or used in the load_data below to do all of the preprocessing up front
        '''
        data_names = self.args.dataset_config['data_names']
        main_data_names = [name for name in data_names if 'main' in name or 'dependent' in name or 'entity' in name or 'relation' in name]
        padding = 'max_length' if self.args.preprocess_data_when_loading else True
        # will expect a list if this is the DataLoader.collate_fn. will expect dict if this is being used to process data-points in load_data
        if type(orig_batch) is dict:
            orig_batch = [orig_batch]
        # add, e.g., main_input_ids, main_attention_mask, main_opt_context_input_ids, etc. to batch
        batch = {
            f"{data_name}_{tensor_name}" : tensor
            for data_name in main_data_names
            for tensor_name, tensor in self.tokenizer(
                [item[data_name] for item in orig_batch if item[data_name] is not None], # NOTE None excluded here. should only be used with entity paraphrases where there are none, see wikidata5m case in load_data
                return_tensors="pt",
                padding=padding,
                max_length=self.args.max_seq_len,
                truncation=True,
            ).items()
        }
        main_label_source = 'orig' if not self.args.fit_to_alt_labels else 'update'
        self.add_labels_to_batch(batch, orig_batch, padding, from_name=main_label_source, to_name='main')
        self.add_labels_to_batch(batch, orig_batch, padding, from_name='orig', to_name='orig')
        if 'dependent_proposition' in data_names:
            self.add_labels_to_batch(batch, orig_batch, padding, from_name='dependent_proposition', to_name='dependent_proposition')
            self.add_labels_to_batch(batch, orig_batch, padding, from_name='dependent_proposition_orig', to_name='dependent_proposition_orig')
        if 'independent_proposition' in data_names:
            self.add_labels_to_batch(batch, orig_batch, padding, from_name='independent_proposition', to_name='independent_proposition')
        if 'entity_paraphrase' in data_names:
            self.add_labels_to_batch(batch, orig_batch, padding, from_name='orig', to_name='entity_paraphrase')
        if 'relation_paraphrase' in data_names:
            self.add_labels_to_batch(batch, orig_batch, padding, from_name='orig', to_name='relation_paraphrase')
        self.add_paraphrases_to_batch(batch, orig_batch, data_names, padding) # should call after main_labels have been added
        batch["text_data"] = orig_batch # string representations of data
        # add mask token idx for cloze prompts
        if self.args.probing_style == 'cloze':
            input_ids = batch['main_input_ids']
            _, where_mask_tokens = (input_ids==self.tokenizer.mask_token_id).nonzero(as_tuple=True)
            batch['main_mask_token_idx'] = where_mask_tokens
        # add id
        batch['id'] = torch.tensor([item['id'] for item in orig_batch])
        # lastly, perform this padding if using fp16
        if self.args.fp16:
            pad_kwargs_to_multiple_of_8(batch, self.tokenizer)
        return batch

def offline_collate_fn(batch):
    # used as DataLoader.collate_fn when the above collate_fn is used in load_data to process data points before training
    return_batch = {}
    keys = list(batch[0].keys())
    for key in keys:
        if type(batch[0][key]) is torch.Tensor:
            return_batch[key] = torch.stack([item[key] for item in batch], dim=0).squeeze(1)
        elif key == 'paraphrases' or key=='text_data':
            return_batch[key] = [item[key][0] for item in batch]
        else:
            return_batch[key] = [item[key] for item in batch]
    return return_batch

def load_data(args, tokenizer, shuffle_loaders=True):
    '''
    returns train, dev, and test dataloaders
    '''
    split_names = ['train','dev','test']
    data_names = args.dataset_config['data_names']
    data_names.extend(['main', 'orig_label']) # main is primary model input
    max_points = (100 if args.small_data or args.update_small_data else 2e8)
    max_train_points = args.num_train_points if args.num_train_points > 0 else max_points
    max_eval_points = args.num_eval_points if args.num_eval_points > 0 else max_points

    if args.dataset == 'FEVER':
        train_path = os.path.join(args.data_dir, 'KILT', 'fever_reshuffled-train-kilt.jsonl')
        dev_path = os.path.join(args.data_dir, 'KILT', 'fever_reshuffled-dev-kilt.jsonl')
        test_path = os.path.join(args.data_dir, 'KILT', 'fever_reshuffled-test-kilt.jsonl')
        data_splits = {name : [] for name in split_names}
        data_paths = [train_path, dev_path, test_path]
        for split_name, data_path in zip(split_names, data_paths):
            use_alt_labels = False if (split_name != 'train' and args.update_eval_truthfully) else args.fit_to_alt_labels
            with jsonlines.open(data_path) as file:
                for data_num, datapoint in enumerate(file):
                    if data_num >= (max_train_points if split_name=='train' else max_eval_points):
                        break
                    orig_label = datapoint["output"][0]["answer"]
                    is_true = (orig_label == args.dataset_config['var_for_true'])
                    model_pred = datapoint[f'{args.base_experiment_name}_pred'] if f'{args.base_experiment_name}_pred' in datapoint else None
                    model_pred_str = ['false', 'true'][int(model_pred)] if model_pred is not None else None
                    orig_label_str = 'true' if is_true else 'false'
                    if not use_alt_labels:
                        update_label = orig_label
                        update_label_str = orig_label_str
                    if use_alt_labels:
                        assert model_pred is not None, "need to write task model predictions to file with --write_preds_to_file true "
                        # want label that belongs to opposite from predicted
                        update_label = ['REJECTS', 'SUPPORTS'][1-int(model_pred)]
                        update_label_str = ['false', 'true'][1-int(model_pred)]
                    proposition = datapoint["input"]
                    add_data_point = {
                        "proposition": proposition,
                        "orig_label": orig_label,
                        "update_label": update_label, # label that model is updated to predict when use_learned_optimizer or update_beliefs
                        "main_opt_context" : f"{model_pred_str} >> {update_label_str} || {proposition}", # de Cao format
                        "id" : data_num,
                        'prediction' : model_pred,
                    }
                    if args.probing_style == 'model':
                        add_data_point['main'] = proposition
                    if args.probing_style == 'cloze':
                        add_data_point['main'] = f"It is {tokenizer.mask_token} that {lower_case_first_letter(proposition)}",
                    # add graph viz data
                    if args.do_graph_analysis and f'{args.experiment_name}_flips' in datapoint:
                        add_data_point.update({
                            f'flipped_points' : datapoint[f'{args.experiment_name}_flips'],
                            f'update_pred': datapoint[f'{args.experiment_name}_update_pred']
                        })
                    data_splits[split_name].append(add_data_point)
    if args.dataset == 'ZSRE':
        train_path = os.path.join(args.data_dir, 'KILT', 'zeroshot_reshuffled-train-kilt.jsonl')
        dev_path = os.path.join(args.data_dir, 'KILT', 'zeroshot_reshuffled-dev-kilt.jsonl')
        test_path = os.path.join(args.data_dir, 'KILT', 'zeroshot_reshuffled-test-kilt.jsonl')
        data_splits = {name : [] for name in split_names}
        data_paths = [train_path, dev_path, test_path]
        for split_name, data_path in zip(split_names, data_paths):
            use_alt_labels = False if (split_name != 'train' and args.update_eval_truthfully) else args.fit_to_alt_labels
            # first read all the labels to get alt_labels, if need be
            if args.fit_to_alt_labels and not args.beam_search_alt_labels:
                with jsonlines.open(data_path) as file:
                    eligible_alt_labels = []
                    for data_num, datapoint in enumerate(file):
                        eligible_alt_labels.append(datapoint['output'][0]['answer'])
                        if data_num >= (max_train_points if split_name=='train' else max_eval_points):
                            break
                    eligible_alt_labels = np.array(eligible_alt_labels)
                    n_points = len(eligible_alt_labels)
            with jsonlines.open(data_path) as file:
                for data_num, datapoint in enumerate(file):
                    if data_num >= (max_train_points if split_name=='train' else max_eval_points):
                        break
                    # model_pred = datapoint[f'prediction'] if f'prediction' in datapoint else None
                    model_pred = datapoint[f'{args.base_experiment_name}_pred'] if f'{args.base_experiment_name}_pred' in datapoint else None
                    orig_label = datapoint["output"][0]["answer"]
                    paraphrases = np.array(datapoint['meta']['template_questions'])
                    if not use_alt_labels:
                        update_label = orig_label
                    if use_alt_labels:
                        assert model_pred is not None, "need to write task model predictions to file with --write_preds_to_file true "
                        if args.beam_search_alt_labels or args.eval_beam_search_alt_labels:
                            update_label = np.random.RandomState(seed=args.seed).choice(datapoint[f"{args.base_experiment_name}_alt_preds"])
                        # want label that belongs to random choice from alternatives in the data split
                        else:
                            # branch alt labels based on correctness
                            correct = metrics.compute_acc_sum(args.probing_style, [model_pred], [orig_label], tokenizer)
                            if correct:
                                eligible_idx = np.setdiff1d(np.arange(n_points), data_num)
                                update_label = np.random.choice(eligible_alt_labels[eligible_idx])
                            if not correct:
                                update_label = orig_label
                    multiple_labels = [output["answer"] for output in datapoint['output']]
                    individual_point_idx_to_add = [0] if not args.paraphrases_to_unique_points else np.arange(len(paraphrases))
                    for idx in individual_point_idx_to_add:
                        proposition = paraphrases[idx]
                        other_idx = np.setdiff1d(np.arange(len(paraphrases)), idx)
                        add_data_point = {
                                "main": proposition,
                                "proposition": proposition,
                                "paraphrases": paraphrases[other_idx].tolist(),
                                "orig_label": orig_label,
                                "update_label": update_label, # label that model is updated to predict when use_learned_optimizer or update_beliefs
                                "eligible_labels": multiple_labels,
                                "main_eligible_labels": multiple_labels,
                                "main_opt_context" : f"{model_pred} >> {update_label} || {proposition}", # de Cao format
                                'prediction' : model_pred,
                                'id' : data_num,
                            }
                        # add graph viz data
                        if args.do_graph_analysis and f'{args.experiment_name}_flips' in datapoint:
                            add_data_point.update({
                                f'flipped_points' : datapoint[f'{args.experiment_name}_flips'],
                                f'update_pred': datapoint[f'{args.experiment_name}_update_pred']
                            })
                        data_splits[split_name].append(add_data_point)
    if args.dataset == 'LeapOfThought':
        assert args.probing_style=='model'
        identifier = 'shuffled' if not args.leapofthought_main == 'main' else 'combined'
        train_path = os.path.join(args.data_dir, 'LeapOfThought', f'taxonomic_reasonings_training_mix_{identifier}_train.jsonl')
        dev_path = os.path.join(args.data_dir, 'LeapOfThought', f'taxonomic_reasonings_training_mix_{identifier}_dev.jsonl')
        test_path = os.path.join(args.data_dir, 'LeapOfThought', f'taxonomic_reasonings_training_mix_{identifier}_test.jsonl')
        data_splits = {name : [] for name in split_names}
        data_paths = [train_path, dev_path, test_path]
        all_train_main_inputs = []
        for split_name, data_path in zip(split_names, data_paths):
            use_alt_labels = False if (split_name != 'train' and args.update_eval_truthfully) else args.fit_to_alt_labels
            is_train = (split_name == 'train')
            with jsonlines.open(data_path) as file:
                for data_num, datapoint in enumerate(file):
                    if data_num >= (max_train_points if split_name=='train' else max_eval_points):
                        break
                    if 'separate_sentences' in datapoint['metadata']:
                        sentences = datapoint['metadata']['separate_sentences']
                    else:
                        sentences = {}
                    # skip points in training data that don't have metadata (i.e. are knowledge-only, not hypothesis) when using learned opt
                    if is_train and datapoint['context'] == '' and args.use_learned_optimizer:
                        continue
                    if 'implicit_rule' not in sentences: sentences['implicit_rule'] = ""
                    if 'property' not in sentences:  sentences['property'] = ""
                    # typically, just add one data point. the exception is --leapofthought_add_both_for_training true. this combines both cases (1) and (2) below
                    add_point_twice = args.leapofthought_add_both_for_training if split_name=='train' else False
                    if add_point_twice:                                         add_point_conditions = ['implicit_rule', 'hypothesis']    
                    elif args.leapofthought_main == 'implicit_rule':            add_point_conditions = ['implicit_rule']
                    elif args.leapofthought_main == 'hypothesis':               add_point_conditions = ['hypothesis']
                    elif args.leapofthought_main == 'main':                     add_point_conditions = ['main']
                    for condition in add_point_conditions:
                        add_data_point = {}
                        pred_name = f'{args.base_experiment_name}_{args.leapofthought_main}_pred'
                        model_pred = datapoint[pred_name] if pred_name in datapoint else None
                        model_pred_str = ['false', 'true'][int(model_pred)] if model_pred is not None else None
                        if not use_alt_labels:
                            update_label = args.dataset_config['var_for_true']
                            update_label_str = 'true'
                        if use_alt_labels:
                            assert model_pred is not None, "need to write task model predictions to file with --write_preds_to_file true "
                            # want label that belongs to opposite from predicted
                            update_label = [0, 1][1-int(model_pred)]
                            update_label_str = ['false', 'true'][1-int(model_pred)]     
                        # case (0): train contains both hypothesis and implicit_rule as main inputs, for writing graphs to file
                        if condition == 'main':
                            orig_label = datapoint["answer"]
                            is_true = datapoint["answer"] == args.dataset_config['var_for_true']
                            label_str = 'true' if is_true else 'false'
                            main_input = datapoint['main']
                            dependent_proposition = ""
                            add_data_point['label_str'] = label_str
                            add_data_point['orig_label'] = orig_label
                            dependent_label = None
                            if split_name == 'train':
                                all_train_main_inputs.append(main_input)
                            else:
                                assert not main_input in all_train_main_inputs, "found dev/test inputs in train inputs!"
                        # case (1): train and eval on hypothesis+labels for base model
                        if condition == 'hypothesis':
                            orig_label = datapoint["answer"]
                            is_true = datapoint["answer"] == args.dataset_config['var_for_true']
                            label_str = 'true' if is_true else 'false'
                            main_input = datapoint['phrase']
                            dependent_proposition = ""
                            add_data_point['label_str'] = label_str
                            add_data_point['orig_label'] = orig_label
                            dependent_label = None    
                            if split_name == 'train':
                                all_train_main_inputs.append(main_input)
                            else:
                                assert not main_input in all_train_main_inputs, "found dev/test inputs in train inputs!"
                        # case (2) for learned optimizer: use relevant rule and/or property as input, use (hypothesis, y) as entailed statement
                        elif condition == 'implicit_rule':
                            main_input = f"{sentences['implicit_rule']}"
                            # check for train/test overlap for finetuning setting
                            if split_name == 'train':
                                all_train_main_inputs.append(main_input)
                            else:
                                assert not main_input in all_train_main_inputs, "found dev/test inputs in train inputs!"
                            label_str = 'true' 
                            dependent_proposition = datapoint['phrase']
                            add_data_point['label_str'] = label_str
                            add_data_point['orig_label'] = args.dataset_config['var_for_true']
                            # note if training learned optimizer, we only include dependent loss if updating a point to be true
                            dependent_label = datapoint["answer"]
                        add_data_point['proposition'] = main_input
                        add_data_point['main'] = f"{main_input}"
                        add_data_point['dependent_proposition'] = dependent_proposition
                        add_data_point['dependent_proposition_label'] = dependent_label
                        add_data_point['dependent_proposition_orig_label'] = datapoint["answer"]
                        add_data_point['main_opt_context'] = f"{model_pred_str} >> {update_label_str} || {main_input}" # de Cao format
                        add_data_point['update_label'] = update_label
                        add_data_point['implicit_rule'] = sentences['implicit_rule']
                        add_data_point['property'] = sentences['property']
                        add_data_point['id'] = data_num
                        add_data_point['prediction'] = model_pred
                        # add graph viz data
                        if args.do_graph_analysis and f'{args.experiment_name}_flips' in datapoint:
                            add_data_point.update({
                                f'flipped_points' : datapoint[f'{args.experiment_name}_flips'],
                                f'update_pred': datapoint[f'{args.experiment_name}_update_pred']
                            })
                        data_splits[split_name].append(add_data_point)
    if args.dataset == 'Wikidata5m':
        train_path = os.path.join(args.data_dir, 'Wikidata5m', 'filtered_wikidata5m_transductive_train.jsonl')
        dev_path = os.path.join(args.data_dir, 'Wikidata5m', 'filtered_wikidata5m_transductive_dev.jsonl')
        test_path = os.path.join(args.data_dir, 'Wikidata5m', 'filtered_wikidata5m_transductive_test.jsonl')
        # make entity and relation dictionaries
        entity_dict = {}
        relation_dict = {}
        entity_path = os.path.join(args.data_dir, 'Wikidata5m', 'wikidata5m_entity.txt')
        relation_path = os.path.join(args.data_dir, 'Wikidata5m', 'wikidata5m_relation.txt')
        with open(entity_path, 'r') as file:
            for line in file:
                id = line.split()[0]
                entities = [text.strip('\n') for text in line.split('\t')[1:]]
                entity_dict[id] = entities
        with open(relation_path, 'r') as file:
            for line in file:
                id = line.split()[0]
                relations = [text.strip('\n') for text in line.split('\t')[1:]]
                relation_dict[id] = relations
        data_splits = {name : [] for name in split_names}
        data_paths = [train_path, dev_path, test_path]
        for split_name, data_path in zip(split_names, data_paths):
            # first read all the labels to get alt_labels, if need be
            if args.fit_to_alt_labels and not args.beam_search_alt_labels:
                with jsonlines.open(data_path) as file:
                    eligible_alt_labels = np.array([datapoint['entity2'][0] for datapoint in file])
                    n_points = len(eligible_alt_labels)
            use_alt_labels = False if (split_name != 'train' and args.update_eval_truthfully) else args.fit_to_alt_labels
            # now, structure the data to get an entity_dict which gives relations and objects of those relations
            entity_info_dict = {}
            with jsonlines.open(data_path) as file:
                for data_num, datapoint in enumerate(file):
                    e1_str = datapoint['entity1'][0]
                    e2_strs = datapoint['entity2']
                    rel_str = datapoint['relation'][0]
                    e1_strs = datapoint['entity1']
                    rel_strs = datapoint['relation']
                    if e1_str not in entity_info_dict:
                        entity_info_dict[e1_str] = {'e1_synonyms' : e1_strs}
                    if rel_str not in entity_info_dict[e1_str]:
                        entity_info_dict[e1_str][rel_str] = {
                            'e2_strs' : e2_strs,
                            'rel_synonyms' : rel_strs
                        }
                    else:
                        entity_info_dict[e1_str][rel_str]['e2_strs'].append(e2_strs)
            with jsonlines.open(data_path) as file:
                for data_num, datapoint in enumerate(file):
                    if data_num >= (max_train_points if split_name=='train' else max_eval_points):
                        break
                    entities1 = datapoint['entity1']
                    entities2 = datapoint['entity2']
                    relations = datapoint['relation']
                    entity1_idx = 0 if not args.Wikidata5m_use_synonyms else np.random.randint(len(entities1))
                    entity2_idx = 0 if not args.Wikidata5m_use_synonyms else np.random.randint(len(entities2))
                    relation_idx = 0 if not args.Wikidata5m_use_synonyms else np.random.randint(len(relations))
                    e1_key = entities1[0]
                    e1_use = entities1[entity1_idx]
                    rel_key = relations[0]
                    rel_use = relations[relation_idx]
                    pred_name = f'{args.base_experiment_name}_pred'
                    model_pred = datapoint[pred_name] if pred_name in datapoint else None
                    # import pdb; pdb.set_trace()
                    orig_label = entities2[entity2_idx]
                    proposition = f"{e1_use} has relation '{rel_use}' to "
                    if not use_alt_labels:
                        update_label = orig_label
                    if use_alt_labels:
                        assert model_pred is not None, "need to write task model predictions to file with --write_preds_to_file true "
                        if args.beam_search_alt_labels or args.eval_beam_search_alt_labels:
                            # want label that belongs to random choice from alternatives in the data split
                            update_label = np.random.RandomState(seed=args.seed).choice(datapoint[f"{args.base_experiment_name}_alt_preds"])
                        else:
                            # branch alt labels based on correctness
                            correct = metrics.compute_acc_sum(args.probing_style, [model_pred], [orig_label], tokenizer)
                            if correct:
                                eligible_idx = np.setdiff1d(np.arange(n_points), data_num)
                                update_label = np.random.choice(eligible_alt_labels[eligible_idx])
                            if not correct:
                                update_label = orig_label
                    eligible_labels = entities2
                    # get independent statements
                    independent_relations = [rel for rel in entity_info_dict[e1_key].keys() if rel not in [rel_key, 'e1_synonyms']]
                    # add one independent prop per main input
                    assert len(independent_relations) > 0, "missing independent relations from data!"
                    pick_rel = np.random.choice(independent_relations)
                    rel_form = np.random.choice(entity_info_dict[e1_key][pick_rel]['rel_synonyms']) if args.Wikidata5m_use_synonyms else pick_rel
                    independent_propositions = f"{e1_use} has relation '{rel_form}' to "
                    independent_entities2 = entity_info_dict[e1_key][pick_rel]['e2_strs']
                    independent_proposition_labels = np.random.choice(independent_entities2) if args.Wikidata5m_use_synonyms else independent_entities2[0]
                    independent_eligible_labels = independent_entities2
                    # add paraphrases. will do one entity paraphrase and one relation paraphrase
                    if len(entities1) > 1:
                        not_used_e1_idx = np.setdiff1d(np.arange(len(entities1)), entity1_idx)
                        entity_paraphrase_idx = np.random.choice(not_used_e1_idx)
                        entity_paraphrase = f"{entities1[entity_paraphrase_idx]} has relation '{rel_use}' to "
                    else:
                        entity_paraphrase = None
                    if len(relations) > 1:
                        not_used_rel_idx = np.setdiff1d(np.arange(len(relations)), relation_idx)
                        relation_paraphrase_idx = np.random.choice(not_used_rel_idx)
                        relation_paraphrase = f"{e1_use} has relation '{relations[relation_paraphrase_idx]}' to "
                    else:
                        relation_paraphrase = None
                    # get a bunch of combinations, starting with the combination of paraphrases used above (we move this up to first index)
                    max_per_point = args.wikidata_para_per_point
                    paraphrases = [] # should be empty list, not None
                    if len(entities1) > 1 and len(relations) > 1:
                        paraphrases = []
                        combinations = list(itertools.product(not_used_e1_idx, not_used_rel_idx))
                        np.random.shuffle(combinations)
                        add_combination = (entity_paraphrase_idx, relation_paraphrase_idx)
                        remove_comb_index = combinations.index(add_combination)
                        combinations.pop(remove_comb_index)
                        combinations.insert(0, add_combination)
                        for combination in combinations:
                            idx1, idx2 = combination
                            paraphrases.append(f"{entities1[idx1]} has relation '{relations[idx2]}' to ")
                            if len(paraphrases) >= max_per_point:
                                break
                    add_data_point = {
                            "main": proposition,
                            "proposition": proposition,
                            "independent_proposition": independent_propositions,
                            "independent_proposition_label": independent_proposition_labels,
                            "independent_proposition_eligible_labels": independent_eligible_labels,
                            "entity_paraphrase": entity_paraphrase,
                            "entity_paraphrase_eligible_labels" : eligible_labels,
                            "relation_paraphrase" : relation_paraphrase,
                            "relation_paraphrase_eligible_labels" : eligible_labels,
                            "paraphrases": paraphrases,
                            "orig_label": orig_label,
                            "update_label": update_label, # label that model is updated to predict when use_learned_optimizer or update_beliefs
                            "eligible_labels": eligible_labels,
                            "main_eligible_labels": eligible_labels,
                            "main_opt_context" : f"{model_pred} >> {update_label} || {proposition}", # de Cao format
                            'prediction' : model_pred,
                            'id' : data_num,
                            'seen_in_training' : datapoint['seen_in_training'] if 'seen_in_training' in datapoint else 1
                        }
                    # add id in the data of this point's independent proposition -- depends on these data being paired up in data files!!
                    # if 0, it's 1. if 1, it's 0. if 2, it's 3. if 3 it's 2. and so on
                    add_data_point['independent_point_id'] = data_num+1 if data_num % 2 == 0 else data_num-1
                    # add graph viz data
                    if args.do_graph_analysis and f'{args.experiment_name}_flips' in datapoint:
                        add_data_point.update({
                            f'flipped_points' : datapoint[f'{args.experiment_name}_flips'],
                            f'update_pred': datapoint[f'{args.experiment_name}_update_pred']
                        })
                    data_splits[split_name].append(add_data_point)
                    # print("data point")
                    # print(add_data_point['main'])
                    # print(add_data_point['orig_label'])
                    # print("independent statements")
                    # print(add_data_point['independent_proposition'])
                    # print(add_data_point['independent_proposition_label'])
                    # print(add_data_point['paraphrases'])
                    # import pdb; pdb.set_trace()

    # add load paths to args
    args.train_path = train_path
    args.dev_path = dev_path
    args.test_path = test_path

    # preprocess data here rather than during training/eval
    use_collate_fn = PropositionDataset(args, tokenizer, []).collate_fn
    if args.preprocess_data_when_loading:
        for split_name in split_names:
            data_splits[split_name] = [use_collate_fn(item) for item in data_splits[split_name]]
        use_collate_fn = offline_collate_fn

    # make DataLoaders
    train_dataset, dev_dataset, test_dataset = \
        [PropositionDataset(args, tokenizer, data_splits[split_name], split_name=split_name) for split_name in split_names]
    if args.fit_to_dev_data:
        train_dataset = deepcopy(dev_dataset)
        train_dataset.split_name = 'train'
        dev_dataset = deepcopy(test_dataset)
        dev_dataset.split_name = 'dev'
    num_workers = 0 # if not (args.small_data or args.update_small_data) else 0
    train_dataloader  = DataLoader(train_dataset,  shuffle=shuffle_loaders,  collate_fn=use_collate_fn, pin_memory=True, num_workers=num_workers, batch_size=args.train_batch_size)
    dev_dataloader    = DataLoader(dev_dataset,    shuffle=False,            collate_fn=use_collate_fn, pin_memory=True, num_workers=num_workers, batch_size=args.test_batch_size)
    test_dataloader   = DataLoader(test_dataset,   shuffle=False,            collate_fn=use_collate_fn, pin_memory=True, num_workers=num_workers, batch_size=args.test_batch_size)
    return train_dataloader, dev_dataloader, test_dataloader

def str2bool(v):
    # used for boolean argparse values
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class Report():
    """
    Report stores evaluation results during the training process as text files.
    """

    def __init__(self, args, file_path, experiment_name, score_names, overwrite_existing=True):
        self.fn = file_path
        self.args = args
        self.text = ''
        self.max_len = 10
        self.score_names = score_names
        self.old_running_time = 0
        self.curr_speed = 0
        if not os.path.exists(args.report_dir): 
            os.mkdir(args.report_dir)

        # init from existing train report if args.do_train is false and not using off-the-shelf optimizer
        if not overwrite_existing and os.path.exists(file_path):
            self.text = ''.join([line for line in open(file_path, 'r')])
            self.text += '\n\n Evaluaton run:\n'

        # write input arguments at the top
        self.text += 'Input: python %s %s \n\n' % \
                         (sys.argv[0], 
                          ' '.join([arg for arg in sys.argv[1:]]))

        # make header
        header = '%6s |' % 'epoch'
        for n, score_name in enumerate(self.score_names):
            len_name = len(score_name)
            if len_name > self.max_len:
                score_name = score_name[:self.max_len]
            header += (' %10s' % score_name)
            if n < len(score_names) - 1: header += '|'
        self.header = header

        # write header
        self.blank_line = '-'*len(header)
        self.text += self.blank_line + \
                    f"\nTraining report for model: {experiment_name}" + \
                    '\n' + self.blank_line + \
                    '\n'
        self.text += header

    def print_training_prog(self, split_name, stats_dict, batch_num, n_batches, running_time, est_epoch_run_time, forward_time, stats=None):
        if stats is None:
            stats = set([key.replace("train_","").replace("dev_","").replace("test_","") for key in list(stats_dict.keys())])
        last_batch = batch_num == n_batches-1
        print_str = f" {split_name.capitalize():5s} | Batch: {batch_num+1}/{n_batches}"
        for stat in stats:
            key = f"{split_name}_{stat}"
            if key in stats_dict:
                print_str += f" | {stat.capitalize()}: {stats_dict[key]:.2f}"
        curr_speed = running_time - self.old_running_time
        curr_speed = curr_speed if self.curr_speed == 0 else .1*curr_speed + .9*self.curr_speed
        print_str += f" | Runtime: {running_time/60:.1f} min. / {est_epoch_run_time/60:.1f} min."# | Forward time: {forward_time/(batch_num+1):.3f} sec. "#Curr. speed: {curr_speed:.4f}"
        print(print_str, end='\r' if not last_batch else '\n')
        self.old_running_time = running_time
        self.curr_speed = curr_speed

    def write_epoch_scores(self, epoch, scores):
        # write scores
        self.text += '\n%6s ' % str(epoch)
        for n, (name, score) in enumerate(scores.items()):
            if name in self.score_names:
                self.text += '| %10s' % ('%1.2f' % score)
        self.__save()

    def write_final_score(self, args, final_score_str, time_msg):
        self.text += '\n' + self.blank_line
        self.text += '\n%s' % final_score_str
        self.text += '\n' + self.blank_line + '\n\n'
        if time_msg is not None:
            self.text += f'\n{time_msg}\n'
        self._write_all_arguments(args)

        self.__save()

    def write_msg(self, msg):
        self.text += self.blank_line
        self.text += msg
        self.__save()

    def _write_all_arguments(self, args):
        self.text += "\nAll arguments:\n"
        self.text += '\n'.join(['\t' + hp for hp in str(args).replace('Namespace(', '').replace(')', '').split(', ')])        
        self.__save()

    def print_epoch_scores(self, epoch, scores):
        epoch_text = ' %6s ' % 'epoch'
        for n, score_name in enumerate(scores.keys()):
            if score_name in self.score_names:
                len_name = len(score_name)
                if len_name > self.max_len:
                    score_name = score_name[:self.max_len]
                epoch_text += '| %10s' % score_name
        epoch_text += '\n %6s ' % ('%d'% epoch)
        for n, (name, score) in enumerate(scores.items()):
            if name in self.score_names:
                epoch_text += '| %10s' % ('%1.2f' % score)
        print(epoch_text)

    def full_print(self):
        print('\n' + self.text + '\n')

    def job_finished(self):
        self.text += '\n\n Job finished!'
        self.__save()

    def __save(self):
        if self.fn is not None:
            with open(self.fn, mode='w') as text_file:
                text_file.write(self.text)
