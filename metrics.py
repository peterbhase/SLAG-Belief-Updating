import torch
import numpy as np
import itertools
import utils
from scipy import stats

def get_metrics_on_named_data(args, grad_req, model, tokenizer, batch, data_name,
                              keep_idx=None, exclude_idx=None, reference_preds=None) -> dict:
    '''
    given a batch of data returned by utils.PropositionDataset.collate_fn, extract data of the data_name and
    compute the named metrics in metrics on this data
    force_labels forces compute_acc_sum to use the provided labels rather than any automatically associated with the data
    returns the statisticts in a dictionary
    '''
    assert data_name in ['main', 'paraphrases', 'dependent_proposition', 'independent_proposition', 'entity_paraphrase', 'relation_paraphrase'],\
        "invalid data_name requested in get_metrics_on_named_data"
    # make universal keep_idx here
    if keep_idx is not None or exclude_idx is not None:
        if data_name != 'paraphrases':
            whole_batch_size = len(batch[f'{data_name}_input_ids']) 
        else:
            whole_batch_size = len(batch['concatenated_paraphrases']['input_ids']) if batch['concatenated_paraphrases'] is not None else 0
        keep_idx = keep_idx if keep_idx is not None else utils.arraydiff1d(np.arange(whole_batch_size), np.array(exclude_idx))
    with grad_req and torch.cuda.amp.autocast(enabled=args.fp16):
        # get metrics for paraphrases
        if data_name == 'paraphrases':
            assert reference_preds is not None, "must provide reference outputs with paraphrases, either main_outputs or update_outputs"
            orig_labels = batch['orig_labels'] if args.probing_style!='seq2seq' else [item['eligible_labels'] for item in batch['text_data']] 
            metrics = compute_paraphrase_metrics_batched(args, model, batch, keep_idx, tokenizer, main_preds=reference_preds, labels=orig_labels)
        # get metrics for named data propositions
        if data_name in ['main', 'dependent_proposition', 'independent_proposition', 'entity_paraphrase', 'relation_paraphrase']:
            input_kwargs = {k.replace(f"{data_name}_","") : v for k,v in batch.items() if data_name in k}
            if data_name == 'main':
                input_kwargs.update({'orig_labels' : batch['orig_labels']})
            if data_name == 'dependent_proposition':
                input_kwargs.update({'orig_labels' : batch['dependent_proposition_orig_labels']})
            # figure out where there are none points, translate the keep_idx to reflect this
            orig_batch_size = input_kwargs['input_ids'].size(0)
            if keep_idx is not None:
                where_not_none = np.argwhere([item[data_name] is not None for item in batch['text_data']]).reshape(-1)
                old_idx_to_new_idx = {k:v for k,v in zip(where_not_none, range(orig_batch_size))}
                keep_idx = np.array([old_idx_to_new_idx[idx] for idx in keep_idx if idx in old_idx_to_new_idx])
                if len(keep_idx) == 0: 
                    return {'acc_sum' : 0, 'n_points' : 0, 'binary_correct' : [], 'where_correct' : []}
                input_kwargs = utils.slice_kwargs(input_kwargs, keep_idx)
            if args.use_learned_optimizer: 
                input_kwargs['use_base_model'] = True
            utils.move_kwargs_to_gpu(input_kwargs)
            with torch.no_grad() and torch.cuda.amp.autocast(enabled=args.fp16):
                outputs = model(is_eval=True, **input_kwargs)
            input_preds = outputs['preds']
            # define labels based on whether problem is seq2seq. filter based on keep_idx
            if args.probing_style=='seq2seq':
                labels = [item[f'{data_name}_eligible_labels'] for item in batch['text_data']] 
                if keep_idx is not None:
                    labels = [label for idx, label in enumerate(labels) if idx in keep_idx]
            else:
                labels = input_kwargs['labels']
            n_correct, binary_correct = compute_acc_sum(args.probing_style, input_preds, labels, tokenizer, return_where_correct=True)
            metrics = {
                'acc_sum' : n_correct,
                'n_points' : len(binary_correct),
                'binary_correct' : binary_correct,
                'where_correct' : np.argwhere(binary_correct).reshape(-1),
                'preds' : input_preds,
                'model_outputs' : outputs,
            }
            if 'orig_labels' in input_kwargs:
                n_correct, binary_correct = compute_acc_sum(args.probing_style, input_preds, input_kwargs['orig_labels'], tokenizer, return_where_correct=True)
                metrics.update({
                    'orig_acc_sum' : n_correct,
                    'orig_binary_correct' : binary_correct,
                    'orig_where_correct' : np.argwhere(binary_correct).reshape(-1),
                })
            if reference_preds is not None:
                n_correct, binary_correct = compute_acc_sum(args.probing_style, input_preds, reference_preds, tokenizer, return_where_correct=True)
                metrics.update({
                    'ref_acc_sum' : n_correct,
                    'ref_binary_correct' : binary_correct,
                    'ref_where_correct' : np.argwhere(binary_correct).reshape(-1),
                })
        return metrics

def pull_independent_acc_when_wrong(main_correctness):
    # should be exclusively used to get independent data accuracy from the main predictions on wikidata5m, where data is paired in adjacent positions
    assert len(main_correctness) % 2 == 0, "correctness indicators must be even, since they must be paired in adjacent positions"
    ind_acc_sum = 0
    num_wrong = 0
    iter_idx = [(end-2,end) for end in range(2, len(main_correctness)+1, 2)]
    for (start, end) in iter_idx:
        pair = main_correctness[start:end]
        for pair_idx in range(2):
            if pair[pair_idx] == 0:
                ind_acc_sum += pair[1-pair_idx]
                num_wrong += 1
    return ind_acc_sum / num_wrong

def jaccard_similarity(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union

def print_dependency_metrics(knowledge1, consequent1, knowledge2=None):
    # compute statistics for relationships between knowledge and their consequents
    # returns str for printing
    knowledge1 = np.array(knowledge1)
    consequent1 = np.array(consequent1)
    if knowledge2 is None or len(knowledge2) == 0:
        k1_alone_table = contingency_table(knowledge1, consequent1) # cell for array1==0 and array2==1 from binary contingency table
        print(f" Belief 1 correct      : {k1_alone_table[1,1]:.2f} ({100*np.mean(knowledge1==1):.2f}% of data)")
        print(f" Belief 1 incorrect    : {k1_alone_table[0,1]:.2f} ({100*np.mean(knowledge1==0):.2f}% of data)")
        k1stats = stats.pearsonr(knowledge1, consequent1)
        print(f"  correlation: {k1stats[0]:.3f}")# (p-value: {k1stats[1]:.4f})")
    else:
        knowledge2 = np.array(knowledge2)
        Andk1k2 = knowledge1 * knowledge2
        Ork1k2 = np.max(np.stack((knowledge1, knowledge2)), axis=0)
        Xork1k2 = np.sum(np.stack((knowledge1, knowledge2)), axis=0) == 1
        and_prop_one = contingency_table(Andk1k2, consequent1)[1,1] # cell for array1==1 and array2==1 from binary contingency tabl
        xor_prop_one = contingency_table(Xork1k2, consequent1)[1,1] # cell for array1==1 and array2==1 from binary contingency tablee
        or_prop_one = contingency_table(Ork1k2, consequent1)[0,1] # cell for array1==0 and array2==1 from binary contingency table
        k1_alone_table = contingency_table(knowledge1, consequent1) # cell for array1==0 and array2==1 from binary contingency table
        k2_alone_table = contingency_table(knowledge2, consequent1) # cell for array1==0 and array2==1 from binary contingency table
        print(f" Both beliefs correct  : {and_prop_one:.2f} ({100*np.mean(Andk1k2==1):.2f}% of data)")
        print(f" One belief correct    : {xor_prop_one:.2f} ({100*np.mean(Xork1k2==1):.2f}% of data)")
        print(f" Neither belief correct: {or_prop_one:.2f} ({100*np.mean(Ork1k2==0):.2f}% of data)")
        print(f" Belief 1 correct      : {k1_alone_table[1,1]:.2f} ({100*np.mean(knowledge1==1):.2f}% of data)")
        print(f" Belief 1 incorrect    : {k1_alone_table[0,1]:.2f} ({100*np.mean(knowledge1==0):.2f}% of data)")
        k1stats = stats.pearsonr(knowledge1, consequent1)
        print(f"  correlation: {k1stats[0]:.3f}")# (p-value: {k1stats[1]:.4f})")
    return

def get_proportions_discrete(array):
    props = {k : sum(array==k) / len(array) for k in set(array)}
    props = [(k, round(prop, 2)) for k, prop in props.items()]
    props = sorted(props, key = lambda x : x[0])
    return props

def contingency_table(array1, array2):
    # return contingency table with axis1 : [0,1] and axis2: [0,1] for binary arrays array1 and array2
    # table is row-normalized
    table = np.zeros((2,2))
    table[0,0] = np.sum((array1==0)*(array2==0))
    table[0,1] = np.sum((array1==0)*(array2==1))
    table[1,0] = np.sum((array1==1)*(array2==0))
    table[1,1] = np.sum((array1==1)*(array2==1))
    table[0,:] /= sum(table[0,:])
    table[1,:] /= sum(table[1,:])
    table = np.round(100*table, 2)
    return table

def safe_seq(seq):
    # filter to non -100 values in seq, which is a list. -100 is the default ignore_index in pytorch
    return [x for x in seq if x >= 0]

def standardize_preds_or_labels(probing_style, input, tokenizer):
    # input should be list, 1-d np.array, or 1-d torch.tensor of ints or strs
    # returns input formatted into 1-d np array, decoding for encoded inputs using the tokenizer
    if type(input) is list and type(input[0]) is torch.Tensor or type(input[0]) is np.ndarray:
        input = [item.tolist() for item in input]
    if type(input) is not list:
        input = input.tolist()
    if type(input) is torch.Tensor and input.dim() == 0:
        input = input.view(1)
    if type(input) in [int, torch.int, str, np.str_]:
        input = [input]
    if probing_style == 'seq2seq':
        # decode if elements are not already strings, or lists of strings (which would suggest it had been decoded already)
        decode = not (type(input[0]) is str or type(input[0]) is np.str_ or (type(input) is list and type(input[0][0]) is str))
        if decode:
            input = [tokenizer.decode(safe_seq(seq), skip_special_tokens=True) for seq in input]
        if type(input[0]) in [str, np.str_]:
            input = [x.lower().strip() for x in input]
        if type(input[0]) is list:
            input = [[x.lower().strip() for x in eligible_labels] for eligible_labels in input]
    if type(input) is torch.Tensor:
        input = input.detach().cpu().numpy()
    elif type(input) is list and type(input[0]) is list:
        input = input # skip the array formatting here as it will not be used in downstream metrics
    else:
        input = np.array(input)
    return input

def force_not_dimensionless(input):
    if type(input) is torch.Tensor:
        if input.dim()==0:
            input = input.view(1)
    return input
    
def get_num_corrupted(probing_style, before_preds, after_preds, labels, tokenizer):
    assert len(before_preds) == len(after_preds) and len(after_preds) == len(labels), "unequal lengths of inputs in get_num_corrupted"
    before_preds = standardize_preds_or_labels(probing_style, before_preds, tokenizer)
    after_preds = standardize_preds_or_labels(probing_style, after_preds, tokenizer)
    labels = standardize_preds_or_labels(probing_style, labels, tokenizer)
    right_before_wrong_after = (before_preds==labels) * (after_preds!=labels)
    n_corrupted = np.sum(right_before_wrong_after)
    n_right_before = np.sum(before_preds==labels)
    return n_corrupted, n_right_before

def compute_acc_sum(probing_style, preds, labels, tokenizer, return_where_correct=False):
    # preds and labels should be list, 1-d np.array, or 1-d torch.tensor of ints or strs
    # eligible_labels is list of lists of string labels, which are all valid answers
    # returns number correct, and optionally the array of 1/0 correctness indicators
    preds = force_not_dimensionless(preds) # dimensionless happens when using one_d_tensor[int] slicing
    labels = force_not_dimensionless(labels)
    if len(preds)==0:
        return 0
    if labels is not None and len(labels) == 0:
        return 0
    preds = standardize_preds_or_labels(probing_style, preds, tokenizer)
    labels = standardize_preds_or_labels(probing_style, labels, tokenizer)
    assert len(preds) == len(labels), "len of preds and labels not equal"
    many_eligible_labels = type(labels[0]) is list
    if not many_eligible_labels:
        binary_correct = preds==labels
    else:
        binary_correct = np.array([pred in eligible_labels for pred, eligible_labels in zip(preds, labels)])
    acc_sum = np.sum(binary_correct)
    if not return_where_correct:
        return acc_sum
    else:
        return (acc_sum, binary_correct)

def get_number_matching_pairs(preds_list):
    '''
    preds : list of lists preds (outer list is unique point, inner list is # paraphrases of that point)
    computes the number of matching pairs of preds, and the number of total pairs
    returns (number_matching, number_total_pairs)
    '''
    number_total_pairs = 0
    number_matching = 0
    for preds in preds_list:
        pairs = list(itertools.combinations(preds, 2))
        number_total_pairs += len(pairs)
        for pair in pairs:
            if pair[0] == pair[1]:
                number_matching += 1
    return number_matching, number_total_pairs

def compute_paraphrase_metrics(args, model, paraphrases, tokenizer, main_preds, labels=None):
    num_paraphrases_sum = 0
    num_par_eq_comparisons = 0
    all_paraphrase_preds = []
    all_paraphrase_labels = []
    all_combined_preds = []
    orig_preds = standardize_preds_or_labels(args.probing_style, main_preds, tokenizer)
    labels = standardize_preds_or_labels(args.probing_style, labels, tokenizer)
    for data_id, paraphrase_kwargs in enumerate(paraphrases):
        if paraphrase_kwargs is not None:
            num_paraphrases = paraphrase_kwargs['input_ids'].size(0)
            num_paraphrases_sum += num_paraphrases
            num_par_eq_comparisons += (num_paraphrases - 1)
            if labels is None:
                all_paraphrase_labels.extend(paraphrase_kwargs['labels'].tolist())    
            else:
                extend_labels = [labels[data_id]] * num_paraphrases
                all_paraphrase_labels.extend(extend_labels)    
            with torch.no_grad():
                paraphrase_outputs = model(is_eval=True, **paraphrase_kwargs)
            paraphrase_preds = [pred.lower().strip() for pred in paraphrase_outputs['preds']]
            all_paraphrase_preds.extend(paraphrase_preds)
            combined_preds = paraphrase_preds + [orig_preds[data_id]]
        else:
            combined_preds = []
        all_combined_preds.append(combined_preds)
    n_consistent, n_paraphrase_pairs = get_number_matching_pairs(all_combined_preds)
    return {
        'par_acc_sum' : compute_acc_sum(args.probing_style, all_paraphrase_preds, all_paraphrase_labels, tokenizer),
        'n_paraphrases' :  num_paraphrases_sum,
        'n_par_eq_comparisons' : num_par_eq_comparisons,
        'n_consistent' :  n_consistent,
        'n_paraphrase_pairs' :  n_paraphrase_pairs,
        'all_paraphrase_preds' : all_paraphrase_preds,
        'all_paraphrase_labels' : all_paraphrase_labels,
    }


def compute_paraphrase_metrics_batched(args, model, batch, keep_idx, tokenizer, main_preds, labels):
    '''
    computes statistics of model performance on paraphrases in a batched manner for efficiency
    - acc: preds on paras equal to real labels
    - eq: preds on paras equal to preds on main inputs
    - cons: preds on paras equal to one another
    '''
    all_paraphrase_preds = []
    all_paraphrase_labels = []
    all_main_preds_as_labels = [] # adding this in cases where we want another set of labels, like new model preds vs. orig labels
    all_combined_preds = []
    non_flat_all_paraphrase_preds = [] # used to get point-level par eq stats
    non_flat_all_main_preds_as_labels = []
    main_preds = standardize_preds_or_labels(args.probing_style, main_preds, tokenizer)
    labels = standardize_preds_or_labels(args.probing_style, labels, tokenizer)
    list_of_paraphrases = batch['paraphrases']
    num_paraphrases_per_point = np.array([paraphrases['input_ids'].size(0) if paraphrases is not None else 0 for paraphrases in list_of_paraphrases])
    all_paraphrases = batch['concatenated_paraphrases']
    # iterate through keep_idx, and add the corresponding indices to an overall idx list
    if keep_idx is None: keep_idx = np.arange(len(num_paraphrases_per_point))
    paraphrase_keep_idx = []
    if all_paraphrases is None: # happens when no dev points have paraphrases
        return {
            'par_acc_sum' : 0,
            'par_eq_sum' : 0,
            'n_paraphrases' :  0,
            'n_consistent' :  0,
            'n_paraphrase_pairs' :  0,
            'all_paraphrase_preds' : [np.nan] * len(keep_idx),
            'all_paraphrase_labels' : [np.nan] * len(keep_idx),
            'point_level_cons' : None,
            'point_level_par_eqs' : None,
        }
    for idx in keep_idx:
        this_point_idx_start = sum(num_paraphrases_per_point[:idx])
        this_point_idx_end = this_point_idx_start + num_paraphrases_per_point[idx]
        paraphrase_keep_idx.extend(list(range(this_point_idx_start, this_point_idx_end)))
    all_paraphrases = utils.slice_kwargs(all_paraphrases, np.array(paraphrase_keep_idx))
    num_paraphrases_per_point_after_filtering = [num for idx, num in enumerate(num_paraphrases_per_point) if idx in keep_idx]
    num_paraphrases_total = all_paraphrases['input_ids'].size(0)
    utils.move_kwargs_to_gpu(all_paraphrases)
    # repeat/extend labels for each point
    assert len(main_preds) in [len(keep_idx), len(labels)], "length of main_preds does not match keep_idx or labels -- must match one of these"
    if num_paraphrases_total > 0:
        for _keep_idx, idx in enumerate(keep_idx):
            num_for_point = num_paraphrases_per_point[idx]
            if num_for_point > 0:
                extend_labels = [labels[idx]] * num_paraphrases_per_point[idx]
                all_paraphrase_labels.extend(extend_labels)
            # add main_preds back as labels for par_eq_sum computation. main_preds length either matches labels or equal to length of keep_idx
            main_idx = _keep_idx if len(main_preds) == len(keep_idx) else idx
            extend_labels = [main_preds[main_idx]] * num_paraphrases_per_point[idx]
            all_main_preds_as_labels.extend(extend_labels)
            if num_for_point > 0:
                non_flat_all_main_preds_as_labels.append(extend_labels)
        for batch_kwargs in utils.kwargs_into_batches(all_paraphrases, batch_size=args.test_batch_size): 
            paraphrase_outputs = model(is_eval=True, **batch_kwargs)
            paraphrase_preds = standardize_preds_or_labels(args.probing_style, paraphrase_outputs['preds'], tokenizer)
            all_paraphrase_preds.extend(paraphrase_preds.tolist())
        assert len(all_paraphrase_preds) == len(paraphrase_keep_idx), "note paraphrase preds must be as many as paraphrase_keep_idx"
        # now break up the predictions based on which go with which data points. add the original preds from main_preds here as well, for computing consistency
        # note we index num_paraphrases_per_point_after_filtering with _keep_idx, since this will match the ordering that preds were put into all_paraphrase_preds
        for _keep_idx, idx in enumerate(keep_idx):
            # from keep_idx to idx in filtered data
            num_for_point = num_paraphrases_per_point[idx]
            main_idx = _keep_idx if len(main_preds) == len(keep_idx) else idx
            if num_for_point > 0:
                this_point_idx_start = sum(num_paraphrases_per_point_after_filtering[:_keep_idx])
                this_point_idx_end = this_point_idx_start + num_paraphrases_per_point_after_filtering[_keep_idx]
                this_point_preds = all_paraphrase_preds[this_point_idx_start:this_point_idx_end]
                preds_with_orig_preds = this_point_preds + [main_preds[main_idx]]
                all_combined_preds.append(preds_with_orig_preds)
                non_flat_all_paraphrase_preds.append(this_point_preds)
    # n_consistent, n_paraphrase_pairs = get_number_matching_pairs(all_combined_preds)
    # make point level consistency stats
    point_level_cons_stats = [(n_cons, n_pairs) for (n_cons, n_pairs) in [get_number_matching_pairs([combined_preds]) for combined_preds in all_combined_preds]]
    pair_threshold = 0 # only count cons for data where there is more than x pair of points
    n_consistent = sum([n_cons for n_cons, n_pairs in point_level_cons_stats if n_pairs > pair_threshold])
    n_paraphrase_pairs = sum([n_pairs for n_cons, n_pairs in point_level_cons_stats if n_pairs > pair_threshold])
    point_level_cons = [n_cons/n_pairs if n_pairs > pair_threshold else None for (n_cons, n_pairs) in point_level_cons_stats]
    # make point_level par eqs
    point_level_par_eqs = [
        compute_acc_sum(args.probing_style, paraphrase_preds, main_preds_as_labels, tokenizer, return_where_correct=True)[1]
        for paraphrase_preds, main_preds_as_labels in zip(non_flat_all_paraphrase_preds, non_flat_all_main_preds_as_labels)
    ]
    assert not any([len(combined)==0 for combined in all_combined_preds]), "there are combined_preds with single preds per point in all_combined_preds"
    return {
        'par_acc_sum' : compute_acc_sum(args.probing_style, all_paraphrase_preds, all_paraphrase_labels, tokenizer),
        'par_eq_sum' : compute_acc_sum(args.probing_style, all_paraphrase_preds, all_main_preds_as_labels, tokenizer),
        'n_paraphrases' :  num_paraphrases_total,
        'n_consistent' :  n_consistent,
        'n_paraphrase_pairs' :  n_paraphrase_pairs,
        'all_paraphrase_preds' : all_paraphrase_preds,
        'all_paraphrase_labels' : all_paraphrase_labels,
        'point_level_cons' : point_level_cons,
        'point_level_par_eqs' : point_level_par_eqs
    }


    