'''
file contains learned optimizer class and wrapper for Probe class, as well as some support functions
'''

import numpy as np
from math import e
import torch
from higher.patch import monkeypatch as make_functional
from torch import nn
from copy import deepcopy
import time
from allennlp.modules.feedforward import FeedForward
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper
from higher.patch import monkeypatch as make_functional
import metrics
import utils

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1, ignore_index=-100):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim
        self.ignore_index=ignore_index

    def forward(self, pred, target, use_smoothing=True):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # get ignore_index positions then replace this with 0 -- will ignore this later
            where_ignore = torch.where((target==self.ignore_index).view(-1))[0]
            target.masked_fill_(target==self.ignore_index, 0)
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
            true_dist[where_ignore,:] = 0 # set to 0 to ignore this term in the sum below
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

class ConditionedParameter(torch.nn.Module):
    def __init__(self, parameter, condition_dim=1024, hidden_dim=128, max_scale=1):
        super().__init__()
        self.parameter_shape = parameter.shape

        if len(self.parameter_shape) == 2:
            self.conditioners = torch.nn.Sequential(
                torch.nn.utils.weight_norm(torch.nn.Linear(condition_dim, hidden_dim)),
                torch.nn.Tanh(),
                torch.nn.utils.weight_norm(
                    torch.nn.Linear(
                        hidden_dim, 2 * (parameter.shape[0] + parameter.shape[1]) + 1
                    )
                ),
            )
        elif len(self.parameter_shape) == 1:
            self.conditioners = torch.nn.Sequential(
                torch.nn.utils.weight_norm(torch.nn.Linear(condition_dim, hidden_dim)),
                torch.nn.Tanh(),
                torch.nn.utils.weight_norm(
                    torch.nn.Linear(hidden_dim, 2 * parameter.shape[0] + 1)
                ),
            )
        else:
            raise RuntimeError()

        self.max_scale = max_scale

    def forward(self, inputs, grad):

        if len(self.parameter_shape) == 2:
            (
                conditioner_cola,
                conditioner_rowa,
                conditioner_colb,
                conditioner_rowb,
                conditioner_norm,
            ) = self.conditioners(inputs).split(
                [
                    self.parameter_shape[1],
                    self.parameter_shape[0],
                    self.parameter_shape[1],
                    self.parameter_shape[0],
                    1,
                ],
                dim=-1,
            )

            a = conditioner_rowa.softmax(-1).T @ conditioner_cola
            b = conditioner_rowb.softmax(-1).T @ conditioner_colb

        elif len(self.parameter_shape) == 1:
            a, b, conditioner_norm = self.conditioners(inputs).split(
                [self.parameter_shape[0], self.parameter_shape[0], 1], dim=-1
            )
        else:
            raise RuntimeError()

        new_grad = (self.max_scale
            * conditioner_norm.sigmoid().squeeze()
            * (grad * a.squeeze() + b.squeeze()))

        return new_grad


class LSTMConditioner(torch.nn.Module):
    def __init__(
        self,
        vocab_dim=30522,
        embedding_dim=768,
        hidden_dim=256,
        output_dim=1024,
        embedding_init=None,
    ):
        super().__init__()
        self.embedding = torch.nn.Embedding(
            num_embeddings=vocab_dim,
            embedding_dim=embedding_dim,
            padding_idx=0,
            _weight=embedding_init,
        )
        self.lstm = PytorchSeq2VecWrapper(
            torch.nn.LSTM(
                input_size=embedding_dim,
                hidden_size=hidden_dim,
                num_layers=1,
                bidirectional=True,
                batch_first=True,
            )
        )
        self.linear = FeedForward(
            input_dim=hidden_dim * 2,
            num_layers=1,
            hidden_dims=[output_dim],
            activations=[torch.nn.Tanh()],
        )

    def forward(self, inputs, masks):
        return self.linear(self.lstm(self.embedding(inputs), masks))


class HyperNetwork(torch.nn.Module):
    def __init__(
        self,
        model,
        vocab_dim=30522,
        embedding_dim=768,
        hidden_dim=128,
        condition_dim=1024,
        include_set={},
        max_scale=1e-3,
        embedding_init=None,
    ):
        super().__init__()

        self.param2conditioner_map = {
            n: "{}_conditioner".format(n).replace(".", "_")
            for n, p in model.named_parameters()
            if n in include_set
        }

        self.conditioners = torch.nn.ModuleDict(
            {
                self.param2conditioner_map[n]: ConditionedParameter(
                    p,
                    condition_dim,
                    hidden_dim,
                    max_scale=max_scale,
                )
                for n, p in model.named_parameters()
                if n in include_set
            }
        )

        self.condition = LSTMConditioner(
            vocab_dim,
            embedding_dim,
            hidden_dim,
            condition_dim,
            embedding_init=embedding_init,
        )

    def forward(self, inputs, masks, grads=None):
        condition = self.condition(inputs, masks)
        return {
            p: self.conditioners[self.param2conditioner_map[p]](
                condition,
                grad=grads[p] if grads else None,
            )
            for p, c in self.param2conditioner_map.items()
        }


class ModelWithLearnedOptimizer(nn.Module):

    def __init__(self, args, probe, tokenizer):
        super().__init__()
        self.model = probe
        self.args = args
        self.hparams = args
        self.tokenizer = tokenizer
        self.ignore_index=-100
        self.hparams.eps = 0.1
        self.hparams.margin_kl_max = 1e-3
        self.hparams.margin_kl_min = 1e-5
        self.hparams.margin_lp_max = 1e-3
        self.hparams.margin_lp_min = 1e-7
        self.hparams.p = 2
        self.hparams.max_scale = 1
        self.hparams.use_views = True
        self.successive_updates_counter = 0 # used to keep track of successive_updates currently, to never exceed self.args.num_successive_updates
        self.running_update_dict = None # used with successive_updating
        embeds = self.model.model.embeddings.word_embeddings.weight.data if args.probing_style != 'seq2seq' else self.model.model.model.shared.weight.data
        self.learner = HyperNetwork(
            self.model,
            vocab_dim= self.tokenizer.vocab_size,
            embedding_dim=self.model.model.config.hidden_size,
            hidden_dim=128,
            condition_dim=1024,
            include_set={
                n
                for n, _ in self.model.named_parameters()
                if all(
                    e not in n.lower()
                    for e in (
                        "bias",
                        "norm",
                        "embeddings",
                        "classifier",
                        "pooler",
                        "shared",
                        "embed",
                        "positions",
                    )
                )
            },
            max_scale=self.hparams.max_scale,
            embedding_init=embeds
        )
        # alpha_kl used when self.args.implementation == 'de_cao'
        self.alpha_kl = torch.nn.Parameter(torch.ones(()))
        self.alpha_kl.register_hook(lambda grad: -grad)
        self.register_buffer("margin_kl", torch.tensor(self.hparams.margin_kl_max))
        if args.probing_style == 'seq2seq' and self.args.implementation == 'de_cao':
            self.loss = LabelSmoothingLoss(classes=tokenizer.vocab_size, smoothing=.1, ignore_index=-100)
        else:
            self.loss = nn.CrossEntropyLoss(ignore_index=-100)
        self.CE_loss = nn.CrossEntropyLoss(ignore_index=-100)
        self.kl_loss = nn.KLDivLoss(reduction='batchmean', log_target=True)

        # for param_name in self.learner.conditioners.keys():
        #     param = self.learner.conditioners[param_name]
        #     sum_sizes = sum([np.prod(p.shape) for p in param.parameters()])
        #     print(f"{param_name:50s} : {sum_sizes:8d}")
        #     print(f"     orig size: {np.prod(param.parameter_shape):30d}")
        
    def get_updated_model(self):
        assert self.running_update_dict is not None, "asking for updating model without any updates performed"
        new_params = {n : self.running_update_dict.get(n, 0) + p for n, p in self.model.named_parameters()}
        updated_model = deepcopy(self.model).eval()
        updated_model.load_state_dict(new_params, strict=False)
        return updated_model

    def compute_param_update(self, logits, single_point_kwargs, current_update_dict=None):
        # compute the original model's grad on a single point given the logits predicted for that point, or use use_model if it is supplied
        # returns params_dict of form {name : param_update} for name, param in model.named_parameters()
        if self.args.probing_style == 'seq2seq':
            labels = single_point_kwargs['labels'][:,1:]
            loss = torch.nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), labels.reshape(-1))
        else:
            labels = single_point_kwargs['labels'] 
            loss = torch.nn.CrossEntropyLoss()(logits, labels)
        grads = torch.autograd.grad(
            loss,
            self.model.parameters(),
            allow_unused=True,
        )
        grads = {
            name: grad
            for (name, _), grad in zip(self.model.named_parameters(), grads)
        }
        params_dict = self.learner(
            single_point_kwargs["opt_context_input_ids"],
            single_point_kwargs["opt_context_attention_mask"],
            grads=grads,
        )
        if current_update_dict is not None:
            # split by whether we detach prev updates. this determines whether backprop of Loss at update step T is wrt all T hypernetwork outputs or just the most recent output
            # detaching is default behavior
            if self.args.detach_prev_updates:
                params_dict = {n : (params_dict[n] + p.detach()) for n, p in current_update_dict.items()}
            else:
                params_dict = {n : (params_dict[n] + p) for n, p in current_update_dict.items()}
        return params_dict

    def forward(self, is_eval=False, return_new_model=False, update_steps=None, *args, **kwargs):
        '''
        if kwargs['use_base_model'] is true, this simply applies self.model(**kwargs)
        otherwise, this function runs the model forward pass, gets the gradient, and applies the learned optimizer to get
        a model w/ updated parameters, then uses that for the final forward pass.

        if batch_size > 1, randomly select a point to get the updated model on, and use the rest to compute the KL term
        note that labels is main_labels from main.py, which could be real labels or alt labels. orig_labels are always real labels
        '''
        # return original model's output if requested
        if ('use_base_model' in kwargs and kwargs['use_base_model']):
            return self.model.eval()(is_eval=is_eval, *args, **kwargs)
        # get local variables, including number of updates
        batch_size = kwargs['input_ids'].size(0)
        using_single_point = (batch_size == 1)
        max_num_updates = self.args.learned_opt_steps if update_steps is None else update_steps
        # first perform a forward pass with the entire batch
        with torch.enable_grad():
            fmodel = make_functional(self.model).eval()
            input_kwargs = {k:v for k,v in kwargs.items() if k in ['input_ids', 'attention_mask', 'labels', 'decoder_input_ids']}
            if self.running_update_dict is not None: # occurs after num_successive_updates many previous updates (see main.py and self.reset_successive_update_vars)
                init_use_params = [self.running_update_dict.get(n, 0) + p for n, p in self.model.named_parameters()]
            else:
                init_use_params = [p for n, p in self.model.named_parameters()]
            orig_outputs = fmodel(
                is_eval=False,
                **input_kwargs,
                use_cache=False,
                params=init_use_params
            )
            all_orig_logits = orig_outputs['logits']
            preds = all_orig_logits.argmax(-1)
            # if batch size is 1, have to update with that point
            if batch_size == 1:
                eligible_idx = [0]
            # pick out a wrong point for getting model grad if fitting to wrong points. if none available, will selected random point
            elif self.args.fit_to_wrong_points:
                _, binary_correct = metrics.compute_acc_sum(self.args.probing_style, preds, kwargs['orig_labels'], self.tokenizer,
                                                                      return_where_correct=True)
                where_wrong = np.argwhere(1-binary_correct).reshape(-1)
                eligible_idx = where_wrong if len(where_wrong) > 0 else np.arange(batch_size)
            # otherwise, pick a random point for computing model grad. IF fitting to paraphrases, favor train point that has paraphrases
            else:
                if not (self.args.fit_opt_to_paraphrases or self.args.fit_opt_to_dependent_propositions):
                    eligible_idx = np.arange(batch_size)
                # elif self.args.fit_opt_to_paraphrases:
                #     eligible_idx = np.arange(batch_size)
                elif self.args.fit_opt_to_paraphrases: # this oversamples points that have paraphrases. does slightly better than picking random point
                    has_paraphrases = np.argwhere([paraphrase is not None for paraphrase in kwargs['paraphrases']]).reshape(-1)
                    does_not_have_paraphrases = np.setdiff1d(np.arange(batch_size), has_paraphrases)
                    not_all_points_have_paraphrases = len(does_not_have_paraphrases) > 0
                    if not_all_points_have_paraphrases:
                        chance_to_pick_with_paraphrases = .7
                        if np.random.random() < chance_to_pick_with_paraphrases and len(has_paraphrases) > 0:
                            eligible_idx = has_paraphrases
                        else:
                            eligible_idx = does_not_have_paraphrases
                    else:
                        eligible_idx = has_paraphrases
                elif self.args.fit_opt_to_dependent_propositions: # this oversamples points that have dependent data. does slightly better than picking random point
                    # a few of these details are peculiar to LeapOfThought data and need not be general principles
                    has_dependent_prop = np.argwhere([not indicator for indicator in kwargs['nan_dependent_proposition_labels']]).reshape(-1)
                    eligible_idx = has_dependent_prop if len(has_dependent_prop) > 0 else np.arange(batch_size)
                    update_to_true_idx = np.argwhere(kwargs['labels'].cpu().numpy()).reshape(-1)
                    chance_to_pick_true_update = .5
                    if np.random.random() < chance_to_pick_true_update and len(update_to_true_idx) > 0:
                        eligible_idx = update_to_true_idx
            # get single point, its logits, and other points and their logits
            use_for_model_grad_idx = np.random.choice(eligible_idx, size=1)
            other_point_idx = np.setdiff1d(np.arange(batch_size), use_for_model_grad_idx)
            single_point = utils.select_point_from_kwargs(kwargs, use_for_model_grad_idx)
            single_point_logits = orig_outputs['logits'][use_for_model_grad_idx,...]
            # detach orig logits for backprop purposes (used in computing KL between old and new inputs, and CE with targets)
            other_orig_logits = orig_outputs['logits'][other_point_idx,...].detach()

            # get pred on ind data if fitting to ind data
            if self.args.fit_opt_to_independent_propositions:
                independent_kwargs = {k.replace("independent_proposition_","") : v for k,v in kwargs.items() if 'independent_proposition' in k}
                single_independent_kwargs = utils.slice_kwargs(independent_kwargs, use_for_model_grad_idx)
                orig_independent_outputs = fmodel(
                    is_eval=False,
                    **single_independent_kwargs,
                    use_cache=False,
                    params=init_use_params
                )
                orig_independent_logits = orig_independent_outputs['logits'].detach()

        # until num_updates reached or point is correctly predicted, update and get new predictions (see break statement elsewhere)
        num_updates = 0
        running_loss = 0
        while num_updates < max_num_updates:

            # get new params. run new model on the entire batch
            param_update_dict = self.compute_param_update(single_point_logits, single_point, 
                                                          current_update_dict=self.running_update_dict)
            self.running_update_dict = param_update_dict
            fmodel = make_functional(self.model).eval()
            input_kwargs = {k:v for k,v in kwargs.items() if k in ['input_ids', 'attention_mask', 'labels', 'decoder_input_ids']}
            new_outputs = fmodel(
                is_eval=False,
                **input_kwargs,
                use_cache=False,
                params=[param_update_dict.get(n, 0) + p for n, p in self.model.named_parameters()]
            )

            # get new single point and other point logits
            new_single_point_logits = new_outputs['logits'][use_for_model_grad_idx,...]
            other_new_logits = new_outputs['logits'][other_point_idx,...]

            # add loss if training (incl. train on paraphrases)
            if not is_eval:
                # main point loss for target label
                main_ce = self.get_cross_ent(self.args.probing_style, new_single_point_logits, kwargs['labels'][use_for_model_grad_idx])
                # accumulate losses and loss weights to be used
                losses = [main_ce]
                loss_weights = [self.args.lambda_main]
                
                # get other loss terms if requested
                if not using_single_point:
                    if self.args.min_corruption:
                        labels = kwargs['orig_labels'][other_point_idx,...]
                        other_ce = self.get_cross_ent(self.args.probing_style, other_new_logits, labels)
                        losses.append(other_ce)
                        loss_weights.append(self.args.lambda_corruption)
                    if self.args.divergences == "kl":
                        KL = self.kl_loss(other_new_logits.log_softmax(-1), other_orig_logits.log_softmax(-1))
                        losses.append(KL)
                        loss_weights.append(self.args.lambda_kl)

                # fit to paraphrases
                if self.args.fit_opt_to_paraphrases:
                    paraphrases_loss = 0
                    n_local_paraphrases = 0
                    use_paraphrase_kwargs = kwargs['paraphrases'][use_for_model_grad_idx.item()]
                    all_paraphrases_labels = []
                    all_paraphrase_preds = []
                    new_labels = None
                    if use_paraphrase_kwargs is not None:
                        # paraphrase labels will be defined at end of loop, using generative preds if applicable
                        paraphrase_size = use_paraphrase_kwargs['input_ids'].size(0)
                        new_pred = new_outputs['preds']
                        new_labels = [new_pred for _ in range(paraphrase_size)]
                        list_kwargs = utils.kwargs_into_batches(use_paraphrase_kwargs, self.args.train_batch_size)
                        for paraphrase_kwargs in list_kwargs:
                            bs = paraphrase_kwargs['input_ids'].size(0)
                            utils.move_kwargs_to_gpu(paraphrase_kwargs)
                            paraphrase_outputs = fmodel(**paraphrase_kwargs,
                                use_cache=False,
                                params=[param_update_dict.get(n, 0) + p for n, p in self.model.named_parameters()]
                            )
                            paraphrases_loss += bs*paraphrase_outputs['loss']
                            n_local_paraphrases += bs
                            all_paraphrase_preds.extend(paraphrase_outputs['preds'].tolist())
                        losses.append(paraphrases_loss / n_local_paraphrases)
                        loss_weights.append(self.args.lambda_paraphrase)
                    new_outputs['all_paraphrase_preds'] = all_paraphrase_preds
                    new_outputs['all_paraphrase_eq_labels'] = new_labels # will be overwritten if seq2seq below
                    # get orig labels for paraphrases

                # fit to depependent propositions. get metrics for entailed propositions
                if self.args.fit_opt_to_dependent_propositions:
                    dependent_kwargs = {k.replace("dependent_proposition_","") : v for k,v in kwargs.items() if 'dependent_proposition' in k}
                    if not kwargs['nan_dependent_proposition_labels'][use_for_model_grad_idx]:
                        # only apply this loss if changing model pred to true
                        if kwargs['labels'][use_for_model_grad_idx].item() == 1:
                            single_dependent_kwargs = utils.slice_kwargs(dependent_kwargs, use_for_model_grad_idx)
                            new_outputs['all_dependent_orig_labels'] = single_dependent_kwargs['orig_labels'] # defined in utils
                            utils.move_kwargs_to_gpu(single_dependent_kwargs)
                            dependent_outputs = fmodel(**single_dependent_kwargs,
                                        use_cache=False,
                                        params=[param_update_dict.get(n, 0) + p for n, p in self.model.named_parameters()]
                            )
                            new_outputs['all_dependent_preds'] = dependent_outputs['preds']
                            losses.append(dependent_outputs['loss'])
                            loss_weights.append(self.args.lambda_dependents_updated)

                if self.args.fit_opt_to_independent_propositions:
                    new_outputs['all_independent_labels'] = single_independent_kwargs['labels']
                    utils.move_kwargs_to_gpu(single_independent_kwargs)
                    independent_outputs = fmodel(**single_independent_kwargs,
                                use_cache=False,
                                params=[param_update_dict.get(n, 0) + p for n, p in self.model.named_parameters()]
                    )
                    new_outputs['all_independent_preds'] = independent_outputs['preds']
                    new_ind_logits = independent_outputs['logits']
                    KL = self.kl_loss(new_ind_logits.log_softmax(-1), orig_independent_logits.log_softmax(-1))
                    losses.append(KL)
                    loss_weights.append(self.args.lambda_independents_updated)

                # make weighted loss
                loss_weight_sum = sum(loss_weights)
                loss_weights = [loss_weight / loss_weight_sum for loss_weight in loss_weights]
                loss = sum([loss*loss_weight for loss, loss_weight in zip(losses, loss_weights)])

                running_loss += loss
            new_outputs['loss'] = running_loss

            # add before and after preds and labels
            if not using_single_point:
                new_outputs['all_before_preds'] = orig_outputs['preds'][other_point_idx] # EXCLUDING the updated point
                new_outputs['all_after_preds'] = new_outputs['preds'][other_point_idx] # EXCLUDING the updated point
                new_outputs['all_labels'] = kwargs['orig_labels'][other_point_idx,...]
            # add update_succ indicator
            updated_pred = new_outputs['preds'][use_for_model_grad_idx.item()]
            new_outputs['updated_pred'] = updated_pred
            new_outputs['update_succ'] = metrics.compute_acc_sum(self.args.probing_style, [updated_pred], kwargs['labels'][use_for_model_grad_idx], self.tokenizer)

            # if correct, break rather than continuing to update
            if new_outputs['update_succ']:
                break
            # if going to iterate another time, need to get new logits for the next compute_param_update
            if num_updates < max_num_updates-1:
                # recompute logits on the single point
                new_outputs = fmodel(**single_point,
                    use_cache=False,
                    params=[param_update_dict.get(n, 0) + p for n, p in self.model.named_parameters()]
                )
                single_point_logits = new_outputs.logits
            num_updates += 1

        # attach new model to new_outputs if it will be needed outside, and the selected point idx
        if return_new_model:
            new_outputs['updated_model'] = self.get_updated_model()
        new_outputs['use_for_model_grad_idx'] = new_outputs['update_idx'] = use_for_model_grad_idx

        # if seq2seq and is_eval and about to return, get generative preds
        if self.args.probing_style == 'seq2seq' and is_eval:
            assert using_single_point
            fmodel = make_functional(self.model).eval()
            input_kwargs = {k:v for k,v in kwargs.items() if k in ['input_ids', 'attention_mask']}
            gen_outputs = fmodel(
                is_eval=is_eval,
                **input_kwargs,
                use_cache=False,
                params=[param_update_dict.get(n, 0) + p for n, p in self.model.named_parameters()]
            )
            new_outputs['preds'] = gen_outputs['preds'] 
            if 'all_paraphrase_eq_labels' in new_outputs: # overwrite labels for paraphrases
                paraphrase_size = use_paraphrase_kwargs['input_ids'].size(0)
                new_pred = new_outputs['preds']
                new_outputs['all_paraphrase_eq_labels'] = [new_pred for _ in range(paraphrase_size)]
        if self.args.implementation == 'de_cao':
            self.on_before_zero_grad()

        # handle num_successive_updates counter
        self.successive_updates_counter += 1
        if self.successive_updates_counter >= self.args.num_successive_updates:
            self.reset_successive_update_vars()

        return new_outputs

    def get_cross_ent(self, probing_style, logits, labels):
        if probing_style == 'seq2seq':
            labels = labels[:,1:]
            loss = self.loss(logits.view(-1, logits.size(-1)), labels.reshape(-1))
        else:
            loss = torch.nn.CrossEntropyLoss()(logits, labels)
        return loss
    
    def increase_kl_weight(self, update_succ):
        if update_succ > 90:
            self.args.lambda_kl += 10

    def reduce_constraint_margins(self, update_succ):
        if update_succ > 90:
            self.margin_kl = max(
                self.margin_kl * 0.8, self.margin_kl * 0 + self.hparams.margin_kl_min
            )

    def on_before_zero_grad(self):
        self.alpha_kl.data = torch.where(
            self.alpha_kl.data < 0,
            torch.full_like(self.alpha_kl.data, 0),
            self.alpha_kl.data,
        )

    def reset_successive_update_vars(self):
        self.successive_updates_counter = 0
        self.running_update_dict = None