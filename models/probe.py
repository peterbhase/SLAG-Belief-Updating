import torch
from torch import nn
from transformers import AutoConfig, BertModel
import utils
import numpy as np

class IndexingModule(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, hidden_states, **kwargs):
        return hidden_states[:,0,...]

class Classifier(nn.Module):
    # applied to last_hidden_state, this is equivalent to the classifier from transformers.ModelForSequenceClassification
    def __init__(self, hidden_size, dropout_prob=.1, num_classes=2):
        super().__init__()
        self.classifier = nn.Sequential(IndexingModule(),
                                    nn.Linear(hidden_size, hidden_size), # 2 for true/false
                                    nn.Tanh(),
                                    nn.Dropout(p=dropout_prob),
                                    nn.Linear(hidden_size, num_classes), # 2 for true/false
                                )
    def forward(self, hidden_states, **kwargs):
        return self.classifier(hidden_states)

class TransformerProbe(nn.Module):
    def __init__(self, config):
        super().__init__()
        transformer = BertModel(config=config) # does not use actual bert weights
        self.transformer = transformer
        self.classifier = Classifier(config.hidden_size, config.hidden_dropout_prob, num_classes=2)
    def forward(self, hidden_states, **kwargs):
        outputs = self.transformer(inputs_embeds=hidden_states, attention_mask=kwargs['attention_mask'])
        scores = self.classifier(outputs.last_hidden_state)
        return scores

class Probe(nn.Module):

    def __init__(self, args, model, tokenizer):
        super().__init__()
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size
        self.loss = torch.nn.CrossEntropyLoss()
        self.hidden_size = model.config.hidden_size
        
        if args.probing_style=='model':
            # technically, one nonlinear layer and then a linear layer. this achieves an 
            # equivalent implementation to *ModelForSequenceClassification from HuggingFace transformers
            if args.probe == 'linear': 
                self.probe = Classifier(hidden_size=self.hidden_size, dropout_prob=.1, num_classes=2)
            if args.probe=='transformer':
                probe_config = AutoConfig.from_pretrained('bert-base-uncased', cache_dir=args.cache_dir)
                probe_config.hidden_size = self.hidden_size
                probe_config.num_hidden_layers = 1
                self.probe = TransformerProbe(probe_config)

    def non_probe_parameters(self):
        return self.model.parameters()

    def probe_parameters(self):
        return self.probe.parameters()

    def forward(self, is_eval=False, **kwargs):
        '''
        branch function output based on self.probing_style
        '''
        if self.args.probing_style=='seq2seq':
            # case one: need to generate an output
            if is_eval:
                outputs = {} # since won't get outputs obj/dict from model.generate
                batch_size = kwargs['input_ids'].size(0)
                # get encoder outputs first
                enc_kwargs = {k:v for k,v in kwargs.items() if k in ['input_ids', 'attention_mask']}
                encoder_outputs = self.model.get_encoder()(**enc_kwargs)
                gen_kwargs = {'encoder_outputs' : encoder_outputs, 'attention_mask' : kwargs['attention_mask']}
                decoder_input_ids = torch.ones((batch_size, 1)).to(self.args.device).long()
                gen_kwargs['decoder_input_ids'] = decoder_input_ids * (self.tokenizer.eos_token_id)
                # gen_kwargs['decoder_input_ids'] = input_ids * (self.tokenizer.bos_token_id)
                # gen_kwargs['decoder_input_ids'] = input_ids * (self.tokenizer.bos_token_id if 'bart' in self.args.model else self.model.config.decoder_start_token_id)
                # gen_kwargs = {k:v for k,v in kwargs.items() if k in ['input_ids', 'attention_mask']}
                # generate predictions
                beam_size = self.args.beam_search_size
                all_pred_tokens = self.model.generate(**gen_kwargs, max_length=20, num_beams=beam_size, num_return_sequences=beam_size, do_sample=False)
                all_pred_tokens = all_pred_tokens.reshape(batch_size, beam_size, -1)
                pred_tokens = all_pred_tokens[:,0,:]
                if beam_size > 1:
                    alt_pred_tokens = all_pred_tokens[:,1:,:]
                    alt_pred_tokens = alt_pred_tokens.reshape(batch_size*(beam_size-1), -1)
                # handle edge case for bart, which begins its generations with eos token
                if 'bart' in self.args.model: 
                    pred_tokens = pred_tokens[:,1:]
                    if beam_size > 1:
                        alt_pred_tokens = alt_pred_tokens[:,1:]
                        outputs['alt_preds'] = utils.decode_until_eos_token(self.tokenizer, alt_pred_tokens)
            # case two: compute probability of output
            else:
                # shifting decoder_input_ids and labels
                kwargs['decoder_input_ids'] = kwargs['decoder_input_ids'][:,:-1]
                kwargs['labels'] = kwargs['labels'][:,1:].contiguous()
                input_kwargs = {k:v for k,v in kwargs.items() if k in ['input_ids', 'attention_mask', 'decoder_input_ids', 'labels']}
                outputs = self.model(**input_kwargs)
                pred_tokens = outputs.logits.argmax(-1)

            pred_strs = utils.decode_until_eos_token(self.tokenizer, pred_tokens) # this function handles edge cases for bart
            outputs['preds'] = np.array(pred_strs)

        if self.args.probing_style == 'clm':
            # case one: need to generate an output
            if is_eval:
                import pdb; pdb.set_trace()
                outputs = {} # since won't get outputs obj/dict from model.generate
                batch_size = kwargs['input_ids'].size(0)
                # get encoder outputs first
                kwargs = {k:v for k,v in kwargs.items() if k in ['prompt_input_ids', 'prompt_attention_mask']}
                beam_size = self.args.beam_search_size
                all_pred_tokens = self.model.generate(**kwargs, max_length=20, num_beams=beam_size, num_return_sequences=beam_size, do_sample=False)
                all_pred_tokens = all_pred_tokens.reshape(batch_size, beam_size, -1)
                pred_tokens = all_pred_tokens[:,0,:]
                alt_pred_tokens = all_pred_tokens[:,1:,:]
                alt_pred_tokens = alt_pred_tokens.reshape(batch_size*(beam_size-1), -1)
                outputs['alt_preds'] = utils.decode_until_eos_token(self.tokenizer, alt_pred_tokens)
            # case two: compute probability of output
            else:
                # shifting decoder_input_ids and labels
                kwargs['input_ids'] = kwargs['input_ids'][:,:-1]
                kwargs['labels'] = kwargs['labels'][:,1:].contiguous()
                input_kwargs = {k:v for k,v in kwargs.items() if k in ['input_ids', 'attention_mask']}
                outputs = self.model(**input_kwargs)
                pred_tokens = outputs.logits.argmax(-1)
                # might need to cut off the pred_tokens that are matched with the question in order to get pred_strs for computing label acc
                import pdb; pdb.set_trace()

            pred_strs = utils.decode_until_eos_token(self.tokenizer, pred_tokens) # this function handles edge cases for bart
            outputs['preds'] = np.array(pred_strs)
                
        if self.args.probing_style == 'cloze':
            labels = kwargs.pop('labels')
            mask_token_idx = kwargs.pop('mask_token_idx')
            input_kwargs = {k:v for k,v in kwargs.items() if k in ['input_ids', 'attention_mask']}
            outputs = self.model(**input_kwargs)
            if 'candidate_token_ids' not in kwargs:
                use_candidate_ids = self.candidate_ids
            scores = outputs.logits # shape batch_size x seq_len x vocab_size
            # now pull out the scores for the mask token idx for each data point
            mask_idx_scores = torch.gather(scores, 1, mask_token_idx.view(-1,1,1).expand(scores.size(0), 1, scores.size(-1))).squeeze(dim=1)
            # mask out the scores corresponding to tokens besides true/false
            non_candidate_mask = torch.ones_like(mask_idx_scores)
            non_candidate_mask[:,use_candidate_ids] = 0
            mask_idx_scores = mask_idx_scores.masked_fill(non_candidate_mask.bool(), -2e8)
            pred_tokens = mask_idx_scores.argmax(dim=-1)
            outputs['preds'] = pred_tokens
            loss = self.loss(mask_idx_scores, labels)
            outputs['loss'] = loss

        if self.args.probing_style == 'model':
            labels = kwargs.pop('labels')
            input_kwargs = {k:v for k,v in kwargs.items() if k in ['input_ids', 'attention_mask']}
            outputs = self.model(**input_kwargs)
            hidden_states = outputs.last_hidden_state
            logits = self.probe(hidden_states, attention_mask=kwargs['attention_mask'] if self.args.probe=='transformer' else None)
            loss = self.loss(logits, labels)
            outputs['loss'] = loss
            preds = logits.argmax(dim=-1)
            outputs['logits'] = logits
            outputs['preds'] = preds

        return outputs
        