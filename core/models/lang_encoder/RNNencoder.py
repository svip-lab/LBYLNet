import os
import sys 
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel
from .. import utils
from ..utils import Corpus
sys.modules['utils'] = utils
import pdb

class BertEncoder(nn.Module):
    def __init__(self, cfg_db, cfg_sys=None):
        super(BertEncoder, self).__init__()
        self.cfg_db = cfg_db
        self.bert_name = cfg_sys.lang_encoder
        self.model = BertModel.from_pretrained(self.bert_name)
        if self.bert_name == 'bert-base-uncased':
            self.lang_dim = 768
        else:
            self.lang_dim = 1024
        self.num_layers = 4

    def forward(self, input, mask=None):
        max_len = (input!=0).sum(1).max().item()
        encoded_layers, _= self.model(input[:, :max_len], attention_mask=mask[:, :max_len])
        features = None
        features = torch.stack(encoded_layers[-4:], 1).mean(1)
        # features have shape [len(phrase), seq_len, lang_dim]
        features = features / self.num_layers
        hidden = ((features * mask[:, :max_len].unsqueeze(-1).float()).sum(1) / mask.sum(-1).unsqueeze(-1).float())
        embedded  = features[:, :, :]
        ret = {
            'hidden':hidden,
            'embedded': embedded,
            'masks': mask
        }
        return ret



class RNNEncoder(nn.Module):
    def __init__(self, cfg_db, cfg_sys=None):
        super(RNNEncoder, self).__init__()
        self.cfg_db = cfg_db

        self.variable_length      = cfg_db['variable_length']
        self.word_embedding_size  = cfg_db['word_embedding_size']
        self.word_vec_size        = cfg_db['word_vec_size']
        self.hidden_size          = cfg_db['hidden_size']
        self.bidirectional        = cfg_db['bidirectional']
        self.input_dropout_p      = cfg_db['input_dropout_p']
        self.dropout_p            = cfg_db['dropout_p']
        self.n_layers             = cfg_db['n_layers']
        self.rnn_type             = 'lstm' # by default LSTM
        self.corpus_path          = cfg_db['corpus_path']
        self.vocab_size = len(torch.load(self.corpus_path, map_location = "cpu"))
        
        # encoder language
        self.embedding      = nn.Embedding(self.vocab_size, self.word_embedding_size)
        self.input_dropout  = nn.Dropout(self.input_dropout_p)
        self.mlp            = nn.Sequential(nn.Linear(self.word_embedding_size, self.word_vec_size), nn.ReLU())
        self.rnn            = getattr(nn, self.rnn_type.upper())(self.word_vec_size, 
                                                            self.hidden_size, 
                                                            self.n_layers,
                                                            batch_first=True,
                                                            bidirectional=self.bidirectional,
                                                            dropout = self.dropout_p)
        self.num_dirs = 2 if self.bidirectional else 1

    def forward(self, input, mask=None):
        word_id = input
        max_len = (word_id!=0).sum(1).max().item()
        word_id = word_id[:, :max_len] # mask zero
        # embedding
        output, hidden, embedded, final_output = self.RNNEncode(word_id)
        return {
            'hidden': hidden,
            'output': output,
            'embedded': embedded,
            'final_output': final_output,
        }

    def RNNEncode(self, input_labels):
        """
        Inputs:
        - input_labels: Variable long (batch, seq_len)
        Outputs:
        - output  : Variable float (batch, max_len, hidden_size * num_dirs)
        - hidden  : Variable float (batch, num_layers * num_dirs * hidden_size)
        - embedded: Variable float (batch, max_len, word_vec_size)
        """
        device = input_labels.device
        if self.variable_length:
            input_lengths_list, sorted_lengths_list, sort_idxs, recover_idxs = self.sort_inputs(input_labels)
            input_labels = input_labels[sort_idxs]
        
        embedded = self.embedding(input_labels) #(n, seq_len, word_embedding_size)
        embedded = self.input_dropout(embedded) #(n, seq_len, word_embedding_size)
        embedded = self.mlp(embedded)           #(n, seq_len, word_vec_size)

        if self.variable_length:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, \
                                                        sorted_lengths_list,\
                                                         batch_first=True)
        # forward rnn
        self.rnn.flatten_parameters()
        output, hidden = self.rnn(embedded)
        
        # recover
        if self.variable_length:
            # recover embedded
            embedded, _ = nn.utils.rnn.pad_packed_sequence(embedded, batch_first=True)  # (batch, max_len, word_vec_size)
            embedded = embedded[recover_idxs]

            # recover output
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)      # (batch, max_len, hidden_size * num_dir)
            output = output[recover_idxs]

            # recover hidden
            if self.rnn_type == 'lstm':
                hidden = hidden[0]                      # hidden state
            hidden = hidden[:, recover_idxs, :]         # (num_layers * num_dirs, batch, hidden_size)
            hidden = hidden.transpose(0,1).contiguous() # (batch, num_layers * num_dirs, hidden_size)
            hidden = hidden.view(hidden.size(0), -1)    # (batch, num_layers * num_dirs * hidden_size)
        
        # finnal output
        finnal_output = []
        for ii in range(output.shape[0]):
            finnal_output.append(output[ii, int(input_lengths_list[ii]-1), :])
        finnal_output = torch.stack(finnal_output, dim=0)   # (batch, number_dirs * hidden_size)

        return output, hidden, embedded, finnal_output

    def sort_inputs(self, input_labels):                                                # sort input labels by descending
        device = input_labels.device
        input_lengths = (input_labels!=0).sum(1)
        input_lengths_list = input_lengths.data.cpu().numpy().tolist()
        sorted_input_lengths_list = np.sort(input_lengths_list)[::-1].tolist()          # list of sorted input_lengths
        sort_idxs = np.argsort(input_lengths_list)[::-1].tolist()
        s2r = {s:r for r, s in enumerate(sort_idxs)}
        recover_idxs = [s2r[s] for s in range(len(input_lengths_list))]
        assert max(input_lengths_list) == input_labels.size(1)
        # move to long tensor
        sort_idxs = input_labels.data.new(sort_idxs).long().to(device)             # Variable long
        recover_idxs = input_labels.data.new(recover_idxs).long().to(device)       # Variable long
        return input_lengths_list, sorted_input_lengths_list, sort_idxs, recover_idxs
