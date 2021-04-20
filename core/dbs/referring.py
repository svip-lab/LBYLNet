import numpy as np

from .base import BASE

class REFERDB(BASE):
    def __init__(self, db_config):
        super(REFERDB, self).__init__()
        # Configs for 
        self._configs["data_aug"]         = True
        self._configs["random_flip"]      = True
        self._configs["random_affine"]    = True
        self._configs["random_color"]     = True
        self._configs["random_lighting"]  = True
        self._configs["input_size"]     = [256, 256]
        self._configs["output_sizes"]   = [32,32]

        # Configs for both training and testing
        self._configs["anchors"]              = None

        # Configs for language model 
        self._configs["vocab_size"]           = 0       
        self._configs["word_embedding_size"]  = 512
        self._configs["word_vec_size"]        = 512
        self._configs["hidden_size"]          = 512
        self._configs["bidirectional"]        = True
        self._configs["input_dropout_p"]      = 0.5
        self._configs["dropout_p"]            = 0.2
        self._configs["n_layers"]             = 1
        self._configs["max_query_len"]        =  128
        self._configs["variable_length"]      = True
        self._configs["joint_embedding_size"] = 256
        self._configs["joint_out_dim"]        = 256
        self._configs["joint_embedding_dropout"] = 0.1
        self._configs["joint_mlp_layers"]     = 2   
        self._configs["corpus_path"]          = None
        self.update_config(db_config)
