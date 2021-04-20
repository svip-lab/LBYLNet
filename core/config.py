import os
import numpy as np

class SystemConfig(object):
    def __init__(self):
        self._configs = {}
        self._configs["dataset"] = None
        self._configs["language"]           = "lstm"
        self._configs["model"]              = "LBYLNet"
        self._configs["anchor_based"]       = False
        self._configs["context"]            = "LandmarkP4"
        self._configs["ctx_dim"]            = 128
        self._configs["visu_weight"] = "yolov3.weights"
        # optimizer 
        self._configs["lr_scheduler"]      = "cosin_lr"
        self._configs["warm_up"]           = True
        self._configs["warm_up_epoch"]     = 0
        self._configs["warm_up_from_lr"]   = 0.0001
        self._configs["learning_rate"]     = 0.001
        self._configs["decay_rate"]        = 10
        # multi step lr
        self._configs["milestone"]         = [30, 80]
        self._configs["gamma"]             = 0.1

        self._configs["opt_algo"]          = "adam"
        # Training Config
        self._configs["display"]           = 5
        self._configs["nb_epoch"]          = 100
        self._configs["print_freq"]        = 100
        self._configs["snapshot"]          = 10
        self._configs["stepsize"]          = 10
        self._configs["val_iter"]          = 20
        self._configs["batch_size"]        = 1
        self._configs["snapshot_name"]     = None
        self._configs["prefetch_size"]     = 100
        self._configs["pretrain"]          = None
        self._configs["chunk_sizes"]       = None

        # Directories
        self._configs["corpus_dir"] = "./data/refer/"
        self._configs["data_dir"]   = "./data"
        self._configs["cache_dir"]  = "./cache"
        self._configs["config_dir"] = "./config"
        self._configs["result_dir"] = "./results"

        # Split
        self._configs["train_split"] = "training"
        self._configs["val_split"]   = "validation"
        self._configs["test_split"]  = "test"

        # Rng
        self._configs["data_rng"] = np.random.RandomState(123)
        self._configs["nnet_rng"] = np.random.RandomState(317)

    @property
    def lstm(self):
        return self._configs['language'] == 'lstm'

    @property
    def lang_encoder(self):
        return self._configs['language']

    @property
    def model(self):
        return self._configs["model"]

    @property
    def ctx_dim(self):
        return self._configs["ctx_dim"]

    def freeze_epoch(self):
        return self._configs["frezze_epoch"]

    @property
    def visu_weight(self):
        return self._configs["visu_weight"]

    @property
    def warm_up_lr(self):
        return self._configs["warm_up_from_lr"]
    @property
    def warm_up(self):
        return self._configs["warm_up"]
    
    @property
    def context(self):
        return self._configs["context"]
    
    @property
    def corpus_dir(self):
        return self._configs["corpus_dir"]

    @property
    def print_freq(self):
        return self._configs["print_freq"]
    
    @property
    def nb_epoch(self):
        return self._configs["nb_epoch"]

    @property
    def chunk_sizes(self):
        return self._configs["chunk_sizes"]

    @property
    def train_split(self):
        return self._configs["train_split"]

    @property
    def val_split(self):
        return self._configs["val_split"]

    @property
    def test_split(self):
        return self._configs["test_split"]

    @property
    def full(self):
        return self._configs

    @property
    def sampling_function(self):
        return self._configs["sampling_function"]

    @property
    def data_rng(self):
        return self._configs["data_rng"]

    @property
    def nnet_rng(self):
        return self._configs["nnet_rng"]

    @property
    def opt_algo(self):
        return self._configs["opt_algo"]

    @property
    def prefetch_size(self):
        return self._configs["prefetch_size"]

    @property
    def pretrain(self):
        return self._configs["pretrain"]

    @property
    def result_dir(self):
        result_dir = os.path.join(self._configs["result_dir"], self.snapshot_name, self.dataset)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        return result_dir

    @property
    def dataset(self):
        return self._configs["dataset"]

    @property
    def snapshot_name(self):
        return self._configs["snapshot_name"]

    @property
    def snapshot_dir(self):
        snapshot_dir = os.path.join(self.cache_dir, "nnet", self.snapshot_name, self.dataset)

        if not os.path.exists(snapshot_dir):
            os.makedirs(snapshot_dir)
        return snapshot_dir

    @property
    def snapshot_file(self):
        snapshot_file = os.path.join(self.snapshot_dir, self.snapshot_name + "_{}.pkl")
        return snapshot_file

    @property
    def config_dir(self):
        return self._configs["config_dir"]

    @property
    def batch_size(self):
        return self._configs["batch_size"]

    @property
    def learning_rate(self):
        return self._configs["learning_rate"]

    @property
    def stepsize(self):
        return self._configs["stepsize"]

    @property
    def snapshot(self):
        return self._configs["snapshot"]

    @property
    def display(self):
        return self._configs["display"]

    @property
    def val_iter(self):
        return self._configs["val_iter"]

    @property
    def data_dir(self):
        return self._configs["data_dir"]

    @property
    def cache_dir(self):
        if not os.path.exists(self._configs["cache_dir"]):
            os.makedirs(self._configs["cache_dir"])
        return self._configs["cache_dir"]

    def update_config(self, new):
        unrecognized = []
        for key in new:
            if key in self._configs:
                self._configs[key] = new[key]
            else:
                unrecognized.append(key)
        if len(unrecognized):
            print("warning : unrecognized sys keys {}".format(unrecognized))
        return self
