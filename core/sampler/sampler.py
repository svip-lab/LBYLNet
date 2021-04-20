import sys
import cv2
import random
import math
import numpy as np
import torch
import pdb
# import logging

from torch.utils.data import Dataset

from .utils import (normalize_, color_jittering_, \
                lighting_, random_flip_, random_affine_, clip_bbox_, \
                show_example, random_crop_, resize_image_, valid_affine)

from .utils import convert_examples_to_features, read_examples
from pytorch_pretrained_bert.tokenization import BertTokenizer

class Referring(Dataset):
    def __init__(self, db, system_configs, data_aug=True, debug=False, shuffle=False, test=False):
        super(Referring, self).__init__()
        self.test = test
        self._db = db
        self._sys_config = system_configs
        self.lstm = system_configs.lstm
        self.data_rng = system_configs.data_rng
        self.data_aug = data_aug
        self.debug = debug
        self.input_size    = self._db.configs["input_size"]
        self.output_size   = self._db.configs["output_sizes"]
        # self.rand_scales   = self._db.configs["rand_scales"]
        self.rand_color    = self._db.configs["random_color"]
        self.random_flip   = self._db.configs["random_flip"]
        self.random_aff    = self._db.configs["random_affine"]
        self.lighting      = self._db.configs["random_lighting"]
        self.query_len     = self._db.configs["max_query_len"]
        self.corpus        = self._db.corpus
        self.tokenizer     = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        if shuffle:
            self._db.shuffle_inds()
        
    def __len__(self):
        return len(self._db.db_inds)

    def _tokenize_phrase(self, phrase):
        return self.corpus.tokenize(phrase, self.query_len)

    def __getitem__(self, k_ind):
        db_ind = self._db.db_inds[k_ind]
        while True:
            # reading images
            image_path = self._db.image_path(db_ind)
            image = cv2.imread(image_path)
                # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # else:
            if not image.shape[-1] > 1:
                image = np.stack([image] * 3) # duplicate channel if gray image
            
            # original scale
            original_shape = image.shape
            # reading bbox annnotation
            bbox = self._db.annotation_box(db_ind)
            # reading phrase
            phrase = self._db.phrase(db_ind)
            phrase = phrase.lower()

            if self.data_aug:
                if self.random_flip and random.random() > 0.5:
                    image, phrase, bbox = random_flip_(image, phrase, bbox.copy()) # should ensure bbox read-only
                    
                # resize images
                image, bbox = resize_image_(image, bbox.copy(), self.input_size)
                if self.random_aff:
                    aff_image, aff_bbox = random_affine_(image, bbox.copy())
                    if valid_affine(aff_bbox, aff_image.shape[:2]):
                        # only keep valid_affine
                        image = aff_image
                        bbox = aff_bbox

                if self.debug and k_ind % 5000 == 0:
                    show_example(image, bbox, phrase, name="input_sample{}".format(k_ind))

                image = image.astype(np.float32) / 255.
                if self.rand_color:
                    color_jittering_(self.data_rng, image)
                    if self.lighting:
                        lighting_(self.data_rng, image, 0.1, self._db.eig_val, self._db.eig_vec)
                normalize_(image, self._db.mean, self._db.std)
            else:   ## should be inference, or specified training
                image, bbox = resize_image_(image, bbox.copy(), self.input_size, \
                    padding_color=tuple((self._db.mean * 255).tolist()))    
                image = image.astype(np.float32) / 255.
                normalize_(image, self._db.mean, self._db.std)
            
            bbox = clip_bbox_(bbox.copy(), image.shape[0:2])

            if not ((bbox[2] - bbox[0] > 0) and (bbox[3] - bbox[1] > 0)):
                # show_example(image, bbox.copy(), phrase, name="failure_case_{}".format(k_ind))
                # if failure, choose next image
                db_ind = random.choice(self._db.db_inds)
                continue
            
            image = image.transpose((2, 0, 1))
            if not self.lstm: # for BERT 
                examples = read_examples(phrase, db_ind)
                features = convert_examples_to_features(examples=examples, \
                    seq_length=self.query_len, tokenizer=self.tokenizer)
                word_id = features[0].input_ids
                word_mask = features[0].input_mask
                if self.test:
                    word_id = torch.tensor(word_id, dtype=torch.long)
                    return image, word_id, original_shape
                else:
                    return image, bbox, word_id, word_mask
            else: # for lstm
                phrase = self._tokenize_phrase(phrase)
                if self.test:
                    return image, phrase, original_shape
                else:
                    return image, phrase, bbox
