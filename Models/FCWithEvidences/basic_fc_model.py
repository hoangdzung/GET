import torch
import torch.nn.functional as F
import torch.nn as nn
from matchzoo.utils import parse
from Models.base_model import BaseModel
from matchzoo.modules import GaussianKernel
import torch_utils
from setting_keywords import KeyWordSettings
import numpy as np

class BasicFCModel(BaseModel):
    """
    Basic Fact-checking model used for all other models
    """
    def __init__(self, params):
        super(BaseModel, self).__init__()
        self._params = params
        self.embedding = self._make_default_embedding_layer(params)
        self.num_classes = self._params["num_classes"]
        self.fixed_length_right = self._params["fixed_length_right"]
        self.fixed_length_left = self._params["fixed_length_left"]
        self.use_claim_source = self._params["use_claim_source"]
        self.use_article_source = self._params["use_article_source"]
        self._use_cuda = self._params["cuda"]
        self.num_heads = 1  # self._params["num_att_heads"]
        self.dropout_left = self._params["dropout_left"]
        self.dropout_right = self._params["dropout_right"]
        self.hidden_size = self._params["hidden_size"]
        if self.use_claim_source:
            self.claim_source_embs = self._make_entity_embedding_layer(
                self._params["claim_source_embeddings"], freeze = False)  # trainable
            self.claim_emb_size = self._params["claim_source_embeddings"].shape[1]

        if self.use_article_source:
            self.article_source_embs = self._make_entity_embedding_layer(
                self._params["article_source_embeddings"], freeze = False)  # trainable
            self.article_emb_size = self._params["article_source_embeddings"].shape[1]

    def forward(self, query: torch.Tensor, document: torch.Tensor, verbose = False, **kargs):
        pass

