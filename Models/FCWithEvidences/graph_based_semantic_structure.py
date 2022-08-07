import time

import torch

from Models.FCWithEvidences.basic_fc_model import BasicFCModel
import torch_utils
from setting_keywords import KeyWordSettings
from Models.BiDAF.wrapper import GGNN, GGNN_with_GSL, Linear
from thirdparty.two_branches_attention import *
import numpy as np
import torch_utils as my_utils
from transformers import AutoModel

torch.set_printoptions(profile="full")

class Graph_basedSemantiStructure(BasicFCModel):
    """ Hierarchical Multi-Head Attention Network for Fact-Checking (MAC)"""

    def __init__(self, params):
        super(Graph_basedSemantiStructure, self).__init__(params)
        self._params = params
        if self._params['embedding'] is not None:
            self.embedding = self._make_default_embedding_layer(params)
            self.is_bert = False
        else:
            self.embedding = AutoModel.from_pretrained('bert-base-uncased')
            self._params['embedding_output_dim'] = 768
            self.is_bert = True
        self.num_classes = self._params["num_classes"]
        self.fixed_length_right = self._params["fixed_length_right"]
        self.fixed_length_left = self._params["fixed_length_left"]
        self.use_claim_source = self._params["use_claim_source"]
        self.use_article_source = self._params["use_article_source"]
        self._use_cuda = self._params["cuda"]
        # number of attention heads
        self.num_att_heads_for_words = self._params["num_att_heads_for_words"]
        self.num_att_heads_for_evds = self._params["num_att_heads_for_evds"]

        self.dropout_gnn = self._params['dropout_gnn']
        self.dropout_left = self._params["dropout_left"]
        self.dropout_right = self._params["dropout_right"]
        self.hidden_size = self._params["hidden_size"]
        self.output_size = self._params['output_size']
        self.gsl_rate = self._params["gsl_rate"]

        if self.use_claim_source:
            self.claim_source_embs = self._make_entity_embedding_layer(
                self._params["claim_source_embeddings"], freeze=False)  # trainable
            self.claim_emb_size = self._params["claim_source_embeddings"].shape[1]

        if self.use_article_source:
            self.article_source_embs = self._make_entity_embedding_layer(
                self._params["article_source_embeddings"], freeze=False)  # trainable
            self.article_emb_size = self._params["article_source_embeddings"].shape[1]

        D = self._params["embedding_output_dim"]
        
        # Graph Gated Neural Network with structural learning
        # self.ggnn4claim_1 = GGNN(in_features=D, out_features=self.hidden_size)
        
        # self.ggnn_with_gsl = GGNN_with_GSL(input_dim=D, hidden_dim=self.hidden_size, output_dim=self.hidden_size, rate=self.gsl_rate, dropout=self.dropout_gnn)
            
        # mapping query vector + claim's source vector if possible. Experiments show that without using claims'
        # src, Politifact dataset has lower performance
        dim = self.hidden_size # the dimension of the output of representation models (e.g., ggnn, bilstm)
        self._set_word_attention_func(dim=dim)
        self._set_evd_attention_func(dim=dim)

        evd_input_size = dim  # the first is for claim, the second is for
        if self.use_claim_source: 
            evd_input_size += self.claim_emb_size
        evd_input_size += dim * self.num_att_heads_for_words * self.num_att_heads_for_evds  # twice times for two times attention
        if self.use_article_source: 
            evd_input_size += self.article_emb_size * self.num_att_heads_for_evds
        self.out = nn.Sequential(
            nn.Linear(evd_input_size, self.hidden_size),
            nn.Linear(self.hidden_size, self.output_size)
        )
        self.out[0].apply(torch_utils.init_weights)
        self.out[1].apply(torch_utils.init_weights)

    def forward(self, query: torch.Tensor, document: torch.Tensor, verbose=False, **kargs):
        """
        query and document have shaped as described. Each query is assumed to have `n = 30` evidences. If a query has
        less than 30 evidences, I pad them with all zeros. The length of all-zeros evidence is 0. However, PyTorch
        does not allow empty sequences input to RNN. Therefore, I have to use
        `kargs[KeyWordSettings.QueryContentNoPaddingEvidence]` and `kargs[KeyWordSettings.DocContentNoPaddingEvidence]`
        with shape (n1 + n2 + ... + nx, L) and (n1 + n2 + ... + nx, R) respectively.
        Parameters
        ----------
        query: `torch.Tensor`  (B, L)
        document: `torch.Tensor` (B, n = 30, R)
        """
        # assert KeyWordSettings.Doc_lens in kargs
        _, L = query.size()
        D = self._params["embedding_output_dim"]
        assert query.size(0) == document.size(0)
        batch_size, n, R = document.size()  # batch_size = 32 which is real batch_size of each of mini-batches
        assert n == 30
        # for documents
        # d_new_indices, d_restoring_indices, d_lens = kargs[KeyWordSettings.DocLensIndices]
        assert KeyWordSettings.DocContentNoPaddingEvidence in kargs
        doc = kargs[KeyWordSettings.DocContentNoPaddingEvidence]  # (n1 + n2 + n3 + .. n_b, R)
        doc_mask = (doc >= 1)  # (B1, R) 0 is for padding word
        # doc_adj = kargs[KeyWordSettings.Evd_Docs_Adj].float()  # (n1 + n2 + n3 + .. n_b, R, R)
        # query_adj = kargs[KeyWordSettings.Query_Adj].float()  # (n1 + n2 + n3 + .. n_b, L, L)
        max_num_evd = kargs[KeyWordSettings.FIXED_NUM_EVIDENCES]
        evd_count_per_query = kargs[KeyWordSettings.EvidenceCountPerQuery]
        doc_source_idx = kargs[KeyWordSettings.DocSources]

        # import pdb;pdb.set_trace()
        if not self.is_bert:
            doc_out_ggnn = self.embedding(doc.long())  # (n1 + n2 + n3 + .. n_b, R, D)
        else:
            doc_out_ggnn = self.embedding(doc.long(), doc_mask.int()).last_hidden_state  # (n1 + n2 + n3 + .. n_b, R, D)
        # assert d_lens.shape[0] == embed_doc.size(0)

        # ggnn for query
        query_repr = self._generate_query_repr_gnn(query, evd_count_per_query)  # output's shape is always (B1, self.hidden_size)
        # import pdb;pdb.set_trace()
        # ggnn for doc
        # doc_out_ggnn = self.ggnn_with_gsl(doc_adj, embed_doc)
        # import pdb;pdb.set_trace()
        # Step 1: word-level attention to simplify the graph
        avg, word_att_weights = self._word_level_attention(left_tsr=query_repr, right_tsr=doc_out_ggnn,
                                                           right_mask=doc_mask)
        # Step 2: evidence-level attention. We will override this function in sub-classes
        if self.use_claim_source:
            query_source_idx = kargs[KeyWordSettings.QuerySources]
            claim_embs = self.claim_source_embs(query_source_idx.long())  # (B, 1, D)
            claim_embs = claim_embs.squeeze(1)  # (B, D)
            claim_embs = self._pad_left_tensor(claim_embs, evd_count_per_query)
            query_repr = torch.cat([claim_embs, query_repr], dim=-1)  # (B, 2D + D)
        avg, evd_att_weight = self._evidence_level_attention_new(query_repr, avg, document, 
                                                            max_num_evd, evd_count_per_query, doc_source_idx)

        output = self._get_final_repr(left_tsr=query_repr, right_tsr=avg, max_num_evd=max_num_evd, evd_count_per_query=evd_count_per_query)
        phi = self.out(output)  # (B, )
        
        if kargs.get(KeyWordSettings.OutputRankingKey, False): 
            return phi, (word_att_weights, evd_att_weight)
        return phi

    def _generate_query_repr_gnn(self, queries: torch.Tensor, evd_count_per_query: torch.Tensor):
        query_mask = (queries > 0).unsqueeze(2)  # (B, L, 1) # padded with zeros
        # query_lens = kargs[KeyWordSettings.Query_lens]  # (B, )
        # query_lens = query_lens.unsqueeze(-1)  # (B, 1) # this can be infered from query 
        query_lens = query_mask.reshape((query_mask.shape[0],-1)).sum(-1,keepdims=True)

        # adj = kargs[KeyWordSettings.Query_Adj].float()  # (B, L, L)
        if not self.is_bert:
            query_gnn_hiddens = self.embedding(queries.long())  # (B, L, D)
        else:
            query_gnn_hiddens = self.embedding(queries.long(), query_mask.squeeze(2).int()).last_hidden_state  # (n1 + n2 + n3 + .. n_b, R, D)

        # query_gnn_hiddens = self.ggnn4claim_1(query_adjs, embed_queries)
        # import pdb;pdb.set_trace()
        query_repr = torch.sum(query_gnn_hiddens * query_mask.float(), dim=1) / query_lens.float()  # (B, D)
        query_repr = self._pad_left_tensor(query_repr, evd_count_per_query)  # (n1 + n2 + n3 + .. + nx, H)
        return query_repr

    def _use_article_embeddings(self, article_repr: torch.Tensor, doc_source_idx: torch.Tensor):
        """
        Using article embeddings with articles' representations
        Parameters
        ----------
        article_repr: `torch.Tensor` (B, n, H)
        kargs
        """
        # doc_source_idx = kargs[KeyWordSettings.DocSources]  # (B, n = 30)
        mask = (doc_source_idx == -1)
        # when doc_src has negative values, the exception will be thrown.
        doc_source_idx = doc_source_idx.masked_fill(mask, 0)
        article_embs = self.article_source_embs(doc_source_idx.long())  # (B, n, D)
        article_repr = torch.cat([article_repr, article_embs], dim=-1)  # (B, n, 2D + D)
        return article_repr

    def _word_level_attention(self, left_tsr: torch.Tensor, right_tsr: torch.Tensor, right_mask: torch.Tensor):
        """
            Compute word-level attention of evidences.
        Parameters
        ----------
        left_tsr: `torch.Tensor` of shape (n1 + n2 + ... + nx, H). It represents claims' representation
        right_tsr: `torch.Tensor` of shape (n1 + n2 + ... + nx, R, H). Doc's representations.
        right_mask: `torch.Tensor` (n1 + n2 + ... + nx, R)
        kargs
        Returns
        -------
            Representations of each of evidences of each of claim in the mini-batch of shape (B1, X)
        """
        # for reproducing results in the report
        B1, R, H = right_tsr.size() # [n1+n2..., 100, 300]
        assert left_tsr.size(0) == B1 and len(left_tsr.size()) == 2
        # new_left_tsr = left_tsr.unsqueeze(1).expand(B1, R, -1)
        avg, att_weight = self.self_att_word(left_tsr, right_tsr, right_mask)
        avg = torch.flatten(avg, start_dim=1)  # (n1 + n2 + n3 + ... + nx, n_head * 4D)
        # avg = torch.cat([left_tsr, avg], dim=-1)  # (B1, 2D + D)
        # import pdb;pdb.set_trace()
        return avg, att_weight  # (n1 + n2 + n3 + ... + nx, R)

    def _evidence_level_attention_new(self, left_tsr: torch.Tensor, right_tsr: torch.Tensor,
                                      full_padded_document: torch.Tensor, max_num_evd: int, 
                                      evd_count_per_query: torch.Tensor,
                                      doc_source_idx: torch.Tensor):
        """
        compute evidence-level attention
        Parameters
        ----------
        left_tsr: `torch.Tensor` of shape (n1 + n2 + ... + nx, D)
        right_tsr: `torch.Tensor` of shape (n1 + n2 + ... + nx, D)
        full_padded_document: `torch.Tensor` (B, R). Note, B != (n1 + n2 + ... + nx)

        Returns
        -------
            a tensor of shape (B, _) which stands for representation of `batch_size = B` claims in each of mini-batches
        """
        # for reproducing results in the report
        # if self.evd_attention_type != AttentionType.ConcatNotEqual: left_tsr = self.map_query_level2(left_tsr)

        # max_num_evd = kargs[KeyWordSettings.FIXED_NUM_EVIDENCES]
        # evd_count_per_query = kargs[KeyWordSettings.EvidenceCountPerQuery]

        new_left_tsr = self._pad_right_tensor(left_tsr, max_num_evd, evd_count_per_query)
        new_left = new_left_tsr[:, 0, :]  # (B, X)

        padded_avg = self._pad_right_tensor(right_tsr, max_num_evd, evd_count_per_query)
        mask = (torch.sum(full_padded_document, dim=-1) >= 1).float()  # (B, n), 0 is for padding that why >= 1
        if self.use_article_source:
            padded_avg = self._use_article_embeddings(padded_avg, doc_source_idx)
        attended_avg, att_weight = self.self_att_evd(new_left, padded_avg, mask)
        avg = torch.flatten(attended_avg, start_dim=1)  # (B, num_heads * 2D)
        # import pdb;pdb.set_trace()
        return avg, att_weight

    def _set_word_attention_func(self, dim: int):
        """
        get the function to compute attention weights on word.
        Parameters
        ----------
        dim: `int` the last dimension of an input of attention func
        """
        input_dim = 2 * dim
        self.self_att_word = ConcatNotEqualSelfAtt(inp_dim=input_dim, out_dim=dim,
                                                   num_heads=self.num_att_heads_for_words)
        # else:
        #     raise NotImplemented("Unknown attention type for words")

    def _set_evd_attention_func(self, dim: int):
        """
        get the function to compute attention weights on evidence.
        Parameters
        ----------
        dim: `int` the last dimension of an input of attention func
        """
        # the first is for claim, the second is for word att on evds
        input_dim = dim + self.num_att_heads_for_words * dim
        if self.use_claim_source: input_dim += self.claim_emb_size
        if self.use_article_source: input_dim += self.article_emb_size
        self.self_att_evd = ConcatNotEqualSelfAtt(inp_dim=input_dim, out_dim=dim, num_heads=self.num_att_heads_for_evds)
        # else:
        #     raise NotImplemented("Unknown attention type for evidences")

    def _get_final_repr(self, left_tsr: torch.Tensor, right_tsr: torch.Tensor, max_num_evd: int, evd_count_per_query: torch.Tensor):
        """
        get final representaion of
        Parameters
        ----------
        left_tsr: `torch.Tensor` of shape (n1 + n2 + ... + nx, X) (query features and query's claims features)
        right_tsr: `torch.Tensor` of shape (B, Y) (document features (text + src))
        kargs

        Returns
        -------

        """
        new_left_tsr = self._pad_right_tensor(left_tsr, max_num_evd, evd_count_per_query)
        new_left = new_left_tsr[:, 0, :]  # (B, X)
        tmp = torch.cat([new_left, right_tsr], dim=-1)
        return tmp

    def predict(self, query: torch.Tensor, doc: torch.Tensor, verbose: bool = False, **kargs) -> np.ndarray:
        """ query.shape = (B, L), doc.shape = (B, R) """
        self.train(False)  # very important, to disable dropout
        assert query.size(0) == doc.size(0)
        probs = self(query, doc, **kargs)  # (1, )  it is not softmax yet, how to check?
        return probs

    def _pad_left_tensor(self, left_tsr: torch.Tensor, evd_count_per_query: torch.Tensor):
        """ pad left tensor of shape (B, H) to tensor of shape (n1 + n2 + ... + nx, H) """
        # evd_count_per_query = kargs[KeyWordSettings.EvidenceCountPerQuery]
        B, H = left_tsr.size() ## batch_size and embedding dimension
        assert evd_count_per_query.size(0) == left_tsr.size(0)
        ans = []
        for num_evd, tsr in zip(evd_count_per_query, left_tsr):
            # num_evd = evd_count_per_query[idx] # int(torch_utils.cpu(evd_count_per_query[idx]).detach().numpy())
            tmp = tsr.clone()
            tsr = tmp.expand(num_evd, H)
            ans.append(tsr)
        ans = torch.cat(ans, dim = 0)  # (n1 + n2 + ... + nx, H)
        return ans

    @classmethod
    def _pad_right_tensor(self, tsr: torch.Tensor, max_num_evd: int, evd_count_per_query: torch.Tensor):
        """
        padding the output evidences. I avoid input empty sequence into lstm due to exception. I tried to make add mask
        to empty sequence but I don't have much belief in it.
        Parameters
        ----------
        bilstm_out: `torch.Tensor` (n1 + n2 + ... + n_B, H)
        doc_src: `torch.Tensor` (B, n) where n is the maximum number of evidences
        Returns
        -------

        """
        # max_num_evd = kargs[KeyWordSettings.FIXED_NUM_EVIDENCES]
        # evd_count_per_query = kargs[KeyWordSettings.EvidenceCountPerQuery]
        batch_size = evd_count_per_query.size(0)
        b_prime, H = tsr.size()
        last = 0
        ans = []
        for idx in range(batch_size):
            num_evd = int(torch_utils.cpu(evd_count_per_query[idx]).detach().numpy())
            hidden_vectors = tsr[last: last + num_evd]  # (n1, H)
            padded = F.pad(hidden_vectors, (0, 0, 0, max_num_evd - num_evd), "constant", 0)
            ans.append(padded)
            last += num_evd
        ans = torch.stack(ans, dim=0)
        assert ans.size() == (batch_size, max_num_evd, H)
        return ans