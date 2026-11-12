#!/usr/bin/env python
# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class Convs(nn.Module):
    def __init__(self, emb_dim):
        super(Convs, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=emb_dim, out_channels=1280, kernel_size=(3, 3), stride=(3, 3))
        self.BR1 = nn.Sequential(nn.BatchNorm2d(1280), nn.PReLU())
        
        self.conv2 = nn.Conv2d(in_channels=1280, out_channels=128, kernel_size=(3, 3), stride=(3, 3))
        self.BR2 = nn.Sequential(nn.BatchNorm2d(128), nn.PReLU())
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))


    def forward(self, x):
        x = self.conv1(x)
        x = self.BR1(x)
        x = self.conv2(x)
        x = self.BR2(x)
        x = self.pool(x)
        return torch.flatten(x, start_dim=1)
    
class AttentionModule(nn.Module):
    def __init__(self, esmc_dim=1152, dropout=0.1):
        super(AttentionModule, self).__init__()
        self.query = nn.Linear(esmc_dim, esmc_dim)
        self.key = nn.Linear(esmc_dim, esmc_dim)
        self.value = nn.Linear(esmc_dim, esmc_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):

        batch_size, seq_len, esmc_dim = x.size()
        
        # 计算自注意力
        q = self.query(x) 
        k = self.key(x)   
        v = self.value(x)  
        
        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / (esmc_dim ** 0.5)  

        # 应用掩码
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # 计算加权特征
        context = torch.matmul(attn, v)  
        return context, attn

class CrossAttention(nn.Module):
    def __init__(self, emb_dim=1152, dropout=0.05):
        super(CrossAttention, self).__init__()
        self.query = nn.Linear(emb_dim, emb_dim)
        self.key = nn.Linear(emb_dim, emb_dim)
        self.value = nn.Linear(emb_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x1, x2, query_mask=None, key_mask=None):
        batch_size, seq_len1, esmc_dim = x1.size()
        _, seq_len2, _ = x2.size()
        
        q = self.query(x1)  
        k = self.key(x2)    
        v = self.value(x2)  
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / (esmc_dim ** 0.5)  
        # 屏蔽partner序列的padding，避免padding进入softmax分母。
        if key_mask is not None:
            scores = scores.masked_fill(
                ~key_mask[:, None, :], torch.finfo(scores.dtype).min
            )
        
        attn = F.softmax(scores, dim=-1)
        # padded query行不应产生context或残基分数。
        if query_mask is not None:
            attn = attn * query_mask[:, :, None].to(attn.dtype)
        if key_mask is not None:
            attn = attn * key_mask[:, None, :].to(attn.dtype)
        attn = self.dropout(attn)
        
        context = torch.matmul(attn, v)  
        
        return context, attn
        
class Model(nn.Module):
    def __init__(self, embed_hid_dim=128, dropout=0.2, interaction_gate_init=-2.0,
                 use_aac_dpc=True, use_cksaap_esmc=True,
                 use_contrastive_loss=True, use_cross_attention=True):
        super(Model, self).__init__()
        self.drop = dropout  
        self.ckesmc_size = 5760  
        self.esmc_dim = 1152        
        self.dropout = dropout
        self.embed_hid_dim = embed_hid_dim

        self.use_aac_dpc = use_aac_dpc
        self.use_cksaap_esmc = use_cksaap_esmc
        self.use_contrastive_loss = use_contrastive_loss
        self.use_cross_attention = use_cross_attention
        if not (use_aac_dpc or use_cksaap_esmc):
            raise ValueError("At least one global sequence encoder is required")
        if use_cksaap_esmc:
            self.esmc_conv = Convs(self.ckesmc_size)
        if use_aac_dpc:
            self.fc_ad = torch.nn.Linear(420, 128) 
        if use_aac_dpc and use_cksaap_esmc:
            self.rep = torch.nn.Linear(256, 128)    
        self.Prelu = nn.PReLU()
        if use_cross_attention:
            self.cross_attention = CrossAttention(self.esmc_dim, dropout)
            # 仅新增cross-attention分类融合；原有序列编码器保持不变。
            self.cross_context1 = nn.Linear(self.esmc_dim, embed_hid_dim)
            self.cross_context2 = nn.Linear(self.esmc_dim, embed_hid_dim)

        # 显式分离global决策与cross-attention interaction增量。
        self.global_clf = nn.Sequential(
            nn.Linear(embed_hid_dim * 2, embed_hid_dim),
            nn.ReLU(),
            nn.Dropout(self.drop),
            nn.Linear(embed_hid_dim, 1)
        )
        if use_cross_attention:
            self.interaction_clf = nn.Sequential(
                nn.Linear(embed_hid_dim * 2, embed_hid_dim),
                nn.ReLU(),
                nn.Dropout(self.drop),
                nn.Linear(embed_hid_dim, 1)
            )
            self.interaction_gate_logit = nn.Parameter(
                torch.tensor(float(interaction_gate_init))
            )
      
        if use_contrastive_loss:
            self.projector = nn.Sequential(
                nn.Linear(embed_hid_dim, embed_hid_dim), 
                nn.BatchNorm1d(embed_hid_dim), 
                nn.ReLU(), 
                nn.Linear(embed_hid_dim, embed_hid_dim)
            )
        self.dropout_layer = nn.Dropout(self.drop)

    def _process_single_protein(self, ckesmc, aac, dipeptide):
        if not self.use_aac_dpc:
            return self.esmc_conv(ckesmc)
        if not self.use_cksaap_esmc:
            return self.Prelu(self.fc_ad(torch.cat([aac, dipeptide], dim=1)))
        esm_rep = self.esmc_conv(ckesmc)
        ad = torch.cat([aac, dipeptide], dim=1)
        ad_pro = self.Prelu(self.fc_ad(ad))
        
        seq_rep = torch.cat([esm_rep, ad_pro], dim=1)
        return self.Prelu(self.rep(seq_rep))
    
    @staticmethod
    def _valid_mask(seq_lengths, max_len, device):
        if seq_lengths is None:
            return torch.ones(1, max_len, dtype=torch.bool, device=device)
        return torch.arange(max_len, device=device)[None, :] < seq_lengths[:, None]

    @staticmethod
    def _masked_mean(x, valid_mask):
        weights = valid_mask.unsqueeze(-1).to(x.dtype)
        return (x * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    def _calculate_residue_scores(
        self, cross_attention, query_mask, key_mask, signal_peptide_length=0
    ):
        masked_attention = cross_attention.masked_fill(
            ~key_mask[:, None, :], float('-inf')
        )
        k = max(1, min(10, masked_attention.size(2)))
        top_values = torch.topk(masked_attention, k=k, dim=2).values
        finite = torch.isfinite(top_values)
        top_values = torch.where(finite, top_values, torch.zeros_like(top_values))
        residue_scores = top_values.sum(dim=2) / finite.sum(dim=2).clamp_min(1)

        valid_query = query_mask.clone()
        if signal_peptide_length > 0:
            valid_query[:, :signal_peptide_length] = False
        positive_inf = torch.full_like(residue_scores, torch.inf)
        negative_inf = torch.full_like(residue_scores, -torch.inf)
        minimum = torch.where(
            valid_query, residue_scores, positive_inf
        ).min(dim=1, keepdim=True).values
        maximum = torch.where(
            valid_query, residue_scores, negative_inf
        ).max(dim=1, keepdim=True).values
        residue_scores = (residue_scores - minimum) / (
            maximum - minimum
        ).clamp_min(1e-6)
        return torch.where(
            valid_query, residue_scores, torch.zeros_like(residue_scores)
        )

    def _create_attention_mask(self, seq_lengths, max_len):

        batch_size = seq_lengths.size(0)
        mask = torch.ones(batch_size, max_len, max_len, device=seq_lengths.device)
        
        for i, length in enumerate(seq_lengths):
            mask[i, :length, :length] = 0  
            
        return mask.bool()
    
    def forward(self, ckesmc1, ckesmc2, aac1, aac2, dipeptide1, dipeptide2, 
                esmc1=None, esmc2=None, esmc_lengths1=None, esmc_lengths2=None,
                disable_cross_attention=False):
        
        seq1_embed = self._process_single_protein(ckesmc1, aac1, dipeptide1)
        seq2_embed = self._process_single_protein(ckesmc2, aac2, dipeptide2)
        
        seq1_proj = seq2_proj = dist = None
        if self.use_contrastive_loss:
            seq1_proj = self.projector(seq1_embed)
            seq2_proj = self.projector(seq2_embed)
            dist = F.cosine_similarity(seq1_proj, seq2_proj, dim=-1)
        
        attention_weights = {}
        residue_scores = [None, None]
        
        interaction_embed1 = torch.zeros_like(seq1_embed)
        interaction_embed2 = torch.zeros_like(seq2_embed)
        if self.use_cross_attention and esmc1 is not None and esmc2 is not None:
            if esmc_lengths1 is None:
                esmc_lengths1 = torch.full(
                    (esmc1.size(0),), esmc1.size(1),
                    dtype=torch.long, device=esmc1.device
                )
            if esmc_lengths2 is None:
                esmc_lengths2 = torch.full(
                    (esmc2.size(0),), esmc2.size(1),
                    dtype=torch.long, device=esmc2.device
                )
            valid1 = self._valid_mask(esmc_lengths1, esmc1.size(1), esmc1.device)
            valid2 = self._valid_mask(esmc_lengths2, esmc2.size(1), esmc2.device)

            # 同一cross-attention模块双向调用，使两个蛋白均获得partner context。
            context12, cross_attn = self.cross_attention(
                esmc1, esmc2, valid1, valid2
            )
            context21, cross_attn_reverse = self.cross_attention(
                esmc2, esmc1, valid2, valid1
            )

            interaction_embed1 = self.Prelu(
                self.cross_context1(self._masked_mean(context12, valid1))
            )
            interaction_embed2 = self.Prelu(
                self.cross_context2(self._masked_mean(context21, valid2))
            )

            residue_scores1 = self._calculate_residue_scores(
                cross_attn, valid1, valid2, signal_peptide_length=20
            )
            residue_scores2 = self._calculate_residue_scores(
                cross_attn_reverse, valid2, valid1, signal_peptide_length=0
            )
            residue_scores = [residue_scores1, residue_scores2]
            
            attention_weights = {
                'cross_attn': cross_attn,
                'cross_attn_reverse': cross_attn_reverse,
                'valid_mask1': valid1,
                'valid_mask2': valid2,
            }

        global_features = torch.cat([seq1_embed, seq2_embed], dim=1)
        interaction_features = torch.cat(
            [interaction_embed1, interaction_embed2], dim=1
        )
        global_raw_logit = self.global_clf(global_features)
        interaction_raw_logit = interaction_gate = None
        if self.use_cross_attention:
            interaction_raw_logit = self.interaction_clf(interaction_features)
            interaction_gate = torch.sigmoid(self.interaction_gate_logit)
        if disable_cross_attention or not self.use_cross_attention:
            raw_logit = global_raw_logit
        else:
            raw_logit = (
                global_raw_logit + interaction_gate * interaction_raw_logit
            )
        logits = torch.sigmoid(raw_logit)
        fusion_features = global_features
        if self.use_cross_attention:
            fusion_features = torch.cat(
                [global_features, interaction_gate * interaction_features], dim=1
            )
        fusion_vector = fusion_features.detach().cpu().numpy()
        
        return {
            'logits': logits, 
            'raw_logits': raw_logit,
            'global_raw_logits': global_raw_logit,
            'interaction_raw_logits': interaction_raw_logit,
            'interaction_gate': interaction_gate,
            'dist': dist, 
            'embeddings': [seq1_embed, seq2_embed],
            'interaction_embeddings': [interaction_embed1, interaction_embed2],
            'projection': [seq1_proj, seq2_proj],
            'residue_scores': residue_scores,
            'attention_weights': attention_weights,
            'fusion_vector': fusion_vector  
        }
def pairwise_contrastive_loss(feature1, feature2, labels, margin=0.2,
                              positive_weight=1.0, negative_weight=1.0):
    """Use only experimentally labeled pairs; unknown cross-pairs are not negatives."""
    if not -1.0 <= margin <= 1.0:
        raise ValueError("contrastive margin must be within [-1, 1]")
    if positive_weight < 0 or negative_weight < 0:
        raise ValueError("contrastive class weights must be nonnegative")
    if feature1.shape != feature2.shape or feature1.shape[0] != labels.numel():
        raise ValueError("Projection and label batch sizes differ")
    similarity = (F.normalize(feature1, dim=-1) *
                  F.normalize(feature2, dim=-1)).sum(dim=-1)
    positive = labels.reshape(-1) == 1
    negative = ~positive
    zero = similarity.sum() * 0.0
    positive_loss = (1.0 - similarity[positive]).mean() if positive.any() else zero
    negative_loss = F.relu(similarity[negative] - margin).square().mean() if negative.any() else zero
    return positive_weight * positive_loss + negative_weight * negative_loss
