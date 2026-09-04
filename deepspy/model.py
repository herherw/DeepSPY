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
        
        # Compute self-attention
        q = self.query(x) 
        k = self.key(x)   
        v = self.value(x)  
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (esmc_dim ** 0.5)  

        # Apply the mask
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Compute weighted features
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
        # Mask partner padding before the softmax normalization.
        if key_mask is not None:
            scores = scores.masked_fill(
                ~key_mask[:, None, :], torch.finfo(scores.dtype).min
            )
        
        attn = F.softmax(scores, dim=-1)
        # Padded query rows must not produce context vectors or residue scores.
        if query_mask is not None:
            attn = attn * query_mask[:, :, None].to(attn.dtype)
        if key_mask is not None:
            attn = attn * key_mask[:, None, :].to(attn.dtype)
        attn = self.dropout(attn)
        
        context = torch.matmul(attn, v)  
        
        return context, attn
        
class Model(nn.Module):
    def __init__(self, embed_hid_dim=128, dropout=0.2,
                 interaction_gate_init=-2.0, use_aac_dpc=True,
                 use_cksaap_esmc=True, use_contrastive_loss=True,
                 use_cross_attention=True):
        super(Model, self).__init__()
        self.drop = dropout  
        self.ckesmc_size = 5760  
        self.esmc_dim = 1152        
        self.dropout = dropout
        self.embed_hid_dim = embed_hid_dim
        self.use_aac_dpc = bool(use_aac_dpc)
        self.use_cksaap_esmc = bool(use_cksaap_esmc)
        self.use_contrastive_loss = bool(use_contrastive_loss)
        self.use_cross_attention = bool(use_cross_attention)
        if not self.use_aac_dpc and not self.use_cksaap_esmc:
            raise ValueError(
                "At least one global branch (AAC-DPC or CKSAAP-ESMC) "
                "must remain enabled"
            )

        # Legacy-style structural ablation: an inactive branch is not
        # instantiated.  With both branches enabled, B0 follows the unchanged
        # full-model path.  With one branch enabled, its native 128-D output is
        # used directly, matching the 20260109 ablation implementation logic.
        if self.use_cksaap_esmc:
            self.esmc_conv = Convs(self.ckesmc_size)
        if self.use_aac_dpc:
            self.fc_ad = torch.nn.Linear(420, 128)
        if self.use_aac_dpc and self.use_cksaap_esmc:
            self.rep = torch.nn.Linear(256, 128)
        self.Prelu = nn.PReLU()
        if self.use_cross_attention:
            self.cross_attention = CrossAttention(self.esmc_dim, dropout)
            # Cross-attention contributes through the interaction branch; the sequence encoder is unchanged.
            self.cross_context1 = nn.Linear(self.esmc_dim, embed_hid_dim)
            self.cross_context2 = nn.Linear(self.esmc_dim, embed_hid_dim)

        # Separate the global decision from the gated cross-attention increment.
        self.global_clf = nn.Sequential(
            nn.Linear(embed_hid_dim * 2, embed_hid_dim),
            nn.ReLU(),
            nn.Dropout(self.drop),
            nn.Linear(embed_hid_dim, 1)
        )
        if self.use_cross_attention:
            self.interaction_clf = nn.Sequential(
                nn.Linear(embed_hid_dim * 2, embed_hid_dim),
                nn.ReLU(),
                nn.Dropout(self.drop),
                nn.Linear(embed_hid_dim, 1)
            )
            self.interaction_gate_logit = nn.Parameter(
                torch.tensor(float(interaction_gate_init))
            )
      
        if self.use_contrastive_loss:
            self.projector = nn.Sequential(
                nn.Linear(embed_hid_dim, embed_hid_dim),
                nn.BatchNorm1d(embed_hid_dim),
                nn.ReLU(),
                nn.Linear(embed_hid_dim, embed_hid_dim)
            )
        self.dropout_layer = nn.Dropout(self.drop)

    def _process_single_protein(self, ckesmc, aac, dipeptide):
        if self.use_cksaap_esmc and not self.use_aac_dpc:
            return self.esmc_conv(ckesmc)
        ad = torch.cat([aac, dipeptide], dim=1)
        ad_pro = self.Prelu(self.fc_ad(ad))
        if self.use_aac_dpc and not self.use_cksaap_esmc:
            return ad_pro
        esm_rep = self.esmc_conv(ckesmc)
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
        
        if self.use_contrastive_loss:
            seq1_proj = self.projector(seq1_embed)
            seq2_proj = self.projector(seq2_embed)
            dist = F.cosine_similarity(seq1_proj, seq2_proj, dim=-1)
        else:
            seq1_proj = None
            seq2_proj = None
            dist = None
        
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

            # Apply the shared cross-attention module bidirectionally so each protein receives partner context.
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
        if self.use_cross_attention:
            interaction_raw_logit = self.interaction_clf(interaction_features)
            interaction_gate = torch.sigmoid(self.interaction_gate_logit)
        else:
            interaction_raw_logit = torch.zeros_like(global_raw_logit)
            interaction_gate = torch.zeros(
                (), device=global_raw_logit.device,
                dtype=global_raw_logit.dtype
            )
        if disable_cross_attention or not self.use_cross_attention:
            raw_logit = global_raw_logit
        else:
            raw_logit = (
                global_raw_logit + interaction_gate * interaction_raw_logit
            )
        logits = torch.sigmoid(raw_logit)
        if self.use_cross_attention:
            fusion_features = torch.cat(
                [global_features, interaction_gate * interaction_features], dim=1
            )
        else:
            fusion_features = global_features
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
def sup_contrasitive_loss(feature1, feature2, labels, temperature=1):
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")

    sim_matrix = torch.nn.functional.cosine_similarity(
        feature1.unsqueeze(1), feature2.unsqueeze(0), dim=2
    )
    positive_mask = torch.diag(labels.float())
    positive_count = positive_mask.sum()

    # A shuffled imbalanced batch can contain no positive pairs.  Dividing by
    # p=0 corrupted the model with NaN parameters.  Such a batch contains no
    # supervised positive contrastive signal, so its contrastive contribution
    # is a differentiable zero; the main and interaction BCE terms still train
    # on every sample in the batch.
    if positive_count.detach().item() == 0:
        return sim_matrix.sum() * 0.0

    positive_sim = sim_matrix * positive_mask
    positive = positive_sim.sum(1)

    negative_mask = torch.ones_like(sim_matrix) - positive_mask
    negative_mask = negative_mask * (sim_matrix < 0.7)

    negative_sim = sim_matrix * negative_mask
    # Equivalent to log(sum(exp(...))) for finite inputs, but numerically stable.
    log_denominator = torch.logsumexp(
        (negative_sim / temperature).reshape(-1), dim=0
    )
    loss = -(
        positive / temperature - log_denominator
    ) / positive_count
    return loss.mean()
