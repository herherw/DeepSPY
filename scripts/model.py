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
        
        
        q = self.query(x) 
        k = self.key(x)   
        v = self.value(x)  
        
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / (esmc_dim ** 0.5)  

        
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        
        context = torch.matmul(attn, v)  
        return context, attn

class CrossAttention(nn.Module):
    def __init__(self, emb_dim=1152, dropout=0.05):
        super(CrossAttention, self).__init__()
        self.query = nn.Linear(emb_dim, emb_dim)
        self.key = nn.Linear(emb_dim, emb_dim)
        self.value = nn.Linear(emb_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x1, x2, mask=None):
        batch_size, seq_len1, esmc_dim = x1.size()
        _, seq_len2, _ = x2.size()
        
        q = self.query(x1)  
        k = self.key(x2)    
        v = self.value(x2)  
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / (esmc_dim ** 0.5)  
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        context = torch.matmul(attn, v)  
        
        return context, attn
        
class Model(nn.Module):
    def __init__(self, embed_hid_dim=128, dropout=0.2):
        super(Model, self).__init__()
        self.drop = dropout  
        self.ckesmc_size = 5760  
        self.esmc_dim = 1152        
        self.dropout = dropout
        self.embed_hid_dim = embed_hid_dim

        self.esmc_conv = Convs(self.ckesmc_size)
        self.fc_ad = torch.nn.Linear(420, 128) 
        self.rep = torch.nn.Linear(256, 128)    
        self.Prelu = nn.PReLU()
        self.cross_attention = CrossAttention(self.esmc_dim, dropout)
        
        self.clf = nn.Sequential(
            nn.Linear(embed_hid_dim * 2, embed_hid_dim),
            nn.ReLU(),
            nn.Dropout(self.drop),
            nn.Linear(embed_hid_dim, 1),
            nn.Sigmoid()
        )
      
        self.projector = nn.Sequential(
            nn.Linear(embed_hid_dim, embed_hid_dim), 
            nn.BatchNorm1d(embed_hid_dim), 
            nn.ReLU(), 
            nn.Linear(embed_hid_dim, embed_hid_dim)
        )
        self.dropout_layer = nn.Dropout(self.drop)

    def _process_single_protein(self, ckesmc, aac, dipeptide):
        esm_rep = self.esmc_conv(ckesmc)
        ad = torch.cat([aac, dipeptide], dim=1)
        ad_pro = self.Prelu(self.fc_ad(ad))
        
        seq_rep = torch.cat([esm_rep, ad_pro], dim=1)
        return self.Prelu(self.rep(seq_rep))
    
    def _calculate_residue_scores(
        self, cross_attention, esmc_lengths=None, signal_peptide_length=0
    ):
        score_cross: [B, L]
        topk_score = torch.topk(cross_attention, k=10, dim=2).values.mean(dim=2)  
        residue_scores = topk_score
        residue_scores = (residue_scores - residue_scores.min(dim=1, keepdim=True)[0]) / (
            residue_scores.max(dim=1, keepdim=True)[0] - residue_scores.min(dim=1, keepdim=True)[0] + 1e-6
        )
        residue_scores[:, :signal_peptide_length] = 0
        if esmc_lengths is not None:
            for i, length in enumerate(esmc_lengths):
                residue_scores[i, length:] = 0
        return residue_scores

    def _create_attention_mask(self, seq_lengths, max_len):

        batch_size = seq_lengths.size(0)
        mask = torch.ones(batch_size, max_len, max_len, device=seq_lengths.device)
        
        for i, length in enumerate(seq_lengths):
            mask[i, :length, :length] = 0  
            
        return mask.bool()
    
    def forward(self, ckesmc1, ckesmc2, aac1, aac2, dipeptide1, dipeptide2, 
                esmc1=None, esmc2=None, esmc_lengths1=None, esmc_lengths2=None):
        
        seq1_embed = self._process_single_protein(ckesmc1, aac1, dipeptide1)
        seq2_embed = self._process_single_protein(ckesmc2, aac2, dipeptide2)
        
        combined_features = torch.cat([seq1_embed, seq2_embed], dim=1)
        logits = self.clf(combined_features)
        
        seq1_proj = self.projector(seq1_embed)
        seq2_proj = self.projector(seq2_embed)
        dist = F.cosine_similarity(seq1_proj, seq2_proj, dim=-1)
        
        fusion_vector = combined_features.detach().cpu().numpy() 
    
        attention_weights = {}
        residue_scores = [None, None]
        
        if esmc1 is not None and esmc2 is not None:
            if esmc_lengths1 is not None and esmc_lengths2 is not None:
                mask1 = self._create_attention_mask(esmc_lengths1, esmc1.size(1))
                mask2 = self._create_attention_mask(esmc_lengths2, esmc2.size(1))
            else:
                mask1 = None
                mask2 = None
                        
            _, cross_attn = self.cross_attention(esmc1, esmc2)
            
            residue_scores1 = self._calculate_residue_scores(cross_attn, esmc_lengths1, signal_peptide_length=20)
            residue_scores2 = self._calculate_residue_scores(cross_attn.transpose(1, 2), esmc_lengths2, signal_peptide_length=0)  # protein2 无信号肽
            
            if esmc_lengths1 is not None:
                for i, length in enumerate(esmc_lengths1):
                    residue_scores1[i, length:] = 0
                    
            if esmc_lengths2 is not None:
                for i, length in enumerate(esmc_lengths2):
                    residue_scores2[i, length:] = 0
                    
            residue_scores = [residue_scores1, residue_scores2]
            
            attention_weights = {
                'cross_attn': cross_attn
            }
        
        return {
            'logits': logits, 
            'dist': dist, 
            'embeddings': [seq1_embed, seq2_embed],
            'projection': [seq1_proj, seq2_proj],
            'residue_scores': residue_scores,
            'attention_weights': attention_weights,
            'fusion_vector': fusion_vector  
        }
def sup_contrasitive_loss(feature1, feature2, labels, temperature=1):
    sim_matrix = torch.nn.functional.cosine_similarity(feature1.unsqueeze(1), feature2.unsqueeze(0), dim=2)
    positive_mask = torch.diag(labels.float())
    p=positive_mask.sum()   
    positive_sim = sim_matrix * positive_mask
    positive= positive_sim.sum(1)
    numerator = torch.exp(positive / temperature)

    negative_mask=torch.ones_like(sim_matrix)-positive_mask
    negative_mask=negative_mask*(sim_matrix<0.7)

    negative_sim = sim_matrix * negative_mask
    denominator = torch.sum(torch.exp(negative_sim / temperature)) 
    loss = -torch.log(numerator / denominator)/p  
    return loss.mean()

def attention_consistency_loss(cross_attn, weight_factor=2.0):
    B, L1, L2 = cross_attn.shape
    eps = 1e-8
    weighted_attn = cross_attn ** weight_factor
    weighted_attn = weighted_attn / (weighted_attn.sum(dim=2, keepdim=True) + eps)
    entropy = -(weighted_attn * torch.log(weighted_attn + eps)).sum(dim=2) / torch.log(torch.tensor(L2 + eps, device=cross_attn.device))
    avg_entropy = entropy.mean(dim=1)  
    return avg_entropy.mean()
