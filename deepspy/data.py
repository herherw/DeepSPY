#!/usr/bin/env python
# data_loader.py
import torch
import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset

def get_device():
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def pad_sequences(sequences, max_len=None, padding_value=0):

    if max_len is None:
        max_len = max([seq.size(0) for seq in sequences])

    batch_size = len(sequences)
    feature_dim = sequences[0].size(1)
    output = torch.ones(batch_size, max_len, feature_dim) * padding_value

    for i, seq in enumerate(sequences):
        length = min(seq.size(0), max_len)  
        output[i, :length] = seq[:length]
        
    return output

class Dataset(Dataset):
    def __init__(self, pairs, labels, config):
        super(Dataset, self).__init__()
        self.pairs = pairs
        self.labels = labels
        self.config = config
        
        self.aac_rep1 = np.load(config['data']['pro1_aac_path'], allow_pickle=True)
        self.pro1_aac_dict = dict(zip(self.aac_rep1.files, [self.aac_rep1[f] for f in self.aac_rep1.files]))
        self.aac_rep2 = np.load(config['data']['pro2_aac_path'], allow_pickle=True)
        self.pro2_aac_dict = dict(zip(self.aac_rep2.files, [self.aac_rep2[f] for f in self.aac_rep2.files]))

        self.dipeptide_rep1 = np.load(config['data']['pro1_dipeptide_path'], allow_pickle=True)
        self.pro1_dipeptide_dict = dict(zip(self.dipeptide_rep1.files, [self.dipeptide_rep1[f] for f in self.dipeptide_rep1.files]))
        self.dipeptide_rep2 = np.load(config['data']['pro2_dipeptide_path'], allow_pickle=True)
        self.pro2_dipeptide_dict = dict(zip(self.dipeptide_rep2.files, [self.dipeptide_rep2[f] for f in self.dipeptide_rep2.files]))

        self.seq_dict1 = {}
        self.seq_dict2 = {}
        self.load_protein_sequences(config)

        self.ckesmc_rep1 = np.load(
            config['data']['pro1_ckesmc_path'], allow_pickle=True
        ).item()
        self.ckesmc_rep2 = np.load(
            config['data']['pro2_ckesmc_path'], allow_pickle=True
        ).item()

        self.esmc_rep1 = np.load(
            config['data']['pro1_esmc_path'], allow_pickle=True
        ).item()
        self.esmc_rep2 = np.load(
            config['data']['pro2_esmc_path'], allow_pickle=True
        ).item()

        # Biological sequence length is the sole source of truth.  Training
        # caches must contain exactly one embedding row per biological residue.
        self.seq_lengths1 = {
            protein_id: len(sequence)
            for protein_id, sequence in self.seq_dict1.items()
        }
        self.seq_lengths2 = {
            protein_id: len(sequence)
            for protein_id, sequence in self.seq_dict2.items()
        }
        self._validate_residue_features()

    def load_protein_sequences(self, config):
        seq_file1 = os.path.join(config['data']['pro1_seq_path'])
        seq_file2 = os.path.join(config['data']['pro2_seq_path'])

        df1 = pd.read_csv(seq_file1, sep='\t')
        for _, row in df1.iterrows():
            self.seq_dict1[str(row['protein_id'])] = str(row['sequence']).strip()
        df2 = pd.read_csv(seq_file2, sep='\t')
        for _, row in df2.iterrows():
            self.seq_dict2[str(row['protein_id'])] = str(row['sequence']).strip()

    @staticmethod
    def _validate_one_side(esmc, ckesmc, sequences, protein_ids, side_name):
        for protein_id in sorted(protein_ids):
            if protein_id not in sequences:
                raise KeyError(f"{side_name}: sequence missing for {protein_id}")
            if protein_id not in esmc:
                raise KeyError(
                    f"{side_name}: residue-only ESM-C missing for {protein_id}"
                )
            if protein_id not in ckesmc:
                raise KeyError(f"{side_name}: CKSAAP-ESMC missing for {protein_id}")

            sequence_length = len(sequences[protein_id])
            embedding = np.asarray(esmc[protein_id])
            if embedding.ndim != 2:
                raise ValueError(
                    f"{side_name}/{protein_id}: expected 2-D ESM-C embedding, "
                    f"got shape {embedding.shape}"
                )
            if embedding.shape[0] != sequence_length:
                hint = (
                    " (BOS/EOS appear to be present; run "
                    "prepare_residue_features.py)"
                    if embedding.shape[0] == sequence_length + 2 else ""
                )
                raise ValueError(
                    f"{side_name}/{protein_id}: ESM-C length "
                    f"{embedding.shape[0]} != biological sequence length "
                    f"{sequence_length}{hint}"
                )

            ckesmc_shape = np.asarray(ckesmc[protein_id]).shape
            expected_ckesmc_shape = (5 * embedding.shape[1], 20, 20)
            if ckesmc_shape != expected_ckesmc_shape:
                raise ValueError(
                    f"{side_name}/{protein_id}: CKSAAP-ESMC shape "
                    f"{ckesmc_shape} != expected {expected_ckesmc_shape}"
                )

    def _validate_residue_features(self):
        protein1_ids = {str(pair).split('_', 1)[0] for pair in self.pairs}
        protein2_ids = {str(pair).split('_', 1)[1] for pair in self.pairs}
        self._validate_one_side(
            self.esmc_rep1, self.ckesmc_rep1, self.seq_dict1,
            protein1_ids, "protein1"
        )
        self._validate_one_side(
            self.esmc_rep2, self.ckesmc_rep2, self.seq_dict2,
            protein2_ids, "protein2"
        )
    def get_protein_sequence(self, protein_id, is_first_protein=True):

        seq_dict = self.seq_dict1 if is_first_protein else self.seq_dict2

        if protein_id in seq_dict:
            return seq_dict[protein_id]
        seq_lengths = self.seq_lengths1 if is_first_protein else self.seq_lengths2
        seq_len = seq_lengths.get(protein_id, 0)
        if seq_len > 0:
            return "X" * seq_len
        
        return "SEQUENCE_UNAVAILABLE"

    def load_protein_data(self, pid):
        device = get_device() 
        name1, name2 = pid.split('_')

        ckesmc1 = torch.tensor(self.ckesmc_rep1[name1]).float()
        ckesmc2 = torch.tensor(self.ckesmc_rep2[name2]).float()

        aac1 = torch.tensor(self.pro1_aac_dict[name1]).float().to(device)
        aac2 = torch.tensor(self.pro2_aac_dict[name2]).float().to(device)
        dipeptide1 = torch.tensor(self.pro1_dipeptide_dict[name1]).float().to(device)
        dipeptide2 = torch.tensor(self.pro2_dipeptide_dict[name2]).float().to(device)
        
        esmc1 = torch.tensor(self.esmc_rep1[name1]).float()
        esmc2 = torch.tensor(self.esmc_rep2[name2]).float()

        return ckesmc1, ckesmc2, aac1, aac2, dipeptide1, dipeptide2, esmc1, esmc2, name1, name2

    @staticmethod
    def load_pair(pair_path):
        frame = pd.read_csv(pair_path, sep='\t')
        required = {'pair_id', 'label'}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(
                f"{pair_path}: missing required columns {sorted(missing)}"
            )
        return frame['pair_id'].astype(str).tolist(), frame['label'].astype(int).tolist()


    def __getitem__(self, idx):
        pair = self.pairs[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        ckesmc1, ckesmc2, aac1, aac2, dip1, dip2, esmc1, esmc2, name1, name2 = self.load_protein_data(pair)        
        return ckesmc1, ckesmc2, aac1, aac2, dip1, dip2, esmc1, esmc2, name1, name2, label

    def __len__(self):
        return len(self.pairs)

def collate_fn(batch):
    ckesmc1, ckesmc2, aac1, aac2, dip1, dip2, esmc1, esmc2, name1, name2, labels = zip(*batch)

    ckesmc1_batch = torch.stack(ckesmc1)
    ckesmc2_batch = torch.stack(ckesmc2)
    aac1_batch = torch.stack(aac1)
    aac2_batch = torch.stack(aac2)
    dip1_batch = torch.stack(dip1)
    dip2_batch = torch.stack(dip2)
    
    esmc1_lengths = torch.tensor([len(seq) for seq in esmc1])
    esmc2_lengths = torch.tensor([len(seq) for seq in esmc2])
    
    labels_batch = torch.stack(labels)
    esmc1_batch = pad_sequences(esmc1)
    esmc2_batch = pad_sequences(esmc2)
    
    return (
        ckesmc1_batch, ckesmc2_batch, 
        aac1_batch, aac2_batch, 
        dip1_batch, dip2_batch,
        esmc1_batch, esmc2_batch, 
        esmc1_lengths, esmc2_lengths,
        name1, name2, 
        labels_batch
    )
