#!/usr/bin/env python
# predict.py
import torch
import numpy as np
import os
import argparse
import subprocess
import tempfile
import shutil
import datetime
import pandas as pd
from collections import Counter
from model import Model
from data_loader import get_device, pad_sequences
from util import get_config, analyze_key_residues, visualize_attention_heatmap, save_key_residues_report

class ProteinFeatureGenerator:

    def __init__(self, config):
        self.config = config
        self.device = get_device()
        
    def calculate_AAC(self, sequence):
        """计算氨基酸组成（AAC）"""
        amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        counts = Counter(sequence)
        total_count = len(sequence)
        aac = [counts.get(aa, 0) / total_count for aa in amino_acids]
        return aac

    def calculate_dipeptide_composition(self, sequence):

        amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        dipeptides = [sequence[i:i+2] for i in range(len(sequence)-1)]
        dipeptide_counts = Counter(dipeptides)
        total_dipeptides = len(dipeptides)
        dipeptide_composition = [dipeptide_counts.get(aa, 0) / total_dipeptides 
                               for aa in (a+b for a in amino_acids for b in amino_acids)]
        return dipeptide_composition
    
    def generate_basic_features(self, protein_id, sequence):

        aac = self.calculate_AAC(sequence)
        dipeptide = self.calculate_dipeptide_composition(sequence)
        
        return {
            'aac': torch.tensor(aac).float(),
            'dipeptide': torch.tensor(dipeptide).float()
        }
    
    def generate_esmc_features_standalone(self, protein_id, sequence):
            
        import sys
        esmc_path = "/mnt/data2/model"
        if esmc_path not in sys.path:
            sys.path.append(esmc_path)
        from esm.models.esmc import ESMC
        from esm.sdk.api import ESMProtein, LogitsConfig
        import torch.nn as nn
        import math
        from itertools import product

        class PositionalEncoding(nn.Module):
            def __init__(self, d_model, max_len=5000):
                super(PositionalEncoding, self).__init__()
                pe = torch.zeros(max_len, d_model, device=self.device)
                position = torch.arange(0, max_len, dtype=torch.float, device=self.device).unsqueeze(1)
                div_term = torch.exp(torch.arange(0, d_model, 2, device=self.device).float() * (-math.log(10000.0) / d_model))
                pe[:, 0::2] = torch.sin(position * div_term)
                pe[:, 1::2] = torch.cos(position * div_term)
                pe = pe.unsqueeze(0).transpose(0, 1)
                self.register_buffer('pe', pe)

            def forward(self, x):
                return x + self.pe[:x.size(0), :]

            class CKSAAP(nn.Module):
                def __init__(self, device, is_use_position=False, position_d_model=None):
                    super(CKSAAP, self).__init__()
                    self.device = device
                    AA = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
                    DP = list(product(AA, AA))
                    self.DP_list = [str(i[0]) + str(i[1]) for i in DP]
                    self.position_func = None
                    self.position_d_model = position_d_model
                    if is_use_position:
                        if position_d_model is None:
                            self.position_d_model = 16
                        self.position_func = PositionalEncoding(d_model=self.position_d_model).to(device)

                def return_CKSAAP_Emb_code(self, query_seq, emb, k=3, is_shape_for_3d=False):
                    code_final = []
                    for turns in range(k + 1):
                        DP_dic = {i: torch.zeros(emb.size(-1), device=self.device) for i in self.DP_list}
                        for i in range(len(query_seq) - turns - 1):
                            tmp_dp = query_seq[i] + query_seq[i + turns + 1]
                            tmp_emb = 0.5 * (emb[i] + emb[i + turns + 1])
                            DP_dic[tmp_dp] = DP_dic.get(tmp_dp, torch.zeros(emb.size(-1), device=self.device)) + tmp_emb
                        code = [DP_dic[i] / (len(query_seq) - turns - 1) for i in self.DP_list]
                        code_final += code
                    code_final = torch.stack(code_final).view(k + 1, 20, 20, -1)
                    if is_shape_for_3d:
                        k_plus_one, aa_num_1, aa_num_2, position_posi_emb_size = code_final.size()
                        code_final = code_final.permute(0, 3, 1, 2).contiguous().view(k_plus_one * position_posi_emb_size, aa_num_1, aa_num_2)
                    return code_final.to(self.device)

            class ESMCFineTune(nn.Module):
                def __init__(self, device):
                    super(ESMCFineTune, self).__init__()
                    self.device = device
                    os.environ["INFRA_PROVIDER"] = "True"
                    print("正在加载ESMC模型...")
                    self.client = ESMC.from_pretrained("esmc_600m").to(device)
                    print("ESMC模型加载完成")

                def forward(self, x):
                    protein = ESMProtein(sequence=x[0][1])
                    with torch.no_grad():
                        protein_tensor = self.client.encode(protein)
                        output = self.client.logits(protein_tensor, LogitsConfig(sequence=True, return_embeddings=True))
                    return output.embeddings.squeeze()
            
            max_seq_length = 1200
            protein_sequence = sequence[:max_seq_length]  # 截断序列
            
            esmc_model = ESMCFineTune(self.device)
            cksaap = CKSAAP(self.device)
            
            seq_esm_input = [(protein_id, protein_sequence)]
            print(f"正在生成 {protein_id} 的ESMC嵌入...")
            out = esmc_model(seq_esm_input)
            textembed1 = out.to(self.device)
            
            textembed_1 = cksaap.return_CKSAAP_Emb_code(protein_sequence, textembed1, k=4, is_shape_for_3d=True)
            
            
            return {
                'ckesmc': textembed_1.cpu(),
                'esmc': textembed1.cpu(),
                'seq_length': len(protein_sequence)
            }
            
        except Exception as e:
            return None
    

class ProteinPairPredictor:
    def __init__(self, model_path, config_path):

        self.config = get_config(config_path)
        self.device = get_device()
        self.feature_generator = ProteinFeatureGenerator(self.config)
        
        self.model = Model(
            embed_hid_dim=self.config['model']['embed_hid_dim'],
            dropout=self.config['model']['dropout']
        ).to(self.device)
        

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        self.load_existing_features()
        
        self.new_features_cache = {
            'pro1_aac': {}, 'pro1_dipeptide': {}, 'pro1_ckesmc': {}, 'pro1_esmc': {}, 'pro1_seq_lengths': {},
            'pro2_aac': {}, 'pro2_dipeptide': {}, 'pro2_ckesmc': {}, 'pro2_esmc': {}, 'pro2_seq_lengths': {}
        }
    
    def load_existing_features(self):
        
        try:
            self.aac_rep1 = np.load(self.config['data']['pro1_aac_path'], allow_pickle=True)
            self.pro1_aac_dict = dict(zip(self.aac_rep1.files, [self.aac_rep1[f] for f in self.aac_rep1.files]))
            self.aac_rep2 = np.load(self.config['data']['pro2_aac_path'], allow_pickle=True)
            self.pro2_aac_dict = dict(zip(self.aac_rep2.files, [self.aac_rep2[f] for f in self.aac_rep2.files]))

            self.dipeptide_rep1 = np.load(self.config['data']['pro1_dipeptide_path'], allow_pickle=True)
            self.pro1_dipeptide_dict = dict(zip(self.dipeptide_rep1.files, [self.dipeptide_rep1[f] for f in self.dipeptide_rep1.files]))
            self.dipeptide_rep2 = np.load(self.config['data']['pro2_dipeptide_path'], allow_pickle=True)
            self.pro2_dipeptide_dict = dict(zip(self.dipeptide_rep2.files, [self.dipeptide_rep2[f] for f in self.dipeptide_rep2.files]))

            self.ckesmc_rep1 = np.load(self.config['data']['pro1_ckesmc_path'], allow_pickle=True).item()
            self.ckesmc_rep2 = np.load(self.config['data']['pro2_ckesmc_path'], allow_pickle=True).item()
            
            self.esmc_rep1 = np.load(self.config['data']['pro1_esmc_path'], allow_pickle=True).item()
            self.esmc_rep2 = np.load(self.config['data']['pro2_esmc_path'], allow_pickle=True).item()
            
            self.seq_lengths1 = np.load(self.config['data']['pro1_esmc_path'].replace('.npy', '_seq_lengths.npy'), 
                                      allow_pickle=True).item()
            self.seq_lengths2 = np.load(self.config['data']['pro2_esmc_path'].replace('.npy', '_seq_lengths.npy'), 
                                      allow_pickle=True).item()
            
            
        except Exception as e:

            self.pro1_aac_dict = {}
            self.pro2_aac_dict = {}
            self.pro1_dipeptide_dict = {}
            self.pro2_dipeptide_dict = {}
            self.ckesmc_rep1 = {}
            self.ckesmc_rep2 = {}
            self.esmc_rep1 = {}
            self.esmc_rep2 = {}
            self.seq_lengths1 = {}
            self.seq_lengths2 = {}
    
    def get_or_generate_features(self, protein_name, protein_sequence, is_first_protein=True):

        cache_prefix = 'pro1' if is_first_protein else 'pro2'
        
        aac_dict = self.pro1_aac_dict if is_first_protein else self.pro2_aac_dict
        dipeptide_dict = self.pro1_dipeptide_dict if is_first_protein else self.pro2_dipeptide_dict
        ckesmc_dict = self.ckesmc_rep1 if is_first_protein else self.ckesmc_rep2
        esmc_dict = self.esmc_rep1 if is_first_protein else self.esmc_rep2
        seq_lengths = self.seq_lengths1 if is_first_protein else self.seq_lengths2
        
        if protein_name in self.new_features_cache[f'{cache_prefix}_aac']:
            return {
                'aac': self.new_features_cache[f'{cache_prefix}_aac'][protein_name],
                'dipeptide': self.new_features_cache[f'{cache_prefix}_dipeptide'][protein_name],
                'ckesmc': self.new_features_cache[f'{cache_prefix}_ckesmc'][protein_name],
                'esmc': self.new_features_cache[f'{cache_prefix}_esmc'][protein_name],
                'seq_length': self.new_features_cache[f'{cache_prefix}_seq_lengths'][protein_name]
            }
        

        if (protein_name in aac_dict and protein_name in dipeptide_dict and 
            protein_name in ckesmc_dict and protein_name in esmc_dict):

            return {
                'aac': torch.tensor(aac_dict[protein_name]).float(),
                'dipeptide': torch.tensor(dipeptide_dict[protein_name]).float(),
                'ckesmc': torch.tensor(ckesmc_dict[protein_name]).float(),
                'esmc': torch.tensor(esmc_dict[protein_name]).float(),
                'seq_length': seq_lengths.get(protein_name, len(protein_sequence))
            }

        
        basic_features = self.feature_generator.generate_basic_features(protein_name, protein_sequence)

        esmc_features = self.feature_generator.generate_esmc_features_standalone(protein_name, protein_sequence)

        self.new_features_cache[f'{cache_prefix}_aac'][protein_name] = basic_features['aac']
        self.new_features_cache[f'{cache_prefix}_dipeptide'][protein_name] = basic_features['dipeptide']
        self.new_features_cache[f'{cache_prefix}_ckesmc'][protein_name] = esmc_features['ckesmc']
        self.new_features_cache[f'{cache_prefix}_esmc'][protein_name] = esmc_features['esmc']
        self.new_features_cache[f'{cache_prefix}_seq_lengths'][protein_name] = esmc_features['seq_length']
        
        return {
            'aac': basic_features['aac'],
            'dipeptide': basic_features['dipeptide'],
            'ckesmc': esmc_features['ckesmc'],
            'esmc': esmc_features['esmc'],
            'seq_length': esmc_features['seq_length']
        }
    
    def predict_with_sequences(self, protein1_name, protein1_sequence, 
                             protein2_name, protein2_sequence, 
                             threshold=0.5, out_dir="./prediction_results", 
                             return_details=False):

        try:

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            task_dir = os.path.join(out_dir, f"{protein1_name}_{protein2_name}_{timestamp}")
            os.makedirs(task_dir, exist_ok=True)


            features1 = self.get_or_generate_features(protein1_name, protein1_sequence, is_first_protein=True)
            features2 = self.get_or_generate_features(protein2_name, protein2_sequence, is_first_protein=False)
            
            ckesmc1 = features1['ckesmc'].unsqueeze(0).to(self.device)
            ckesmc2 = features2['ckesmc'].unsqueeze(0).to(self.device)
            aac1 = features1['aac'].unsqueeze(0).to(self.device)
            aac2 = features2['aac'].unsqueeze(0).to(self.device)
            dipeptide1 = features1['dipeptide'].unsqueeze(0).to(self.device)
            dipeptide2 = features2['dipeptide'].unsqueeze(0).to(self.device)
            esmc1 = features1['esmc'].unsqueeze(0).to(self.device)
            esmc2 = features2['esmc'].unsqueeze(0).to(self.device)
            
            esmc1_lengths = torch.tensor([features1['seq_length']]).to(self.device)
            esmc2_lengths = torch.tensor([features2['seq_length']]).to(self.device)
            
            print("Predicting...")
            with torch.no_grad():
                outputs = self.model(
                    ckesmc1, ckesmc2, aac1, aac2, dipeptide1, dipeptide2,
                    esmc1, esmc2, esmc1_lengths, esmc2_lengths
                )
                
                probability = outputs['logits'].squeeze().cpu().item()
                
                csv_path = os.path.join(task_dir, "result.csv")
                pair_name = f"{protein1_name}_{protein2_name}"
                df = pd.DataFrame([[pair_name, probability]], columns=["Protein Pair", "Predicted Interaction Probability"])
                df.to_csv(csv_path, index=False)
                
                result = {
                    'success': True,
                    'protein1': protein1_name,
                    'protein2': protein2_name,
                    'protein1_sequence': protein1_sequence,
                    'protein2_sequence': protein2_sequence,
                    'interaction_probability': probability,
                    'predicted_interaction': probability >= threshold,
                    'confidence': 'High' if abs(probability - 0.5) > 0.3 else 'Medium' if abs(probability - 0.5) > 0.1 else 'Low',
                    'task_dir': task_dir
                }
                
                if probability >= threshold:
                    print("[INFO] The sample is predicted to be a fuctional non-self complex. Visual analysis is being generated...")
                    
                    score1 = outputs['residue_scores'][0][0].cpu().numpy()
                    score2 = outputs['residue_scores'][1][0].cpu().numpy()
                    attn = outputs['attention_weights']['cross_attn'][0].cpu().numpy()

                    key_res1 = analyze_key_residues(
                        protein1_sequence, score1, 
                        os.path.join(task_dir, f"{protein1_name}_key_residues.png"), 
                        protein1_name
                    )
                    key_res2 = analyze_key_residues(
                        protein2_sequence, score2, 
                        os.path.join(task_dir, f"{protein2_name}_key_residues.png"), 
                        protein2_name
                    )

                    save_key_residues_report(
                        key_res1, protein1_sequence, 
                        os.path.join(task_dir, f"{protein1_name}_key_residues.txt"), 
                        protein1_name
                    )
                    save_key_residues_report(
                        key_res2, protein2_sequence, 
                        os.path.join(task_dir, f"{protein2_name}_key_residues.txt"), 
                        protein2_name
                    )

                    visualize_attention_heatmap(
                        sequence1=protein1_sequence,
                        sequence2=protein2_sequence,
                        attention_weights=attn,
                        save_path=os.path.join(task_dir, "attention_heatmap.png"),
                        raw_data_path=os.path.join(task_dir, "attention.csv")
                    )
                    
                else:
                    print("[INFO] The sample is predicted to be non-fuctional complex, so residue analysis figure will not be generated.")
                
                if return_details:
                    result.update({
                        'residue_scores1': score1 if probability >= threshold else None,
                        'residue_scores2': score2 if probability >= threshold else None,
                        'attention_weights': attn if probability >= threshold else None,
                        'sequence_length1': features1['seq_length'],
                        'sequence_length2': features2['seq_length']
                    })
                
                return result
                
        except Exception as e:
            return {
                'success': False,
                'error': f"Failed: {str(e)}",
                'protein1': protein1_name,
                'protein2': protein2_name
            }
    
    def predict_single_pair(self, protein1_name, protein2_name, 
                           threshold=0.5, out_dir="./prediction_results", 
                           return_details=False):

        aac_dict1 = self.pro1_aac_dict
        aac_dict2 = self.pro2_aac_dict
        
        missing_proteins = []
        if protein1_name not in aac_dict1:
            missing_proteins.append(f"protein 1: {protein1_name}")
        if protein2_name not in aac_dict2:
            missing_proteins.append(f"protein 2: {protein2_name}")
            
        if missing_proteins:
            return {
                'success': False,
                'error': f"Feature data for the following proteins is missing: {', '.join(missing_proteins)}. Please provide sequences using the --sequence argument",
                'protein1': protein1_name,
                'protein2': protein2_name
            }
        
        try:

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            task_dir = os.path.join(out_dir, f"{protein1_name}_{protein2_name}_{timestamp}")
            os.makedirs(task_dir, exist_ok=True)
            
            ckesmc1 = torch.tensor(self.ckesmc_rep1[protein1_name]).float()
            ckesmc2 = torch.tensor(self.ckesmc_rep2[protein2_name]).float()
            aac1 = torch.tensor(self.pro1_aac_dict[protein1_name]).float().to(self.device)
            aac2 = torch.tensor(self.pro2_aac_dict[protein2_name]).float().to(self.device)
            dipeptide1 = torch.tensor(self.pro1_dipeptide_dict[protein1_name]).float().to(self.device)
            dipeptide2 = torch.tensor(self.pro2_dipeptide_dict[protein2_name]).float().to(self.device)
            esmc1 = torch.tensor(self.esmc_rep1[protein1_name]).float()
            esmc2 = torch.tensor(self.esmc_rep2[protein2_name]).float()
            
            seq_len1 = self.seq_lengths1.get(protein1_name, esmc1.shape[0])
            seq_len2 = self.seq_lengths2.get(protein2_name, esmc2.shape[0])

            ckesmc1 = ckesmc1.unsqueeze(0).to(self.device)
            ckesmc2 = ckesmc2.unsqueeze(0).to(self.device)
            esmc1 = esmc1.unsqueeze(0).to(self.device)
            esmc2 = esmc2.unsqueeze(0).to(self.device)
            
            esmc1_lengths = torch.tensor([seq_len1]).to(self.device)
            esmc2_lengths = torch.tensor([seq_len2]).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(
                    ckesmc1, ckesmc2, aac1.unsqueeze(0), aac2.unsqueeze(0),
                    dipeptide1.unsqueeze(0), dipeptide2.unsqueeze(0),
                    esmc1, esmc2, esmc1_lengths, esmc2_lengths
                )
                
                probability = outputs['logits'].squeeze().cpu().item()
                
                csv_path = os.path.join(task_dir, "result.csv")
                pair_name = f"{protein1_name}_{protein2_name}"
                df = pd.DataFrame([[pair_name, probability]], columns=["Protein Pair", "Predicted Interaction Probability"])
                df.to_csv(csv_path, index=False)
                
                result = {
                    'success': True,
                    'protein1': protein1_name,
                    'protein2': protein2_name,
                    'interaction_probability': probability,
                    'predicted_interaction': probability >= threshold,
                    'confidence': 'High' if abs(probability - 0.5) > 0.3 else 'Medium' if abs(probability - 0.5) > 0.1 else 'Low',
                    'task_dir': task_dir
                }
                
                if return_details:
                    result.update({
                        'residue_scores1': None, 
                        'residue_scores2': None,
                        'attention_weights': None,
                        'sequence_length1': seq_len1,
                        'sequence_length2': seq_len2
                    })
                
                return result
                
        except Exception as e:
            return {
                'success': False,
                'error': f"Failed: {str(e)}",
                'protein1': protein1_name,
                'protein2': protein2_name
            }
    
    def get_available_proteins(self):
        return {
            'protein1_database': list(self.pro1_aac_dict.keys()),
            'protein2_database': list(self.pro2_aac_dict.keys())
        }

def main():
    parser = argparse.ArgumentParser(description='Predict protein-protein interactions')
    parser.add_argument('--model_path', type=str, required=True,
                        help='File path to the trained model checkpoint')
    parser.add_argument('--config_path', type=str, default='./config.yml',
                        help='File path of the configuration file')
    parser.add_argument('--protein1', type=str,
                        help='Name of the first protein')
    parser.add_argument('--protein2', type=str,
                        help='Name of the second protein')
    parser.add_argument('--sequence1', type=str,
                        help='Amino acid sequence of the first protein')
    parser.add_argument('--sequence2', type=str,
                        help='Amino acid sequence of the second protein')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Prediction threshold (default: 0.5)')
    parser.add_argument('--out_dir', type=str, default='./prediction_results',
                        help='Output directory (default: ./prediction_results)')
    parser.add_argument('--pairs_file', type=str,
                        help='File path containing protein pairs')
    parser.add_argument('--sequences_file', type=str,
                        help='File path containing protein sequences (format: name1,sequence1,name2,sequence2)')
    parser.add_argument('--details', action='store_true',
                        help='Return detailed outputs (attention weights, etc.)')
    parser.add_argument('--list_proteins', action='store_true',
                        help='List all available proteins in the database')

    args = parser.parse_args()

    # Initialize predictor
    try:
        predictor = ProteinPairPredictor(args.model_path, args.config_path)
    except Exception as e:
        print(f"Failed to initialize predictor: {e}")
        return

    # List available proteins
    if args.list_proteins:
        proteins = predictor.get_available_proteins()
        print(f"Protein 1 database contains {len(proteins['protein1_database'])} entries:")
        print("First 10 entries:", proteins['protein1_database'][:10])
        print(f"\nProtein 2 database contains {len(proteins['protein2_database'])} entries:")
        print("First 10 entries:", proteins['protein2_database'][:10])
        return

    # Single pair prediction with custom sequences
    if args.protein1 and args.protein2:
        if args.sequence1 and args.sequence2:
            print(f"Predicting protein pair using input sequences: {args.protein1} vs {args.protein2}")
            result = predictor.predict_with_sequences(
                args.protein1, args.sequence1,
                args.protein2, args.sequence2,
                threshold=args.threshold,
                out_dir=args.out_dir,
                return_details=args.details
            )
        else:
            print(f"Predicting protein pair using preloaded database entries: {args.protein1} vs {args.protein2}")
            result = predictor.predict_single_pair(
                args.protein1, args.protein2,
                threshold=args.threshold,
                out_dir=args.out_dir,
                return_details=args.details
            )

        if result['success']:
            print(f"Prediction Results:")
            print(f"  Interaction Probability: {result['interaction_probability']:.6f}")
            print(f"  Predicted Interaction: {'Yes' if result['predicted_interaction'] else 'No'}")
            print(f"  Confidence Score: {result['confidence']}")
            print(f"  Output Directory: {result.get('task_dir', 'N/A')}")
        else:
            print(f"Prediction failed: {result['error']}")

    # Batch prediction from sequence file
    if args.sequences_file:
        print(f"Running batch prediction, input file: {args.sequences_file}")
        # Create timestamp subfolder
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_dir = os.path.join(args.out_dir, f"batch_{timestamp}")
        os.makedirs(batch_dir, exist_ok=True)
        results = []
        with open(args.sequences_file, "r") as f:
            for line in f:
                # Expected format per line: name1,sequence1,name2,sequence2
                parts = line.strip().split(",")
                if len(parts) != 4:
                    print(f"Invalid line format: {line.strip()}")
                    continue
                name1, seq1, name2, seq2 = parts
                result = predictor.predict_with_sequences(
                    name1, seq1, name2, seq2,
                    threshold=args.threshold,
                    out_dir=batch_dir,
                    return_details=args.details
                )
                results.append(result)
        # Aggregate all valid results into CSV
        df = pd.DataFrame([
            {
                "Protein Pair": f"{r['protein1']}_{r['protein2']}",
                "Predicted Interaction Probability": r.get("interaction_probability", None),
                "Predicted Interaction": r.get("predicted_interaction", None),
                "Confidence": r.get("confidence", None)
            }
            for r in results if r.get("success", False)
        ])
        csv_path = os.path.join(batch_dir, "all_results.csv")
        df.to_csv(csv_path, index=False)
        print(f"All batch results saved to: {csv_path}")
        return

    else:
        print("Please specify target protein pair for prediction:")
        print("1. Use pre-existing database entries: --protein1 <name1> --protein2 <name2>")
        print("2. Provide custom amino acid sequences: --protein1 <name1> --sequence1 <seq1> --protein2 <name2> --sequence2 <seq2>")
        print("Run with --help for full usage instructions")


if __name__ == "__main__":
    main()
