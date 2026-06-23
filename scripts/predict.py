#!/usr/bin/env python
# predict.py
# python predict.py \
#     --model_path /home/wuhe/model/SIR/results/20251106_160847_baseline/fold_4/fold_4_best_model.pkl \
#     --config_path ./config.yml \
#     --protein1 PhS5-rnase \
#     --sequence1 "MFRSQLTSVIFVLFFSLPPIYGYFEYMQLVLTWPPAFCHIKRCRRTPNNFTIHGLWPDNYSTMLNFCTDDEFVKFTDDDKKDKLDKRWPDLITDEADCKGTQDFWKREYEKHGTCCLSSYNQEQYFELAMVLKDRFDLVKSFRNHGIIPGTAGHTVQKINNTVKAITQGFPNLACTKALELKEIGICFDRTGKNVINCPHPRTCKQTRTGIKFP" \
#     --protein2 PaxSm-SLF10 \
#     --sequence2 "MKELPQDVVIYIFVMLPVKSLLRFKCTYKTFCHIIKSSTFINLHLNHTTNFNDELVLLKRSFETDEYNFYKSILSFLFAKEDYDFKPISPDVEIPHLTTTAACICHRLIGPCNGLIVLTDSLTTIVFNPATLKYRLIPPCPFGIPRGFRRSISGIGFGFDSDANDYKVVRLSEVYKEPCDKEMKVDIYDFSVDSWRELLGQDVPFVFWFPCAEILYKRNFHWFAFADVVVILCFEMNTEKFHNMGMPDACHFDDGKSYGLVILFKCMTLICYPDPMPSSPTEKLTDIWIMKEYGEKESWIKRCSIRLLPESPLAVWKDEILLLHSKMGHLIAYDLNSNEVQELDLHGYPESLRIIIYRESLTAIPRNNDCIELQNFRCN" \
#     --threshold 0.5 \
#     --out_dir ./predict_output
# python predict.py \
#     --model_path /home/wuhe/model/SIR/results/20250929_102029A/fold_3/fold_3_best_model.pkl \
#     --config_path ./config.yml \
#     --pairs_file pairs.txt \
#     --threshold 0.5 \
#     --out_dir ./predict_output
# python predict.py \
#     --model_path /home/wuhe/model/SIR/results/20251106_160847_baseline/fold_4/fold_4_best_model.pkl \
#     --config_path ./config.yml \
#     --sequences_file sequences.txt \
#     --threshold 0.5 \
#     --out_dir ./predict_output
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
    """用于生成新蛋白质特征的类"""
    
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
        """计算二肽组成（Dipeptide Composition）"""
        amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        dipeptides = [sequence[i:i+2] for i in range(len(sequence)-1)]
        dipeptide_counts = Counter(dipeptides)
        total_dipeptides = len(dipeptides)
        dipeptide_composition = [dipeptide_counts.get(aa, 0) / total_dipeptides 
                               for aa in (a+b for a in amino_acids for b in amino_acids)]
        return dipeptide_composition
    
    def generate_basic_features(self, protein_id, sequence):
        """生成基本特征（AAC和二肽）"""
        aac = self.calculate_AAC(sequence)
        dipeptide = self.calculate_dipeptide_composition(sequence)
        
        return {
            'aac': torch.tensor(aac).float(),
            'dipeptide': torch.tensor(dipeptide).float()
        }
    
    def generate_esmc_features_standalone(self, protein_id, sequence):
        """独立生成ESMC特征，不依赖外部脚本"""
        try:
            # 动态导入ESMC相关模块
            print(f"正在为 {protein_id} 生成ESMC特征...")
            
            # 添加必要的路径
            import sys
            esmc_path = "/mnt/data2/model"
            if esmc_path not in sys.path:
                sys.path.append(esmc_path)
            
            # 导入必要的模块
            from esm.models.esmc import ESMC
            from esm.sdk.api import ESMProtein, LogitsConfig
            import torch.nn as nn
            import math
            from itertools import product
            
            # 定义位置编码类
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

            # 定义CKSAAP类
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

            # 定义ESMC模型类
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
            
            # 设置序列长度限制
            max_seq_length = 1200
            protein_sequence = sequence[:max_seq_length]  # 截断序列
            
            # 初始化模型
            esmc_model = ESMCFineTune(self.device)
            cksaap = CKSAAP(self.device)
            
            # 生成特征
            seq_esm_input = [(protein_id, protein_sequence)]
            print(f"正在生成 {protein_id} 的ESMC嵌入...")
            out = esmc_model(seq_esm_input)
            textembed1 = out.to(self.device)
            
            print(f"正在生成 {protein_id} 的CKSAAP特征...")
            textembed_1 = cksaap.return_CKSAAP_Emb_code(protein_sequence, textembed1, k=4, is_shape_for_3d=True)
            
            print(f"{protein_id} 的特征生成完成")
            
            return {
                'ckesmc': textembed_1.cpu(),
                'esmc': textembed1.cpu(),
                'seq_length': len(protein_sequence)
            }
            
        except Exception as e:
            print(f"独立生成ESMC特征失败: {e}")
            # 如果独立生成失败，尝试使用简化的替代方案
            return None
    

class ProteinPairPredictor:
    def __init__(self, model_path, config_path):
        """
        初始化预测器
        
        参数:
            model_path: 训练好的模型检查点路径
            config_path: 配置文件路径
        """
        self.config = get_config(config_path)
        self.device = get_device()
        self.feature_generator = ProteinFeatureGenerator(self.config)
        
        # 加载模型
        self.model = Model(
            embed_hid_dim=self.config['model']['embed_hid_dim'],
            dropout=self.config['model']['dropout']
        ).to(self.device)
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            print(f"成功加载模型: {model_path}")
            print(f"模型训练轮次: {checkpoint.get('epoch', '未知')}")
            print(f"模型最佳MCC: {checkpoint.get('mcc', '未知')}")
        except Exception as e:
            raise Exception(f"加载模型失败: {e}")
        
        # 加载现有特征数据
        self.load_existing_features()
        
        # 用于缓存新生成的特征
        self.new_features_cache = {
            'pro1_aac': {}, 'pro1_dipeptide': {}, 'pro1_ckesmc': {}, 'pro1_esmc': {}, 'pro1_seq_lengths': {},
            'pro2_aac': {}, 'pro2_dipeptide': {}, 'pro2_ckesmc': {}, 'pro2_esmc': {}, 'pro2_seq_lengths': {}
        }
    
    def load_existing_features(self):
        """加载现有的特征数据"""
        print("正在加载现有蛋白质特征数据...")
        
        try:
            # AAC features
            self.aac_rep1 = np.load(self.config['data']['pro1_aac_path'], allow_pickle=True)
            self.pro1_aac_dict = dict(zip(self.aac_rep1.files, [self.aac_rep1[f] for f in self.aac_rep1.files]))
            self.aac_rep2 = np.load(self.config['data']['pro2_aac_path'], allow_pickle=True)
            self.pro2_aac_dict = dict(zip(self.aac_rep2.files, [self.aac_rep2[f] for f in self.aac_rep2.files]))

            # Dipeptide features  
            self.dipeptide_rep1 = np.load(self.config['data']['pro1_dipeptide_path'], allow_pickle=True)
            self.pro1_dipeptide_dict = dict(zip(self.dipeptide_rep1.files, [self.dipeptide_rep1[f] for f in self.dipeptide_rep1.files]))
            self.dipeptide_rep2 = np.load(self.config['data']['pro2_dipeptide_path'], allow_pickle=True)
            self.pro2_dipeptide_dict = dict(zip(self.dipeptide_rep2.files, [self.dipeptide_rep2[f] for f in self.dipeptide_rep2.files]))

            # ESMC features (CKSAAP处理后的)
            self.ckesmc_rep1 = np.load(self.config['data']['pro1_ckesmc_path'], allow_pickle=True).item()
            self.ckesmc_rep2 = np.load(self.config['data']['pro2_ckesmc_path'], allow_pickle=True).item()
            
            # 原始ESMC特征
            self.esmc_rep1 = np.load(self.config['data']['pro1_esmc_path'], allow_pickle=True).item()
            self.esmc_rep2 = np.load(self.config['data']['pro2_esmc_path'], allow_pickle=True).item()
            
            # 序列长度信息
            self.seq_lengths1 = np.load(self.config['data']['pro1_esmc_path'].replace('.npy', '_seq_lengths.npy'), 
                                      allow_pickle=True).item()
            self.seq_lengths2 = np.load(self.config['data']['pro2_esmc_path'].replace('.npy', '_seq_lengths.npy'), 
                                      allow_pickle=True).item()
            
            print("现有特征数据加载完成!")
            print(f"蛋白质1数据库包含: {len(self.pro1_aac_dict)} 个蛋白质")
            print(f"蛋白质2数据库包含: {len(self.pro2_aac_dict)} 个蛋白质")
            
        except Exception as e:
            print(f"加载现有特征数据时出现警告: {e}")
            print("将使用空的特征字典，所有蛋白质都将重新生成特征")
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
        """
        获取或生成蛋白质特征
        
        参数:
            protein_name: 蛋白质名称
            protein_sequence: 蛋白质序列
            is_first_protein: 是否是第一个蛋白质
            
        返回:
            特征字典
        """
        cache_prefix = 'pro1' if is_first_protein else 'pro2'
        
        # 检查是否在现有数据库中
        aac_dict = self.pro1_aac_dict if is_first_protein else self.pro2_aac_dict
        dipeptide_dict = self.pro1_dipeptide_dict if is_first_protein else self.pro2_dipeptide_dict
        ckesmc_dict = self.ckesmc_rep1 if is_first_protein else self.ckesmc_rep2
        esmc_dict = self.esmc_rep1 if is_first_protein else self.esmc_rep2
        seq_lengths = self.seq_lengths1 if is_first_protein else self.seq_lengths2
        
        # 检查缓存
        if protein_name in self.new_features_cache[f'{cache_prefix}_aac']:
            print(f"从缓存中获取 {protein_name} 的特征")
            return {
                'aac': self.new_features_cache[f'{cache_prefix}_aac'][protein_name],
                'dipeptide': self.new_features_cache[f'{cache_prefix}_dipeptide'][protein_name],
                'ckesmc': self.new_features_cache[f'{cache_prefix}_ckesmc'][protein_name],
                'esmc': self.new_features_cache[f'{cache_prefix}_esmc'][protein_name],
                'seq_length': self.new_features_cache[f'{cache_prefix}_seq_lengths'][protein_name]
            }
        
        # 检查现有数据库
        if (protein_name in aac_dict and protein_name in dipeptide_dict and 
            protein_name in ckesmc_dict and protein_name in esmc_dict):
            print(f"从现有数据库获取 {protein_name} 的特征")
            return {
                'aac': torch.tensor(aac_dict[protein_name]).float(),
                'dipeptide': torch.tensor(dipeptide_dict[protein_name]).float(),
                'ckesmc': torch.tensor(ckesmc_dict[protein_name]).float(),
                'esmc': torch.tensor(esmc_dict[protein_name]).float(),
                'seq_length': seq_lengths.get(protein_name, len(protein_sequence))
            }
        
        # 生成新特征
        print(f"正在为新蛋白质 {protein_name} 生成特征...")
        
        # 生成基本特征（AAC和二肽）
        basic_features = self.feature_generator.generate_basic_features(protein_name, protein_sequence)
        
        # 生成ESMC特征
        esmc_features = self.feature_generator.generate_esmc_features_standalone(protein_name, protein_sequence)
        
        # 缓存生成的特征
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
        """
        使用蛋白质序列进行预测
        
        参数:
            protein1_name: 第一个蛋白质名称
            protein1_sequence: 第一个蛋白质序列
            protein2_name: 第二个蛋白质名称  
            protein2_sequence: 第二个蛋白质序列
            threshold: 预测阈值
            out_dir: 输出目录
            return_details: 是否返回详细信息
            
        返回:
            预测结果字典
        """
        try:
            # 创建任务目录
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            task_dir = os.path.join(out_dir, f"{protein1_name}_{protein2_name}_{timestamp}")
            os.makedirs(task_dir, exist_ok=True)
            print(f"[INFO] 当前任务输出目录：{task_dir}")
            
            print(f"开始预测蛋白质对: {protein1_name} vs {protein2_name}")
            
            # 获取或生成特征
            features1 = self.get_or_generate_features(protein1_name, protein1_sequence, is_first_protein=True)
            features2 = self.get_or_generate_features(protein2_name, protein2_sequence, is_first_protein=False)
            
            # 准备模型输入
            ckesmc1 = features1['ckesmc'].unsqueeze(0).to(self.device)
            ckesmc2 = features2['ckesmc'].unsqueeze(0).to(self.device)
            aac1 = features1['aac'].unsqueeze(0).to(self.device)
            aac2 = features2['aac'].unsqueeze(0).to(self.device)
            dipeptide1 = features1['dipeptide'].unsqueeze(0).to(self.device)
            dipeptide2 = features2['dipeptide'].unsqueeze(0).to(self.device)
            esmc1 = features1['esmc'].unsqueeze(0).to(self.device)
            esmc2 = features2['esmc'].unsqueeze(0).to(self.device)
            
            # 序列长度张量
            esmc1_lengths = torch.tensor([features1['seq_length']]).to(self.device)
            esmc2_lengths = torch.tensor([features2['seq_length']]).to(self.device)
            
            # 预测
            print("正在进行预测...")
            with torch.no_grad():
                outputs = self.model(
                    ckesmc1, ckesmc2, aac1, aac2, dipeptide1, dipeptide2,
                    esmc1, esmc2, esmc1_lengths, esmc2_lengths
                )
                
                probability = outputs['logits'].squeeze().cpu().item()
                
                # 保存预测结果为CSV
                csv_path = os.path.join(task_dir, "result.csv")
                pair_name = f"{protein1_name}_{protein2_name}"
                df = pd.DataFrame([[pair_name, probability]], columns=["Protein Pair", "Predicted Interaction Probability"])
                df.to_csv(csv_path, index=False)
                print(f"[INFO] 预测结果已保存到: {csv_path}")
                
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
                
                # 可视化与分析
                if probability >= threshold:
                    print("[INFO] 预测为相互作用，正在生成可视化分析...")
                    
                    score1 = outputs['residue_scores'][0][0].cpu().numpy()
                    score2 = outputs['residue_scores'][1][0].cpu().numpy()
                    attn = outputs['attention_weights']['cross_attn'][0].cpu().numpy()

                    # 分析关键残基
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

                    # 保存关键残基报告
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

                    # 生成注意力热图
                    visualize_attention_heatmap(
                        sequence1=protein1_sequence,
                        sequence2=protein2_sequence,
                        attention_weights=attn,
                        save_path=os.path.join(task_dir, "attention_heatmap.png"),
                        raw_data_path=os.path.join(task_dir, "attention.csv")
                    )
                    
                    print(f"[INFO] 可视化分析完成，所有文件已保存到: {task_dir}")
                else:
                    print("[INFO] 预测为非互作，无需生成残基分析图")
                
                if return_details:
                    # 提取注意力权重和残基得分
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
                'error': f"预测过程中出现错误: {str(e)}",
                'protein1': protein1_name,
                'protein2': protein2_name
            }
    
    def predict_single_pair(self, protein1_name, protein2_name, 
                           threshold=0.5, out_dir="./prediction_results", 
                           return_details=False):
        """
        预测单个蛋白质对的相互作用概率（仅使用名称，从现有数据库查找）
        """
        # 检查蛋白质是否存在
        aac_dict1 = self.pro1_aac_dict
        aac_dict2 = self.pro2_aac_dict
        
        missing_proteins = []
        if protein1_name not in aac_dict1:
            missing_proteins.append(f"蛋白质1: {protein1_name}")
        if protein2_name not in aac_dict2:
            missing_proteins.append(f"蛋白质2: {protein2_name}")
            
        if missing_proteins:
            return {
                'success': False,
                'error': f"缺少以下蛋白质的特征数据: {', '.join(missing_proteins)}. 请使用 --sequence 参数提供序列。",
                'protein1': protein1_name,
                'protein2': protein2_name
            }
        
        try:
            # 创建任务目录
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            task_dir = os.path.join(out_dir, f"{protein1_name}_{protein2_name}_{timestamp}")
            os.makedirs(task_dir, exist_ok=True)
            print(f"[INFO] 当前任务输出目录：{task_dir}")
            
            # 加载现有特征
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
            
            # 添加批次维度并移至设备
            ckesmc1 = ckesmc1.unsqueeze(0).to(self.device)
            ckesmc2 = ckesmc2.unsqueeze(0).to(self.device)
            esmc1 = esmc1.unsqueeze(0).to(self.device)
            esmc2 = esmc2.unsqueeze(0).to(self.device)
            
            esmc1_lengths = torch.tensor([seq_len1]).to(self.device)
            esmc2_lengths = torch.tensor([seq_len2]).to(self.device)
            
            # 预测
            with torch.no_grad():
                outputs = self.model(
                    ckesmc1, ckesmc2, aac1.unsqueeze(0), aac2.unsqueeze(0),
                    dipeptide1.unsqueeze(0), dipeptide2.unsqueeze(0),
                    esmc1, esmc2, esmc1_lengths, esmc2_lengths
                )
                
                probability = outputs['logits'].squeeze().cpu().item()
                
                # 保存预测结果为CSV
                csv_path = os.path.join(task_dir, "result.csv")
                pair_name = f"{protein1_name}_{protein2_name}"
                df = pd.DataFrame([[pair_name, probability]], columns=["Protein Pair", "Predicted Interaction Probability"])
                df.to_csv(csv_path, index=False)
                print(f"[INFO] 预测结果已保存到: {csv_path}")
                
                result = {
                    'success': True,
                    'protein1': protein1_name,
                    'protein2': protein2_name,
                    'interaction_probability': probability,
                    'predicted_interaction': probability >= threshold,
                    'confidence': 'High' if abs(probability - 0.5) > 0.3 else 'Medium' if abs(probability - 0.5) > 0.1 else 'Low',
                    'task_dir': task_dir
                }
                
                # 目前无法进行可视化分析，因为没有序列信息
                if probability >= threshold:
                    print("[INFO] 预测为相互作用，但由于没有序列信息，无法生成可视化分析")
                    print("[INFO] 如需生成可视化分析，请提供蛋白质序列")
                else:
                    print("[INFO] 预测为非互作")
                
                if return_details:
                    result.update({
                        'residue_scores1': None,  # 无法获取，因为没有序列
                        'residue_scores2': None,
                        'attention_weights': None,
                        'sequence_length1': seq_len1,
                        'sequence_length2': seq_len2
                    })
                
                return result
                
        except Exception as e:
            return {
                'success': False,
                'error': f"预测过程中出现错误: {str(e)}",
                'protein1': protein1_name,
                'protein2': protein2_name
            }
    
    def get_available_proteins(self):
        """获取可用的蛋白质列表"""
        return {
            'protein1_database': list(self.pro1_aac_dict.keys()),
            'protein2_database': list(self.pro2_aac_dict.keys())
        }

def main():
    parser = argparse.ArgumentParser(description='预测蛋白质-蛋白质相互作用')
    parser.add_argument('--model_path', type=str, required=True,
                       help='训练好的模型检查点路径')
    parser.add_argument('--config_path', type=str, default='./config.yml',
                       help='配置文件路径')
    parser.add_argument('--protein1', type=str,
                       help='第一个蛋白质名称')
    parser.add_argument('--protein2', type=str,
                       help='第二个蛋白质名称')
    parser.add_argument('--sequence1', type=str,
                       help='第一个蛋白质序列')
    parser.add_argument('--sequence2', type=str,
                       help='第二个蛋白质序列')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='预测阈值 (默认: 0.5)')
    parser.add_argument('--out_dir', type=str, default='./prediction_results',
                       help='输出目录 (默认: ./prediction_results)')
    parser.add_argument('--pairs_file', type=str,
                       help='包含蛋白质对的文件路径')
    parser.add_argument('--sequences_file', type=str,
                       help='包含蛋白质序列的文件路径(格式: name1,sequence1,name2,sequence2)')
    parser.add_argument('--details', action='store_true',
                       help='返回详细信息(注意力权重等)')
    parser.add_argument('--list_proteins', action='store_true',
                       help='列出所有可用的蛋白质')
    
    args = parser.parse_args()
    
    # 初始化预测器
    try:
        predictor = ProteinPairPredictor(args.model_path, args.config_path)
    except Exception as e:
        print(f"初始化预测器失败: {e}")
        return
    
    # 列出可用蛋白质
    if args.list_proteins:
        proteins = predictor.get_available_proteins()
        print(f"蛋白质1数据库包含 {len(proteins['protein1_database'])} 个蛋白质:")
        print("前10个:", proteins['protein1_database'][:10])
        print(f"\n蛋白质2数据库包含 {len(proteins['protein2_database'])} 个蛋白质:")
        print("前10个:", proteins['protein2_database'][:10])
        return
    
    # 使用序列进行单个预测
    if args.protein1 and args.protein2:
        if args.sequence1 and args.sequence2:
            print(f"使用序列预测蛋白质对: {args.protein1} vs {args.protein2}")
            result = predictor.predict_with_sequences(
                args.protein1, args.sequence1, 
                args.protein2, args.sequence2,
                threshold=args.threshold,
                out_dir=args.out_dir,
                return_details=args.details
            )
        else:
            print(f"使用现有数据库预测蛋白质对: {args.protein1} vs {args.protein2}")
            result = predictor.predict_single_pair(
                args.protein1, args.protein2, 
                threshold=args.threshold,
                out_dir=args.out_dir,
                return_details=args.details
            )
        
        if result['success']:
            print(f"预测结果:")
            print(f"  相互作用概率: {result['interaction_probability']:.6f}")
            print(f"  预测相互作用: {'是' if result['predicted_interaction'] else '否'}")
            print(f"  置信度: {result['confidence']}")
            print(f"  输出目录: {result.get('task_dir', 'N/A')}")
        else:
            print(f"预测失败: {result['error']}")
    
    # 批量预测（处理 sequences_file）
    if args.sequences_file:
        print(f"正在批量预测，输入文件: {args.sequences_file}")
        # 创建时间戳子文件夹
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_dir = os.path.join(args.out_dir, f"batch_{timestamp}")
        os.makedirs(batch_dir, exist_ok=True)
        results = []
        with open(args.sequences_file, "r") as f:
            for line in f:
                # 假设每行格式: name1,sequence1,name2,sequence2
                parts = line.strip().split(",")
                if len(parts) != 4:
                    print(f"格式错误: {line.strip()}")
                    continue
                name1, seq1, name2, seq2 = parts
                # 修改输出目录为子文件夹
                result = predictor.predict_with_sequences(
                    name1, seq1, name2, seq2,
                    threshold=args.threshold,
                    out_dir=batch_dir,
                    return_details=args.details
                )
                results.append(result)
        # 汇总所有结果到一个csv
        df = pd.DataFrame([
            {
                "Protein Pair": f"{r['protein1']}_{r['protein2']}",
                "Predicted Interaction Probability": r.get("interaction_probability", None),
                "Predicted Interaction": r.get("predicted_interaction", None),
                "Confidence": r.get("confidence", None)
            }
            for r in results if r.get("success", False)
        ])
        df.to_csv(os.path.join(batch_dir, "all_results.csv"), index=False)
        print(f"所有结果已汇总到: {os.path.join(batch_dir, 'all_results.csv')}")
        return

    else:
        print("请指定要预测的蛋白质对:")
        print("1. 使用现有数据库: --protein1 <名称1> --protein2 <名称2>")
        print("2. 提供新序列: --protein1 <名称1> --sequence1 <序列1> --protein2 <名称2> --sequence2 <序列2>") 
        print("使用 --help 查看详细用法")

if __name__ == "__main__":
    main()
