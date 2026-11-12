#!/usr/bin/env python
# predict.py
#
import torch
import numpy as np
import os
import argparse
import datetime
import importlib.util
import pandas as pd
from collections import Counter
from pathlib import Path
from .data import get_device
from .utils import get_config, analyze_key_residues, visualize_attention_heatmap, save_key_residues_report


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_INDEX = {amino_acid: index for index, amino_acid in enumerate(AMINO_ACIDS)}


def normalize_sequence(sequence):
    """Return the biological sequence used by every inference feature."""
    normalized = "".join(str(sequence).split()).upper()
    if len(normalized) < 6:
        raise ValueError("Sequence length must be at least 6 to compute CKSAAP-ESMC for gaps 0 through 4")
    return normalized


def read_fasta_records(fasta_path):
    """Read validated FASTA records in file order."""
    records = []
    seen_ids = set()
    current_id = None
    sequence_parts = []

    def append_record():
        if current_id is None:
            return
        if current_id in seen_ids:
            raise ValueError(f"Duplicate identifier in FASTA: {current_id}")
        sequence = normalize_sequence("".join(sequence_parts))
        invalid = sorted(set(sequence).difference(AMINO_ACIDS))
        if invalid:
            raise ValueError(
                f"{current_id}: contains amino-acid characters unsupported by the model: {''.join(invalid)}"
            )
        seen_ids.add(current_id)
        records.append((current_id, sequence))

    with open(fasta_path, "r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                append_record()
                header = line[1:].strip()
                if not header:
                    raise ValueError(f"{fasta_path}:{line_number}: empty FASTA header")
                current_id = header.split()[0]
                sequence_parts = []
            else:
                if current_id is None:
                    raise ValueError(
                        f"{fasta_path}:{line_number}: sequence occurs before the first FASTA header"
                    )
                sequence_parts.append(line)
    append_record()
    if not records:
        raise ValueError(f"No sequences found in FASTA file: {fasta_path}")
    return records


def residue_only_embedding(raw_embedding, sequence, protein_id):
    """Remove ESM-C BOS/EOS and enforce one row per biological residue."""
    embedding = np.asarray(raw_embedding)
    if embedding.ndim != 2:
        raise ValueError(f"{protein_id}: expected a two-dimensional ESM-C embedding; observed {embedding.shape}")
    sequence_length = len(sequence)
    if embedding.shape[0] == sequence_length + 2:
        embedding = embedding[1:-1]
    elif embedding.shape[0] != sequence_length:
        raise ValueError(
            f"{protein_id}: ESM-C row count {embedding.shape[0]} does not match residue count "
            f"{sequence_length} and is not L+2"
        )
    if embedding.shape[1] != 1152:
        raise ValueError(f"{protein_id}: expected ESM-C dimension 1152; observed {embedding.shape[1]}")
    return np.ascontiguousarray(embedding, dtype=np.float32)


def build_cksaap_esmc(sequence, embedding, max_gap=4):
    """Reproduce the training-time residue-aligned CKSAAP-ESMC layout."""
    sequence_length, embedding_dim = embedding.shape
    if sequence_length != len(sequence):
        raise ValueError("Sequence and embedding lengths differ for CKSAAP-ESMC input")
    output = np.zeros((max_gap + 1, embedding_dim, 20, 20), dtype=np.float32)
    for gap in range(max_gap + 1):
        pair_count = sequence_length - gap - 1
        if pair_count <= 0:
            raise ValueError(f"Sequence length {sequence_length} cannot support gap={gap}")
        for left_position in range(pair_count):
            right_position = left_position + gap + 1
            left_index = AA_TO_INDEX.get(sequence[left_position])
            right_index = AA_TO_INDEX.get(sequence[right_position])
            if left_index is None or right_index is None:
                continue
            output[gap, :, left_index, right_index] += 0.5 * (
                embedding[left_position] + embedding[right_position]
            )
        output[gap] /= float(pair_count)
    return output.reshape((max_gap + 1) * embedding_dim, 20, 20)


def resolve_run_file(model_path, relative_path):
    """Resolve a file from the training run that owns a fold checkpoint."""
    checkpoint_path = Path(model_path).expanduser().resolve()
    candidates = [
        checkpoint_path.parent.parent / "source_code" / relative_path,
        checkpoint_path.parent / "source_code" / relative_path,
    ]
    return next((path for path in candidates if path.is_file()), None)


def load_model_class(model_path, model_source=None):
    """Load the exact archived Model definition instead of the legacy local one."""
    source_path = Path(model_source).expanduser().resolve() if model_source else None
    if source_path is None:
        source_path = resolve_run_file(model_path, "model.py")
    if source_path is None:
        from .model import Model
        return Model, Path(__file__).resolve().with_name("model.py")
    spec = importlib.util.spec_from_file_location("deepspy_checkpoint_model", source_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load model source: {source_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "Model"):
        raise AttributeError(f"{source_path} does not define a Model class")
    return module.Model, source_path

class ProteinFeatureGenerator:
    """Generate features for previously unseen protein sequences"""
    
    def __init__(self, config):
        self.config = config
        self.device = get_device()
        self._esmc_client = None
        
    def calculate_AAC(self, sequence):
        """Calculate amino-acid composition (AAC)"""
        counts = Counter(sequence)
        total_count = len(sequence)
        aac = [counts.get(aa, 0) / total_count for aa in AMINO_ACIDS]
        return aac

    def calculate_dipeptide_composition(self, sequence):
        """Calculate dipeptide composition (DPC)"""
        dipeptides = [sequence[i:i+2] for i in range(len(sequence)-1)]
        dipeptide_counts = Counter(dipeptides)
        total_dipeptides = len(dipeptides)
        dipeptide_composition = [dipeptide_counts.get(aa, 0) / total_dipeptides 
                               for aa in (a+b for a in AMINO_ACIDS for b in AMINO_ACIDS)]
        return dipeptide_composition
    
    def generate_basic_features(self, protein_id, sequence):
        """Generate AAC and DPC features"""
        aac = self.calculate_AAC(sequence)
        dipeptide = self.calculate_dipeptide_composition(sequence)
        
        return {
            'aac': torch.tensor(aac).float(),
            'dipeptide': torch.tensor(dipeptide).float()
        }
    
    def generate_esmc_features_standalone(self, protein_id, sequence):
        """Generate exactly the residue-only features used during training."""
        try:
            print(f"Generating features for {protein_id} using ESM-C...")
            from esm.models.esmc import ESMC
            from esm.sdk.api import ESMProtein, LogitsConfig
            protein_sequence = normalize_sequence(sequence)
            if len(protein_sequence) > 1200:
                raise ValueError(
                    "The sequence exceeds the 1,200-residue inference limit. Silent truncation is disabled because AAC, "
                    "CKSAAP-ESMC, attention, and residue numbering must remain aligned"
                )
            os.environ["INFRA_PROVIDER"] = "True"
            if self._esmc_client is None:
                print("Loading ESM-C 600M...")
                self._esmc_client = ESMC.from_pretrained("esmc_600m").to(self.device)
            client = self._esmc_client
            protein = ESMProtein(sequence=protein_sequence)
            with torch.no_grad():
                protein_tensor = client.encode(protein)
                output = client.logits(
                    protein_tensor,
                    LogitsConfig(sequence=True, return_embeddings=True),
                )
            raw_embedding = output.embeddings.squeeze(0).detach().cpu().numpy()
            embedding = residue_only_embedding(
                raw_embedding, protein_sequence, protein_id
            )
            ckesmc = build_cksaap_esmc(protein_sequence, embedding, max_gap=4)
            print(f"{protein_id} feature generation completed")
            return {
                'ckesmc': torch.from_numpy(ckesmc),
                'esmc': torch.from_numpy(embedding),
                'seq_length': len(protein_sequence)
            }
        except Exception as e:
            raise RuntimeError(
                f"Failed to generate ESM-C features for {protein_id}: {e}"
            ) from e
    

class ProteinPairPredictor:
    def __init__(self, model_path, config_path=None, model_source=None):
        """
        Initialize the predictor
        
        Parameters:
            model_path: Path to a trained model checkpoint
            config_path: Path to the configuration file
        """
        if config_path is None:
            archived_config = resolve_run_file(model_path, "resolved_config.yml")
            if archived_config is None:
                archived_config = (
                    Path(__file__).resolve().parents[1]
                    / "configs" / "benchmark_450ep_t050.yml"
                )
            config_path = str(archived_config)
        self.config = get_config(config_path)
        self.device = get_device()
        self.feature_generator = ProteinFeatureGenerator(self.config)

        # Always instantiate the architecture archived with this run.  The
        # The legacy architecture has a different classifier and cannot load
        # the global + gated-interaction checkpoint strictly.
        Model, resolved_model_source = load_model_class(model_path, model_source)
        self.model = Model(
            embed_hid_dim=self.config['model']['embed_hid_dim'],
            dropout=self.config['model']['dropout']
        ).to(self.device)
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            print(f"Loaded model: {model_path}")
            print(f"Model architecture source: {resolved_model_source}")
            print(f"Configuration source: {config_path}")
            print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
            print(f"Checkpoint MCC: {checkpoint.get('mcc', 'unknown')}")
            print(f"interaction gate: {checkpoint.get('interaction_gate', 'unknown')}")
            print(f"Interaction auxiliary BCE coefficient: {checkpoint.get('interaction_aux_loss_coef', 'unknown')}")
        except Exception as e:
            raise Exception(f"Failed to load model: {e}")
        
        # Load existing feature data
        self.load_existing_features()
        
        # Cache newly generated features
        self.new_features_cache = {
            'pro1_aac': {}, 'pro1_dipeptide': {}, 'pro1_ckesmc': {}, 'pro1_esmc': {}, 'pro1_seq_lengths': {},
            'pro2_aac': {}, 'pro2_dipeptide': {}, 'pro2_ckesmc': {}, 'pro2_esmc': {}, 'pro2_seq_lengths': {}
        }

    def resolve_threshold(self, threshold):
        if threshold is None:
            threshold = self.config.get('train', {}).get('threshold', 0.5)
        threshold = float(threshold)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1]; observed {threshold}")
        return threshold
    
    def load_existing_features(self):
        """Load cached features for proteins present in the release dataset."""
        print("Loading existing protein features...")
        
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

            # ESMC features (processed with CKSAAP)
            self.ckesmc_rep1 = np.load(self.config['data']['pro1_ckesmc_path'], allow_pickle=True).item()
            self.ckesmc_rep2 = np.load(self.config['data']['pro2_ckesmc_path'], allow_pickle=True).item()
            
            # Residue-level ESM-C features
            self.esmc_rep1 = np.load(self.config['data']['pro1_esmc_path'], allow_pickle=True).item()
            self.esmc_rep2 = np.load(self.config['data']['pro2_esmc_path'], allow_pickle=True).item()
            
            # Training uses biological sequence length as the sole source of
            # truth; the old *_seq_lengths.npy files are neither required nor
            # generated by the fixed pipeline.
            seq_df1 = pd.read_csv(self.config['data']['pro1_seq_path'], sep='\t')
            seq_df2 = pd.read_csv(self.config['data']['pro2_seq_path'], sep='\t')
            self.sequences1 = {
                str(row['protein_id']): normalize_sequence(row['sequence'])
                for _, row in seq_df1.iterrows()
            }
            self.sequences2 = {
                str(row['protein_id']): normalize_sequence(row['sequence'])
                for _, row in seq_df2.iterrows()
            }
            self.seq_lengths1 = {key: len(value) for key, value in self.sequences1.items()}
            self.seq_lengths2 = {key: len(value) for key, value in self.sequences2.items()}
            self._validate_loaded_features()
            
            print("Existing feature data loaded.")
            print(f"Protein-1 database contains: {len(self.pro1_aac_dict)} proteins")
            print(f"Protein-2 database contains: {len(self.pro2_aac_dict)} proteins")
            
        except Exception as e:
            print(f"Warning while loading existing feature data: {e}")
            print("Feature dictionaries were reset; features will be regenerated from sequences")
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
            self.sequences1 = {}
            self.sequences2 = {}

    def _validate_loaded_features(self):
        """Reject stale BOS/EOS caches before they can shift residue numbers."""
        for side, sequences, esmc, ckesmc in (
            ("protein1", self.sequences1, self.esmc_rep1, self.ckesmc_rep1),
            ("protein2", self.sequences2, self.esmc_rep2, self.ckesmc_rep2),
        ):
            shared_ids = set(esmc).intersection(ckesmc).intersection(sequences)
            for protein_id in shared_ids:
                embedding = np.asarray(esmc[protein_id])
                expected_length = len(sequences[protein_id])
                if embedding.shape != (expected_length, 1152):
                    raise ValueError(
                        f"{side}/{protein_id}: residue-only ESM-C expected "
                        f"({expected_length}, 1152); observed {embedding.shape}"
                    )
                if np.asarray(ckesmc[protein_id]).shape != (5760, 20, 20):
                    raise ValueError(
                        f"{side}/{protein_id}: CKSAAP-ESMC shape expected (5760, 20, 20)"
                    )
    
    def get_or_generate_features(self, protein_name, protein_sequence, is_first_protein=True):
        """
        Retrieve or generate protein features
        
        Parameters:
            protein_name: Protein identifier
            protein_sequence: Protein sequence
            is_first_protein: Whether the sequence occupies the protein-1 role
            
        Returns:
            Feature dictionary
        """
        protein_sequence = normalize_sequence(protein_sequence)
        cache_prefix = 'pro1' if is_first_protein else 'pro2'
        cache_key = (protein_name, protein_sequence)
        
        # Check the existing feature database
        aac_dict = self.pro1_aac_dict if is_first_protein else self.pro2_aac_dict
        dipeptide_dict = self.pro1_dipeptide_dict if is_first_protein else self.pro2_dipeptide_dict
        ckesmc_dict = self.ckesmc_rep1 if is_first_protein else self.ckesmc_rep2
        esmc_dict = self.esmc_rep1 if is_first_protein else self.esmc_rep2
        sequence_dict = self.sequences1 if is_first_protein else self.sequences2
        
        # Check the in-memory cache
        if cache_key in self.new_features_cache[f'{cache_prefix}_aac']:
            print(f"Retrieved cached features for {protein_name}")
            return {
                'aac': self.new_features_cache[f'{cache_prefix}_aac'][cache_key],
                'dipeptide': self.new_features_cache[f'{cache_prefix}_dipeptide'][cache_key],
                'ckesmc': self.new_features_cache[f'{cache_prefix}_ckesmc'][cache_key],
                'esmc': self.new_features_cache[f'{cache_prefix}_esmc'][cache_key],
                'seq_length': self.new_features_cache[f'{cache_prefix}_seq_lengths'][cache_key]
            }
        
        # Check the existing database
        if (protein_name in aac_dict and protein_name in dipeptide_dict and
            protein_name in ckesmc_dict and protein_name in esmc_dict and
            sequence_dict.get(protein_name) == protein_sequence):
            print(f"Retrieved database features for {protein_name}")
            embedding = np.asarray(esmc_dict[protein_name])
            if embedding.shape[0] != len(protein_sequence):
                raise ValueError(
                    f"{protein_name}: cached ESM-C length {embedding.shape[0]} "
                    f"!= Sequence length {len(protein_sequence)}"
                )
            return {
                'aac': torch.tensor(aac_dict[protein_name]).float(),
                'dipeptide': torch.tensor(dipeptide_dict[protein_name]).float(),
                'ckesmc': torch.tensor(ckesmc_dict[protein_name]).float(),
                'esmc': torch.tensor(esmc_dict[protein_name]).float(),
                'seq_length': len(protein_sequence)
            }
        
        # Generate new features
        print(f"Generating features for new protein {protein_name}...")
        
        # Generate AAC and DPC features
        basic_features = self.feature_generator.generate_basic_features(protein_name, protein_sequence)
        
        # Generate ESM-C features
        esmc_features = self.feature_generator.generate_esmc_features_standalone(protein_name, protein_sequence)
        
        # Cache generated features
        self.new_features_cache[f'{cache_prefix}_aac'][cache_key] = basic_features['aac']
        self.new_features_cache[f'{cache_prefix}_dipeptide'][cache_key] = basic_features['dipeptide']
        self.new_features_cache[f'{cache_prefix}_ckesmc'][cache_key] = esmc_features['ckesmc']
        self.new_features_cache[f'{cache_prefix}_esmc'][cache_key] = esmc_features['esmc']
        self.new_features_cache[f'{cache_prefix}_seq_lengths'][cache_key] = esmc_features['seq_length']
        
        return {
            'aac': basic_features['aac'],
            'dipeptide': basic_features['dipeptide'],
            'ckesmc': esmc_features['ckesmc'],
            'esmc': esmc_features['esmc'],
            'seq_length': esmc_features['seq_length']
        }

    def predict_fasta_cartesian(self, protein1_fasta, protein2_fasta,
                                threshold=None, out_dir="./prediction_results"):
        """Predict every protein1 x protein2 combination from two FASTA files."""
        threshold = self.resolve_threshold(threshold)
        records1 = read_fasta_records(protein1_fasta)
        records2 = read_fasta_records(protein2_fasta)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_dir = Path(out_dir) / f"fasta_all_pairs_{timestamp}"
        batch_dir.mkdir(parents=True, exist_ok=False)

        total = len(records1) * len(records2)
        print(f"[INFO] FASTA all-pairs: {len(records1)} x {len(records2)} = {total} pairs")
        print("[INFO] Features are generated once per unique sequence")

        features1 = {
            protein_id: self.get_or_generate_features(
                protein_id, sequence, is_first_protein=True
            )
            for protein_id, sequence in records1
        }
        features2 = {
            protein_id: self.get_or_generate_features(
                protein_id, sequence, is_first_protein=False
            )
            for protein_id, sequence in records2
        }

        rows = []
        pair_index = 0
        with torch.no_grad():
            for protein1_id, sequence1 in records1:
                for protein2_id, sequence2 in records2:
                    pair_index += 1
                    print(f"[INFO] ({pair_index}/{total}) {protein1_id} vs {protein2_id}")
                    first = features1[protein1_id]
                    second = features2[protein2_id]
                    outputs = self.model(
                        first['ckesmc'].unsqueeze(0).to(self.device),
                        second['ckesmc'].unsqueeze(0).to(self.device),
                        first['aac'].unsqueeze(0).to(self.device),
                        second['aac'].unsqueeze(0).to(self.device),
                        first['dipeptide'].unsqueeze(0).to(self.device),
                        second['dipeptide'].unsqueeze(0).to(self.device),
                        first['esmc'].unsqueeze(0).to(self.device),
                        second['esmc'].unsqueeze(0).to(self.device),
                        torch.tensor([first['seq_length']], device=self.device),
                        torch.tensor([second['seq_length']], device=self.device),
                    )
                    probability = float(outputs['logits'].squeeze().cpu())
                    global_raw_logit = float(outputs['global_raw_logits'].squeeze().cpu())
                    interaction_raw_logit = float(
                        outputs['interaction_raw_logits'].squeeze().cpu()
                    )
                    interaction_gate = float(outputs['interaction_gate'].squeeze().cpu())
                    rows.append({
                        "Protein1_ID": protein1_id,
                        "Protein2_ID": protein2_id,
                        "Protein1_Length": len(sequence1),
                        "Protein2_Length": len(sequence2),
                        "Predicted_Interaction_Probability": probability,
                        "Predicted_Label": int(probability >= threshold),
                        "Threshold": threshold,
                        "Global_Only_Probability": float(torch.sigmoid(
                            outputs['global_raw_logits'].squeeze()
                        ).cpu()),
                        "Interaction_Branch_Probability": float(torch.sigmoid(
                            outputs['interaction_raw_logits'].squeeze()
                        ).cpu()),
                        "Global_Raw_Logit": global_raw_logit,
                        "Interaction_Raw_Logit": interaction_raw_logit,
                        "Interaction_Gate": interaction_gate,
                        "Gated_Interaction_Logit_Contribution": (
                            interaction_gate * interaction_raw_logit
                        ),
                    })

        results = pd.DataFrame(rows)
        details_path = batch_dir / "all_pairs_predictions.csv"
        results.to_csv(details_path, index=False)
        row_order = [protein_id for protein_id, _ in records1]
        column_order = [protein_id for protein_id, _ in records2]
        probability_matrix = results.pivot(
            index="Protein1_ID", columns="Protein2_ID",
            values="Predicted_Interaction_Probability"
        ).reindex(index=row_order, columns=column_order)
        label_matrix = results.pivot(
            index="Protein1_ID", columns="Protein2_ID", values="Predicted_Label"
        ).reindex(index=row_order, columns=column_order)
        probability_matrix.to_csv(batch_dir / "probability_matrix.csv")
        label_matrix.to_csv(batch_dir / "label_matrix.csv")

        with open(batch_dir / "RUN_INFO.txt", "w", encoding="utf-8") as handle:
            handle.write(f"protein1_fasta: {Path(protein1_fasta).resolve()}\n")
            handle.write(f"protein2_fasta: {Path(protein2_fasta).resolve()}\n")
            handle.write(f"protein1_count: {len(records1)}\n")
            handle.write(f"protein2_count: {len(records2)}\n")
            handle.write(f"pair_count: {total}\n")
            handle.write(f"threshold: {threshold}\n")
            handle.write(
                "decision_formula: sigmoid(global_raw_logit + "
                "interaction_gate * interaction_raw_logit)\n"
            )
        print(f"[INFO] Detailed results: {details_path}")
        print(f"[INFO] Probability matrix: {batch_dir / 'probability_matrix.csv'}")
        print(f"[INFO] Label matrix: {batch_dir / 'label_matrix.csv'}")
        return batch_dir
    
    def predict_with_sequences(self, protein1_name, protein1_sequence, 
                             protein2_name, protein2_sequence, 
                             threshold=None, out_dir="./prediction_results", 
                             return_details=False):
        """
        Predict from two supplied protein sequences.
        
        Parameters:
            protein1_name: Identifier of the first protein
            protein1_sequence: Sequence of the first protein
            protein2_name: Identifier of the second protein
            protein2_sequence: Sequence of the second protein
            threshold: Decision threshold
            out_dir: Output directory
            return_details: Whether to return detailed outputs
            
        Returns:
            Prediction result dictionary
        """
        try:
            threshold = self.resolve_threshold(threshold)
            protein1_sequence = normalize_sequence(protein1_sequence)
            protein2_sequence = normalize_sequence(protein2_sequence)
            # Create the task directory
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            task_dir = os.path.join(out_dir, f"{protein1_name}_{protein2_name}_{timestamp}")
            os.makedirs(task_dir, exist_ok=True)
            print(f"[INFO] Task output directory: {task_dir}")
            
            print(f"Predicting protein pair: {protein1_name} vs {protein2_name}")
            
            # Retrieve or generate features.
            features1 = self.get_or_generate_features(protein1_name, protein1_sequence, is_first_protein=True)
            features2 = self.get_or_generate_features(protein2_name, protein2_sequence, is_first_protein=False)
            
            # Prepare model inputs
            ckesmc1 = features1['ckesmc'].unsqueeze(0).to(self.device)
            ckesmc2 = features2['ckesmc'].unsqueeze(0).to(self.device)
            aac1 = features1['aac'].unsqueeze(0).to(self.device)
            aac2 = features2['aac'].unsqueeze(0).to(self.device)
            dipeptide1 = features1['dipeptide'].unsqueeze(0).to(self.device)
            dipeptide2 = features2['dipeptide'].unsqueeze(0).to(self.device)
            esmc1 = features1['esmc'].unsqueeze(0).to(self.device)
            esmc2 = features2['esmc'].unsqueeze(0).to(self.device)
            
            # Sequence-length tensors.
            esmc1_lengths = torch.tensor([features1['seq_length']]).to(self.device)
            esmc2_lengths = torch.tensor([features2['seq_length']]).to(self.device)
            
            # Prediction
            print("Running prediction...")
            with torch.no_grad():
                outputs = self.model(
                    ckesmc1, ckesmc2, aac1, aac2, dipeptide1, dipeptide2,
                    esmc1, esmc2, esmc1_lengths, esmc2_lengths
                )

                probability = outputs['logits'].squeeze().cpu().item()
                global_raw_logit = outputs['global_raw_logits'].squeeze().cpu().item()
                interaction_raw_logit = outputs['interaction_raw_logits'].squeeze().cpu().item()
                interaction_gate = outputs['interaction_gate'].squeeze().cpu().item()
                global_probability = torch.sigmoid(
                    outputs['global_raw_logits'].squeeze()
                ).cpu().item()
                interaction_probability = torch.sigmoid(
                    outputs['interaction_raw_logits'].squeeze()
                ).cpu().item()
                gated_interaction_logit = interaction_gate * interaction_raw_logit
                
                # Save prediction results as CSV.
                csv_path = os.path.join(task_dir, "result.csv")
                pair_name = f"{protein1_name}_{protein2_name}"
                df = pd.DataFrame([{
                    "Protein Pair": pair_name,
                    "Predicted Interaction Probability": probability,
                    "Predicted Label": int(probability >= threshold),
                    "Threshold": threshold,
                    "Global-only Probability": global_probability,
                    "Interaction-branch Probability": interaction_probability,
                    "Global Raw Logit": global_raw_logit,
                    "Interaction Raw Logit": interaction_raw_logit,
                    "Interaction Gate": interaction_gate,
                    "Gated Interaction Logit Contribution": gated_interaction_logit,
                }])
                df.to_csv(csv_path, index=False)
                print(f"[INFO] Prediction result saved to: {csv_path}")
                with open(
                    os.path.join(task_dir, "INTERPRETATION_NOTE.txt"),
                    "w", encoding="utf-8"
                ) as note:
                    note.write(
                        "Decision formula:\n"
                        "final_probability = sigmoid(global_raw_logit + "
                        "interaction_gate * interaction_raw_logit)\n\n"
                        "The residue scores and attention maps are exploratory "
                        "outputs, not validated causal residue probabilities.\n"
                        "Protein-1 positions 1-20 are forced to zero by the "
                        "model's signal-peptide assumption.\n"
                    )
                
                result = {
                    'success': True,
                    'protein1': protein1_name,
                    'protein2': protein2_name,
                    'protein1_sequence': protein1_sequence,
                    'protein2_sequence': protein2_sequence,
                    'interaction_probability': probability,
                    'global_only_probability': global_probability,
                    'interaction_branch_probability': interaction_probability,
                    'interaction_gate': interaction_gate,
                    'gated_interaction_logit_contribution': gated_interaction_logit,
                    'predicted_interaction': probability >= threshold,
                    'confidence': 'High' if abs(probability - threshold) > 0.3 else 'Medium' if abs(probability - threshold) > 0.1 else 'Low',
                    'task_dir': task_dir
                }
                
                # Visualization and analysis
                if probability >= threshold:
                    print(
                        "[INFO] The pair was classified as recognized; "
                        "generating exploratory residue outputs..."
                    )
                    
                    score1 = outputs['residue_scores'][0][0].cpu().numpy()
                    score2 = outputs['residue_scores'][1][0].cpu().numpy()
                    attn = outputs['attention_weights']['cross_attn'][0].cpu().numpy()
                    attn_reverse = outputs['attention_weights']['cross_attn_reverse'][0].cpu().numpy()

                    # Analyze prioritized residues
                    key_res1 = analyze_key_residues(
                        protein1_sequence, score1, 
                        os.path.join(task_dir, f"{protein1_name}_key_residues.svg")
                    )
                    key_res2 = analyze_key_residues(
                        protein2_sequence, score2, 
                        os.path.join(task_dir, f"{protein2_name}_key_residues.svg")
                    )

                    # Save the prioritized-residue report
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

                    # Generate the attention heat map
                    visualize_attention_heatmap(
                        sequence1=protein1_sequence,
                        sequence2=protein2_sequence,
                        attention_weights=attn,
                        save_path=os.path.join(task_dir, "attention_heatmap.png"),
                        raw_data_path=os.path.join(task_dir, "attention.csv")
                    )
                    visualize_attention_heatmap(
                        sequence1=protein2_sequence,
                        sequence2=protein1_sequence,
                        attention_weights=attn_reverse,
                        save_path=os.path.join(task_dir, "attention_reverse.png"),
                        raw_data_path=os.path.join(task_dir, "attention_reverse.csv")
                    )
                    
                    print(f"[INFO] Visualization completed; files saved to: {task_dir}")
                else:
                    print(
                        "[INFO] The pair was classified as non-recognized; "
                        "residue plots were not generated."
                    )
                
                if return_details:
                    # Extract attention weights and residue scores
                    result.update({
                        'residue_scores1': score1 if probability >= threshold else None,
                        'residue_scores2': score2 if probability >= threshold else None,
                        'attention_weights': attn if probability >= threshold else None,
                        'attention_weights_reverse': attn_reverse if probability >= threshold else None,
                        'sequence_length1': features1['seq_length'],
                        'sequence_length2': features2['seq_length']
                    })
                
                return result
                
        except Exception as e:
            return {
                'success': False,
                'error': f"Prediction failed: {str(e)}",
                'protein1': protein1_name,
                'protein2': protein2_name
            }
    
    def predict_single_pair(self, protein1_name, protein2_name, 
                           threshold=None, out_dir="./prediction_results", 
                           return_details=False):
        """
        Predict one pair by retrieving both proteins from the feature database.
        """
        threshold = self.resolve_threshold(threshold)
        # Check protein availability
        aac_dict1 = self.pro1_aac_dict
        aac_dict2 = self.pro2_aac_dict
        
        missing_proteins = []
        if protein1_name not in aac_dict1:
            missing_proteins.append(f"protein1: {protein1_name}")
        if protein2_name not in aac_dict2:
            missing_proteins.append(f"protein2: {protein2_name}")
            
        if missing_proteins:
            return {
                'success': False,
                'error': f"Feature data are missing for: {', '.join(missing_proteins)}. Supply the corresponding sequences.",
                'protein1': protein1_name,
                'protein2': protein2_name
            }

        # The fixed data pipeline stores the biological sequences.  Reuse the
        # common inference path so cached and newly supplied proteins receive
        # identical diagnostics and bidirectional attention exports.
        sequence1 = self.sequences1.get(protein1_name)
        sequence2 = self.sequences2.get(protein2_name)
        if sequence1 is not None and sequence2 is not None:
            return self.predict_with_sequences(
                protein1_name, sequence1, protein2_name, sequence2,
                threshold=threshold, out_dir=out_dir,
                return_details=return_details,
            )
        
        try:
            # Create the task directory
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            task_dir = os.path.join(out_dir, f"{protein1_name}_{protein2_name}_{timestamp}")
            os.makedirs(task_dir, exist_ok=True)
            print(f"[INFO] Task output directory: {task_dir}")
            
            # Load existing features.
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
            
            # Add a batch dimension and move tensors to the selected device
            ckesmc1 = ckesmc1.unsqueeze(0).to(self.device)
            ckesmc2 = ckesmc2.unsqueeze(0).to(self.device)
            esmc1 = esmc1.unsqueeze(0).to(self.device)
            esmc2 = esmc2.unsqueeze(0).to(self.device)
            
            esmc1_lengths = torch.tensor([seq_len1]).to(self.device)
            esmc2_lengths = torch.tensor([seq_len2]).to(self.device)
            
            # Prediction
            with torch.no_grad():
                outputs = self.model(
                    ckesmc1, ckesmc2, aac1.unsqueeze(0), aac2.unsqueeze(0),
                    dipeptide1.unsqueeze(0), dipeptide2.unsqueeze(0),
                    esmc1, esmc2, esmc1_lengths, esmc2_lengths
                )
                
                probability = outputs['logits'].squeeze().cpu().item()
                global_raw_logit = outputs['global_raw_logits'].squeeze().cpu().item()
                interaction_raw_logit = outputs['interaction_raw_logits'].squeeze().cpu().item()
                interaction_gate = outputs['interaction_gate'].squeeze().cpu().item()
                global_probability = torch.sigmoid(
                    outputs['global_raw_logits'].squeeze()
                ).cpu().item()
                interaction_probability = torch.sigmoid(
                    outputs['interaction_raw_logits'].squeeze()
                ).cpu().item()
                gated_interaction_logit = interaction_gate * interaction_raw_logit
                
                # Save prediction results as CSV.
                csv_path = os.path.join(task_dir, "result.csv")
                pair_name = f"{protein1_name}_{protein2_name}"
                df = pd.DataFrame([{
                    "Protein Pair": pair_name,
                    "Predicted Interaction Probability": probability,
                    "Predicted Label": int(probability >= threshold),
                    "Threshold": threshold,
                    "Global-only Probability": global_probability,
                    "Interaction-branch Probability": interaction_probability,
                    "Global Raw Logit": global_raw_logit,
                    "Interaction Raw Logit": interaction_raw_logit,
                    "Interaction Gate": interaction_gate,
                    "Gated Interaction Logit Contribution": gated_interaction_logit,
                }])
                df.to_csv(csv_path, index=False)
                print(f"[INFO] Prediction result saved to: {csv_path}")
                
                result = {
                    'success': True,
                    'protein1': protein1_name,
                    'protein2': protein2_name,
                    'interaction_probability': probability,
                    'global_only_probability': global_probability,
                    'interaction_branch_probability': interaction_probability,
                    'interaction_gate': interaction_gate,
                    'gated_interaction_logit_contribution': gated_interaction_logit,
                    'predicted_interaction': probability >= threshold,
                    'confidence': 'High' if abs(probability - threshold) > 0.3 else 'Medium' if abs(probability - threshold) > 0.1 else 'Low',
                    'task_dir': task_dir
                }
                
                # Residue visualization requires sequence information
                if probability >= threshold:
                    print("[INFO] Residue visualization requires supplied sequences.")
                else:
                    print("[INFO] The pair was classified as non-recognized.")
                
                if return_details:
                    result.update({
                        'residue_scores1': None,  # Unavailable because no sequence was supplied
                        'residue_scores2': None,
                        'attention_weights': None,
                        'sequence_length1': seq_len1,
                        'sequence_length2': seq_len2
                    })
                
                return result
                
        except Exception as e:
            return {
                'success': False,
                'error': f"Prediction failed: {str(e)}",
                'protein1': protein1_name,
                'protein2': protein2_name
            }
    
    def get_available_proteins(self):
        """Return identifiers available in the feature database"""
        return {
            'protein1_database': list(self.pro1_aac_dict.keys()),
            'protein2_database': list(self.pro2_aac_dict.keys())
        }

def main():
    parser = argparse.ArgumentParser(
        description='Predict S-RNase-SLF recognition specificity'
    )
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to a trained model checkpoint')
    parser.add_argument('--config_path', type=str, default=None,
                       help='Configuration file; defaults to the paper configuration')
    parser.add_argument('--model_source', type=str, default=None,
                       help='Model source; by default, use model.py associated with the checkpoint')
    parser.add_argument('--protein1', type=str,
                       help='Identifier of the first protein')
    parser.add_argument('--protein2', type=str,
                       help='Identifier of the second protein')
    parser.add_argument('--sequence1', type=str,
                       help='Sequence of the first protein')
    parser.add_argument('--sequence2', type=str,
                       help='Sequence of the second protein')
    parser.add_argument('--threshold', type=float, default=None,
                       help='Decision threshold; defaults to the training configuration')
    parser.add_argument('--out_dir', type=str, default='./prediction_results',
                       help='Output directory (default: ./prediction_results)')
    parser.add_argument('--pairs_file', type=str,
                       help='Path to a file containing protein pairs')
    parser.add_argument('--sequences_file', type=str,
                       help='Comma-separated rows: name1,sequence1,name2,sequence2')
    parser.add_argument('--protein1_fasta', type=str,
                       help='Protein-1 FASTA for Cartesian-product prediction')
    parser.add_argument('--protein2_fasta', type=str,
                       help='Protein-2 FASTA for Cartesian-product prediction')
    parser.add_argument('--details', action='store_true',
                       help='Return detailed outputs, including attention weights')
    parser.add_argument('--list_proteins', action='store_true',
                       help='List proteins in the feature database')
    
    args = parser.parse_args()
    if (args.protein1_fasta is None) != (args.protein2_fasta is None):
        parser.error("--protein1_fasta and --protein2_fasta must be provided together")
    
    # Initialize the predictor
    try:
        predictor = ProteinPairPredictor(
            args.model_path, args.config_path, args.model_source
        )
    except Exception as e:
        raise RuntimeError(f"Failed to initialize predictor: {e}") from e
    
    # List available proteins
    if args.list_proteins:
        proteins = predictor.get_available_proteins()
        print(f"Protein-1 database contains {len(proteins['protein1_database'])} proteins:")
        print("First 10:", proteins['protein1_database'][:10])
        print(f"\nProtein-2 database contains {len(proteins['protein2_database'])} proteins:")
        print("First 10:", proteins['protein2_database'][:10])
        return

    # Predict the Cartesian product of two FASTA files.
    if args.protein1_fasta and args.protein2_fasta:
        try:
            batch_dir = predictor.predict_fasta_cartesian(
                args.protein1_fasta,
                args.protein2_fasta,
                threshold=args.threshold,
                out_dir=args.out_dir,
            )
            print(f"FASTA all-pairs prediction completed: {batch_dir}")
        except Exception as error:
            print(f"FASTA all-pairs prediction failed: {error}")
            raise
        return
    
    # Predict one pair from supplied sequences or the feature database.
    if args.protein1 and args.protein2:
        if (args.sequence1 is None) != (args.sequence2 is None):
            print("Prediction failed: --sequence1 and --sequence2 must be provided together")
            return
        if args.sequence1 and args.sequence2:
            print(f"Predicting pair from supplied sequences: {args.protein1} vs {args.protein2}")
            result = predictor.predict_with_sequences(
                args.protein1, args.sequence1, 
                args.protein2, args.sequence2,
                threshold=args.threshold,
                out_dir=args.out_dir,
                return_details=args.details
            )
        else:
            print(f"Predicting pair from the feature database: {args.protein1} vs {args.protein2}")
            result = predictor.predict_single_pair(
                args.protein1, args.protein2, 
                threshold=args.threshold,
                out_dir=args.out_dir,
                return_details=args.details
            )
        
        if result['success']:
            print("Prediction result:")
            print(f"  Recognition probability: {result['interaction_probability']:.6f}")
            print(f"  Global-only probability: {result['global_only_probability']:.6f}")
            print(f"  Interaction-branch probability: {result['interaction_branch_probability']:.6f}")
            print(f"  interaction gate: {result['interaction_gate']:.6f}")
            print(f"  Predicted recognition: {'yes' if result['predicted_interaction'] else 'no'}")
            print(f"  Confidence: {result['confidence']}")
            print(f"  Output directory: {result.get('task_dir', 'N/A')}")
        else:
            print(f"Prediction failed: {result['error']}")
        return
    
    # Batch prediction from --sequences_file.
    if args.sequences_file:
        print(f"Running batch prediction from: {args.sequences_file}")
        # Create a timestamped output directory
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_dir = os.path.join(args.out_dir, f"batch_{timestamp}")
        os.makedirs(batch_dir, exist_ok=True)
        results = []
        with open(args.sequences_file, "r") as f:
            for line in f:
                # Expected row format: name1,sequence1,name2,sequence2
                parts = line.strip().split(",")
                if len(parts) != 4:
                    print(f"Invalid row: {line.strip()}")
                    continue
                name1, seq1, name2, seq2 = parts
                # Use the batch directory for pair outputs.
                result = predictor.predict_with_sequences(
                    name1, seq1, name2, seq2,
                    threshold=args.threshold,
                    out_dir=batch_dir,
                    return_details=args.details
                )
                results.append(result)
        # Combine results into one CSV file
        df = pd.DataFrame([
            {
                "Protein Pair": f"{r['protein1']}_{r['protein2']}",
                "Predicted Interaction Probability": r.get("interaction_probability", None),
                "Predicted Interaction": r.get("predicted_interaction", None),
                "Global-only Probability": r.get("global_only_probability", None),
                "Interaction-branch Probability": r.get("interaction_branch_probability", None),
                "Interaction Gate": r.get("interaction_gate", None),
                "Gated Interaction Logit Contribution": r.get("gated_interaction_logit_contribution", None),
                "Confidence": r.get("confidence", None)
            }
            for r in results if r.get("success", False)
        ])
        df.to_csv(os.path.join(batch_dir, "all_results.csv"), index=False)
        print(f"Combined results saved to: {os.path.join(batch_dir, 'all_results.csv')}")
        return

    else:
        print("Specify protein pairs to predict:")
        print("1. Use the existing feature database: --protein1 <ID1> --protein2 <ID2>")
        print("2. Provide new sequences: --protein1 <ID1> --sequence1 <SEQUENCE1> --protein2 <ID2> --sequence2 <SEQUENCE2>") 
        print("3. Cartesian product of two FASTA files: --protein1_fasta <FASTA1> --protein2_fasta <FASTA2>")
        print("Run with --help for complete usage information")

if __name__ == "__main__":
    main()
