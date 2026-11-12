"""Run the released Fig. 2f example."""
import argparse
from pathlib import Path
from .predict import ProteinPairPredictor, read_fasta_records

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True, type=Path)
    p.add_argument('--checkpoint', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    predictor = ProteinPairPredictor(args.checkpoint, args.config)
    data = Path(predictor.config['data']['pair_path']).parent / 'examples'
    first, = read_fasta_records(data / 'protein1.fasta')
    second, = read_fasta_records(data / 'protein2.fasta')
    result = predictor.predict_with_sequences(*first, *second, out_dir=str(args.output), return_details=True)
    if not result.get('success'):
        raise RuntimeError(result.get('error', 'Case prediction failed'))
    print(f"Recognition probability: {result['interaction_probability']:.9f}")

if __name__ == '__main__':
    main()
