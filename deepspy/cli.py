"""Console entry points for DeepSPY."""

import argparse


def train_main() -> None:
    from .train import main

    parser = argparse.ArgumentParser(description="Train and evaluate DeepSPY")
    parser.add_argument("--config", default="configs/full_model.yml")
    args = parser.parse_args()
    result_dir = main(args.config)
    print(f"Training completed. Results were saved to {result_dir}")


def predict_main() -> None:
    from .predict import main

    main()


def prepare_features_main() -> None:
    from .prepare_features import main

    main()
