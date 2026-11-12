"""Installed command entry points."""

def train_main():
    from .train import cli
    cli()

def predict_main():
    from .predict import main
    main()

def prepare_features_main():
    from .prepare_features import main
    main()
