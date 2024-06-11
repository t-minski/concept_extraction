import argparse
import logging
from data.load_data import download_data
from evaluation.evaluator import evaluate
import importlib
from typing import List

def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--models", "-m", nargs='+', help="<Required> Names of models e.g. `SpacyEntities`", required=True
    )

    parser.add_argument(
        "--datasets", "-d", nargs='+', help="Name of datasets e.g. `inspec`"
    )

    parser.add_argument("-v", "--verbose", help="increase output verbosity",
                    action="store_true")

    return parser




models_map = {
    "SpacyEntities": ("models.spacy_models", "SpacyEntities"),
    "SpacyNounChunks": ("models.spacy_models", "SpacyNounChunks")
}

def get_models(model_texts: List[str]) -> List:
    models = []
    for model_text in model_texts:
        if model_text not in models_map:
            raise ValueError(f"Model {model_text} not found")
        module_name, class_name = models_map[model_text]
        my_class = getattr(importlib.import_module(module_name), class_name)
        models.append(my_class())
    return models

dataset_map = {
    "inspec": ("data.inspec_dataset", "Inspec"),
}

def get_datasets(dataset_texts: List[str]) -> List:
    datasets = []
    for dataset_text in dataset_texts:
        if dataset_text not in dataset_map:
            raise ValueError(f"Dataset {dataset_text} not found")
        module_name, class_name = dataset_map[dataset_text]
        my_class = getattr(importlib.import_module(module_name), class_name)
        datasets.append(my_class())
    return datasets

def parse_eval_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
    args = parser.parse_args()

    log_format = "%(asctime)s %(levelname)s %(message)s"

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format=log_format)
    else:
        logging.basicConfig(level=logging.INFO, format=log_format)
    
    return args

def cli_evaluate() -> None:
    parser = setup_parser()
    args = parse_eval_args(parser)

    models = get_models(args.models)
    datasets = get_datasets(args.datasets)

    evaluate(models, datasets)

if __name__ == "__main__":
    cli_evaluate()