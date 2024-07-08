import argparse
import logging
from conexion.evaluation.evaluator import evaluate, evaluate_transfer_learning
from conexion.data import get_datasets
from conexion.models import get_models
from typing import List
import os
import sys
import random

def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--models", "-m", nargs='+', help="<Required> Names of models e.g. `SpacyEntities`", required=True
    )

    parser.add_argument(
        "--traindatasets", "-t", nargs='+', help="Name of datasets for training e.g. `inspec`. If provided, the length needs to be the same as the testing data e.g. --datasets. This option can be used to test transfer learning."
    )

    parser.add_argument(
        "--datasets", "-d", nargs='+', help="Name of datasets for testing e.g. `inspec`", required=True
    )

    parser.add_argument(
        "--output", "-o", type=dir_path, help="Folder of the output files", default="output"
    )
    
    parser.add_argument(
        "--gpu", "-g", help="The GPUs to use (will be passed to CUDA_VISIBLE_DEVICES) e.g. `0,1` or '0'"
    )

    parser.add_argument("-v", "--verbose", help="increase output verbosity",
                    action="store_true")

    return parser

def parse_eval_args(parser: argparse.ArgumentParser, cmd_arguments: List[str]) -> argparse.Namespace:
    args = parser.parse_args(args=cmd_arguments)

    log_format = "%(asctime)s %(levelname)s %(message)s"

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format=log_format)
    else:
        logging.basicConfig(level=logging.INFO, format=log_format)
    
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # setting random seeds
    seed = 42
    random.seed(seed)
    import numpy as np 
    np.random.seed(seed) # importing here because of cuda visible devices set before
    import torch
    torch.manual_seed(seed)

    return args


def cli_evaluate() -> None:
    cmd_arguments = sys.argv[1:]
    #cmd_arguments = [
    #    "-m", "class=LLMBaseModel,model_name=meta-llama/Llama-2-7b-chat-hf,prompt=simple_keywords,with_confidence=False,batched_generation=True", 
    #    "-d", "inspec", 
    #    "-o", "./output",
    #    "-g", "1"
    #]

    parser = setup_parser()
    args = parse_eval_args(parser, cmd_arguments)
    models = get_models(args.models)
    test_datasets = get_datasets(args.datasets)

    if not args.traindatasets:
        evaluate(models, test_datasets, args.output)
    else:
        train_datasets = get_datasets(args.traindatasets)
        assert len(train_datasets) == len(test_datasets), "The length of the training datasets needs to be the same as the testing datasets."
        train_and_test_datasets = list(zip(train_datasets, test_datasets))
        evaluate_transfer_learning(models, train_and_test_datasets, args.output)

if __name__ == "__main__":
    cli_evaluate()