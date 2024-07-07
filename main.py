import argparse
import logging
from conexion.evaluation.evaluator import evaluate
from conexion.data import get_datasets
from conexion.models import get_models
from typing import List
import os

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
        "--datasets", "-d", nargs='+', help="Name of datasets e.g. `inspec`"
    )

    parser.add_argument(
        "--output", "-o", type=dir_path, help="Folder of the output files", default="output"
    )
    
    parser.add_argument(
        "--template", "-t", help="Name of the template to use e.g. `template_1`", default="template_1"
    )

    parser.add_argument("-v", "--verbose", help="increase output verbosity",
                    action="store_true")

    return parser

def setup_logging(verbose: bool) -> None:
    log_format = "%(asctime)s %(levelname)s %(message)s"

    if verbose:
        logging.basicConfig(level=logging.DEBUG, format=log_format)
    else:
        logging.basicConfig(level=logging.INFO, format=log_format)


def cli_evaluate() -> None:
    with_parser = False

    if with_parser:
        parser = setup_parser()
        args = parser.parse_args()
        setup_logging(args.verbose)
        models = get_models(args.models, args.template)
        datasets = get_datasets(args.datasets)
        evaluate(models, datasets, args.output)
    else:
        # testing / debugging
        setup_logging(False)
        models = get_models(['SpacyEntities'], "template_name")
        datasets = get_datasets(['inspec'])
        evaluate(models, datasets, './output')
    

if __name__ == "__main__":
    cli_evaluate()