import argparse
import logging
from data.load_data import download_data



def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--models", "-m", type=str, help="Name of model e.g. `hf`"
    )

    parser.add_argument(
        "--datasets", "-d", type=str, help="Name of datasets e.g. `hf`"
    )

    parser.add_argument("-v", "--verbose", help="increase output verbosity",
                    action="store_true")

    return parser

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
    
    #download_data()
    



if __name__ == "__main__":
    cli_evaluate()