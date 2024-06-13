import argparse
import logging
from evaluation.evaluator import evaluate
import importlib
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

    parser.add_argument("-v", "--verbose", help="increase output verbosity",
                    action="store_true")

    return parser




models_map = {
    "SpacyEntities": ("models.spacy_models", "SpacyEntities"),
    "SpacyNounChunks": ("models.spacy_models", "SpacyNounChunks"),
    "TfIdfEntities": ("models.tfidf_models", "TfIdfEntities"),
    "YakeEntities": ("models.yake_models", "YakeEntities"),
    "SummaEntities": ("models.summa_models", "SummaEntities"),
    "RakeEntities": ("models.rake_models", "RakeEntities"),
    "PyateBasicsEntities": ("models.pyate_models", "PyateBasicsEntities"),
    "PyateComboBasicEntities": ("models.pyate_models", "PyateComboBasicEntities"),
    "PyateCvaluesEntities": ("models.pyate_models", "PyateCvaluesEntities"),
    "LSAEntities": ("models.lsa_models", "LSAEntities"),
    "LDAEntities": ("models.lda_models", "LDAEntities"),
    "TopicRank": ("models.TopicRank_models", "TopicRank"),
    "TopicalPageRank": ("models.TopicalPageRank_models", "TopicalPageRank"),
    "PositionRank": ("models.PositionRank_models", "PositionRank"),
    "MultipartiteRank": ("models.MultipartiteRank_models", "MultipartiteRank"),
    "KeyBERTEntities": ("models.llm_models", "KeyBERTEntities"),
    "Llama2_7b_ZeroShotEntities": ("models.llm_models", "Llama2_7b_ZeroShotEntities"),
    "Llama2_7b_OneShotEntities": ("models.llm_models", "Llama2_7b_OneShotEntities"),
    "Llama3_8b_ZeroShotEntities": ("models.llm_models", "Llama3_8b_ZeroShotEntities"),
    "Llama3_8b_OneShotEntities": ("models.llm_models", "Llama3_8b_OneShotEntities"),
    "Mistral_7b_ZeroShotEntities": ("models.llm_models", "Mistral_7b_ZeroShotEntities"),
    "Mistral_7b_OneShotEntities": ("models.llm_models", "Mistral_7b_OneShotEntities"),    
    "Mixtral_7b_ZeroShotEntities": ("models.llm_models", "Mixtral_7b_ZeroShotEntities"),
    "Mixtral_7b_OneShotEntities": ("models.llm_models", "Mixtral_7b_OneShotEntities"),
}

def get_models(model_texts: List[str]) -> List:
    if "all" in model_texts:
        model_texts = list(models_map.keys())
    models = []
    for model_text in model_texts:
        if model_text not in models_map:
            raise ValueError(f"Model {model_text} not found")
        module_name, class_name = models_map[model_text]
        my_class = getattr(importlib.import_module(module_name), class_name)
        models.append(my_class())
    return models

dataset_map = {
    "inspec": ("data.inspec_dataset", "inspec"),
    "kp20k": ("data.kp20k_dataset", "kp20k"),
    "semeval2010": ("data.semeval2010_dataset", "semeval2010"),
    "semeval2017": ("data.semeval2017_dataset", "semeval2017"),
}

def get_datasets(dataset_texts: List[str]) -> List:
    if "all" in dataset_texts:
        dataset_texts = list(dataset_map.keys())
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
    #parser = setup_parser()
    #args = parse_eval_args(parser)
    log_format = "%(asctime)s %(levelname)s %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format)
    
    models = get_models(['TfIdfEntities']) #args.models)
    datasets = get_datasets(['inspec']) #args.datasets)

    evaluate(models, datasets, 'output')#args.output)

if __name__ == "__main__":
    cli_evaluate()