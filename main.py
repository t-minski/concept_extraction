import argparse
import logging
from conexion.evaluation.evaluator import evaluate
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
    
    parser.add_argument(
        "--template", "-t", help="Name of the template to use e.g. `template_1`", default="template_1"
    )

    parser.add_argument("-v", "--verbose", help="increase output verbosity",
                    action="store_true")

    return parser




models_map = {
    "SpacyEntities": ("conexion.models.spacy_models", "SpacyEntities"),
    "SpacyNounChunks": ("conexion.models.spacy_models", "SpacyNounChunks"),
    "TfIdfEntities": ("conexion.models.tfidf_models", "TfIdfEntities"),
    "YakeEntities": ("conexion.models.yake_models", "YakeEntities"),
    "SummaEntities": ("conexion.models.summa_models", "SummaEntities"),
    "RakeEntities": ("conexion.models.rake_models", "RakeEntities"),
    "PyateBasicsEntities": ("conexion.models.pyate_models", "PyateBasicsEntities"),
    "PyateComboBasicEntities": ("conexion.models.pyate_models", "PyateComboBasicEntities"),
    "PyateCvaluesEntities": ("conexion.models.pyate_models", "PyateCvaluesEntities"),
    "LSAEntities": ("conexion.models.lsa_models", "LSAEntities"),
    "LDAEntities": ("conexion.models.lda_models", "LDAEntities"),
    
    "pke_FirstPhrases": ("conexion.models.pke_models", "pke_FirstPhrases"),
    "pke_TextRank": ("conexion.models.pke_models", "pke_TextRank"),
    "pke_SingleRank": ("conexion.models.pke_models", "pke_SingleRank"),
    "pke_TopicRank": ("conexion.models.pke_models", "pke_TopicRank"),
    "pke_MultipartiteRank": ("conexion.models.pke_models", "pke_MultipartiteRank"),
    "pke_TfIdf": ("conexion.models.pke_models", "pke_TfIdf"),
    "pke_TopicalPageRank": ("conexion.models.pke_models", "pke_TopicalPageRank"),
    "pke_YAKE": ("conexion.models.pke_models", "pke_YAKE"),
    "pke_KPMiner": ("conexion.models.pke_models", "pke_KPMiner"),
    "pke_Kea": ("conexion.models.pke_models", "pke_Kea"),
    
    "EmbedRank": ("conexion.models.EmbedRank_models", "EmbedRank"),
    "KeyBERTEntities": ("conexion.models.llm_models", "KeyBERTEntities"),
    "Llama2_7b_Entities": ("conexion.models.llm_models", "Llama2_7b_Entities"),
    "Llama2_70b_Entities": ("conexion.models.llm_models", "Llama2_70b_Entities"),
    "Llama3_8b_Entities": ("conexion.models.llm_models", "Llama3_8b_Entities"),
    "Llama3_70b_Entities": ("conexion.models.llm_models", "Llama3_70b_Entities"),
    "Mistral_7b_Entities": ("conexion.models.llm_models", "Mistral_7b_Entities"),
    "Mixtral_7b_Entities": ("conexion.models.llm_models", "Mixtral_7b_Entities"),
    "Mixtral_22b_Entities": ("conexion.models.llm_models", "Mixtral_22b_Entities"),
    "AdvancedConceptExtractor": ("conexion.models.conex_models", "AdvancedConceptExtractor"),
    "GPTEntities": ("conexion.models.llm_models", "GPTEntities"),
}

def get_models(model_texts: List[str], template_name: str) -> List:
    if "all" in model_texts:
        model_texts = list(models_map.keys())
    models = []
    for model_text in model_texts:
        if model_text not in models_map:
            raise ValueError(f"Model {model_text} not found")
        module_name, class_name = models_map[model_text]
        my_class = getattr(importlib.import_module(module_name), class_name)
        if 'llm_models' in module_name:  # Only pass template_name to LLM models
            models.append(my_class(template_name=template_name))
        else:
            models.append(my_class())
    return models

dataset_map = {
    "inspec": ("conexion.data.inspec_dataset", "inspec"),
    "kp20k": ("conexion.data.kp20k_dataset", "kp20k"),
    "semeval2010": ("conexion.data.semeval2010_dataset", "semeval2010"),
    "semeval2017": ("conexion.data.semeval2017_dataset", "semeval2017"),
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
    parser = setup_parser()
    args = parse_eval_args(parser)
    #log_format = "%(asctime)s %(levelname)s %(message)s"
    #logging.basicConfig(level=logging.INFO, format=log_format)
    
    models = get_models(args.models, args.template)
    datasets = get_datasets(args.datasets)

    evaluate(models, datasets, args.output)

if __name__ == "__main__":
    cli_evaluate()