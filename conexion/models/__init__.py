import importlib
from typing import List

models_map = {
    "SpacyEntities": ("conexion.models.spacy_models", "SpacyEntities", {}),
    "SpacyNounChunks": ("conexion.models.spacy_models", "SpacyNounChunks", {}),
    "TfIdfEntities": ("conexion.models.tfidf_models", "TfIdfEntities", {}),
    "YakeEntities": ("conexion.models.yake_models", "YakeEntities", {}),
    "SummaEntities": ("conexion.models.summa_models", "SummaEntities", {}),
    "RakeEntities": ("conexion.models.rake_models", "RakeEntities", {}),
    "PyateBasicsEntities": ("conexion.models.pyate_models", "PyateBasicsEntities", {}),
    "PyateComboBasicEntities": ("conexion.models.pyate_models", "PyateComboBasicEntities", {}),
    "PyateCvaluesEntities": ("conexion.models.pyate_models", "PyateCvaluesEntities", {}),
    "LSAEntities": ("conexion.models.lsa_models", "LSAEntities", {}),
    "LDAEntities": ("conexion.models.lda_models", "LDAEntities", {}),
    
    "pke_FirstPhrases": ("conexion.models.pke_models", "pke_FirstPhrases", {}),
    "pke_TextRank": ("conexion.models.pke_models", "pke_TextRank", {}),
    "pke_SingleRank": ("conexion.models.pke_models", "pke_SingleRank", {}),
    "pke_TopicRank": ("conexion.models.pke_models", "pke_TopicRank", {}),
    "pke_MultipartiteRank": ("conexion.models.pke_models", "pke_MultipartiteRank", {}),
    "pke_TfIdf": ("conexion.models.pke_models", "pke_TfIdf", {}),
    "pke_TopicalPageRank": ("conexion.models.pke_models", "pke_TopicalPageRank", {}),
    "pke_YAKE": ("conexion.models.pke_models", "pke_YAKE", {}),
    "pke_KPMiner": ("conexion.models.pke_models", "pke_KPMiner", {}),
    "pke_Kea": ("conexion.models.pke_models", "pke_Kea", {}),
    
    "EmbedRank": ("conexion.models.EmbedRank_models", "EmbedRank", {}),
    "KeyBERTEntities": ("conexion.models.llm_models", "KeyBERTEntities", {}),
    "Llama2_7b_Entities": ("conexion.models.llm_models", "Llama2_7b_Entities", {}),
    "Llama2_70b_Entities": ("conexion.models.llm_models", "Llama2_70b_Entities", {}),
    "Llama3_8b_Entities": ("conexion.models.llm_models", "Llama3_8b_Entities", {}),
    "Llama3_70b_Entities": ("conexion.models.llm_models", "Llama3_70b_Entities", {}),
    "Mistral_7b_Entities": ("conexion.models.llm_models", "Mistral_7b_Entities", {}),
    "Mixtral_7b_Entities": ("conexion.models.llm_models", "Mixtral_7b_Entities", {}),
    "Mixtral_22b_Entities": ("conexion.models.llm_models", "Mixtral_22b_Entities", {}),
    "AdvancedConceptExtractor": ("conexion.models.conex_models", "AdvancedConceptExtractor", {}),
    "GPTEntities": ("conexion.models.llm_models", "GPTEntities", {}),

    # llm base
    "LLMBaseModel": ("conexion.models.llm_confidence_models", "LLMBaseModel", {}),
}


def __handle_arg_string(arg):
    if arg.lower() == "true":
        return True
    elif arg.lower() == "false":
        return False
    elif arg.isnumeric():
        return int(arg)
    try:
        return float(arg)
    except ValueError:
        return arg

def __simple_parse_args_string(args_string):
    """
    Parses something like
        args1=val1,arg2=val2
    Into a dictionary
    """
    args_string = args_string.strip()
    if not args_string:
        return {}
    arg_list = [arg for arg in args_string.split(",") if arg]
    args_dict = {
        k: __handle_arg_string(v) for k, v in [arg.split("=") for arg in arg_list]
    }
    return args_dict

def get_models(model_texts: List[str]) -> List:
    if "all" in model_texts:
        model_texts = list(models_map.keys())
    models = []
    for model_text in model_texts:
        if model_text in models_map:
            module_name, class_name, default_arguments = models_map[model_text]
            my_class = getattr(importlib.import_module(module_name), class_name)
            models.append(my_class(**default_arguments))
        else:
            # default to parameters in command line like
            # --models class=LLM,model_name=meta-llama/Llama-2-70b-chat-hf,prompt=prompt1,revision=154f235,extractive_keywords_only=true
            model_args = __simple_parse_args_string(model_text)
            if "class" not in model_args:
                raise ValueError(f"Model argument {model_text} must contain a class key e.g. class=LLM")
            class_name = model_args.pop("class")
            if class_name not in models_map:
                raise ValueError(f"Model class {class_name} not found in models_map")
            module_name, class_name, default_arguments = models_map[class_name]
            arguments = {**default_arguments, **model_args}
            my_class = getattr(importlib.import_module(module_name), class_name)
            models.append(my_class(**arguments))
    return models