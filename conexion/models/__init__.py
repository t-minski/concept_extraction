import importlib
from typing import List

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