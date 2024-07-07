from keybert import KeyBERT
from torch import bfloat16
import transformers
from ctransformers import AutoModelForCausalLM as CAutoModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from torch import cuda
from typing import List, Tuple
from keybert.llm import TextGeneration
from huggingface_hub import login
from conexion.models.base_model import BaseModel
from conexion.models.prompts import templates
from conexion.models.llm_models import Llama2_70b_Entities
from conexion.models.pke_models import pke_TextRank

login("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")

class AdvancedConceptExtractor(BaseModel):

    def __init__(self):
        self.text_rank_model = pke_TextRank()
        self.template_name = "template_7"
        model_id = 'meta-llama/Llama-2-70b-chat-hf'
        device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
        bnb_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,  # 4-bit quantization
            bnb_4bit_quant_type='nf4',  # Normalized float 4
            bnb_4bit_use_double_quant=True,  # Second quantization after the first
            bnb_4bit_compute_dtype=bfloat16  # Computation type
        )
        
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = transformers.AutoModelForCausalLM.from_pretrained(
                        model_id,
                        trust_remote_code=True,
                        quantization_config=bnb_config,
                        device_map='auto',
                    )
        model.eval()

        self.generator = pipeline(
            "text-generation",
            model=model, tokenizer=tokenizer, max_new_tokens=1000, temperature=0.1,
            model_kwargs={"torch_dtype": torch.float16, "use_cache": False}  # Adjusted to suit LLaMA model specifics
        )
        
        #llm = TextGeneration(self.generator, prompt=prompt)
        #self.kw_model = KeyBERT(llm=llm, model='BAAI/bge-small-en-v1.5')

    def fit(self, abstracts: List[str], keyphrases: List[List[str]]) -> None:
        pass

    def predict(self, abstracts: List[str]) -> List[List[Tuple[str, float]]]:
        # First, extract keywords using TextRank model
        text_rank_keywords = self.text_rank_model.predict(abstracts)

        refined_keywords = []
        # Then refine these keywords using Llama2_70b model with template_1
        for i, abstract in enumerate(abstracts):
            keywords_string = ", ".join([kw[0] for kw in text_rank_keywords[i]])
            prompt = templates["template_7"]["system_prompt"] + templates["template_7"]["main_prompt"] + f"\n{keywords_string}\n" + templates["template_7"]["main_prompt_"]
            llm = TextGeneration(self.generator, prompt=prompt)
            kw_model = KeyBERT(llm=llm, model='BAAI/bge-small-en-v1.5')
            refined_result = kw_model.extract_keywords([abstract], keyphrase_ngram_range=(1, 3), use_mmr=True, diversity=0.5,
                                         stop_words='english', top_n=10)[0]

            # Extract keywords from the refined result
            refined_keywords_with_scores = [
                (keyword.strip(), 1.0) 
                for keyword in refined_result 
                if keyword.strip().lower() in abstract.lower()
            ]
            refined_keywords.append(refined_keywords_with_scores)

        return refined_keywords

    
    def get_template_name(self) -> str:
        return self.template_name

