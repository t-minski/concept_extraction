from conexion.models.base_model import BaseModel
from conexion.models.prompt_utils import get_prepared_prompt_as_text
from typing import List, Tuple, Union
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import random # TODO: set seed in main for all random functions
import logging
import torch
import re

SPIECE_UNDERLINE = "▁"
logger = logging.getLogger(__name__)

def split_keywords(keywords: str) -> List[str]:
    keyword_list = []
    for keyword in re.split(',|;|\*|\n', keywords):
        keyword = keyword.strip()
        if keyword:
            keyword_list.append(keyword)
    return keyword_list


class LLMBaseModel(BaseModel):
    
    def __init__(self, 
                 prompt: Union[str, List], 
                 model_name: str, 
                 revision : str = "main", 
                 with_confidence: bool = False, 
                 batched_generation: bool = False,
                 extractive_keywords_only: bool = True):
        """
        Args:
            prompt (str): The prompt to be used for the model.
            model_name (str): The name of the model to be used.
            revision (str): The revision of the model to be used.
            with_confidence (bool): Whether to return the confidence of the keywords.
            extractive_keywords_only (bool): Whether to extract only the keywords present in the abstract.
        """
        self.prompt = prompt
        self.model_name = model_name
        self.revision = revision
        self.with_confidence = with_confidence
        self.batched_generation = batched_generation
        self.extractive_keywords_only = extractive_keywords_only

    def fit(self, abstracts: List[str], keyphrases: List[List[str]]) -> None:
        pass

    def compute_one_training_data(self, prediction_abstract : str) -> List[Tuple[str, List[str]]]:
        return []

    def compute_all_training_data(self, prediction_abstracts : List[str]) -> List[List[Tuple[str, List[str]]]]:
        return [self.compute_one_training_data(prediction_abstract) for prediction_abstract in prediction_abstracts]


    def batched_computation(self, abstracts: List[str]) -> List[List[Tuple[str, float]]]:
        # https://huggingface.co/docs/transformers/llm_tutorial#wrong-padding-side
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side="left") # padding side is important
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto"
        )

        tokenizer.pad_token = tokenizer.eos_token  # Most LLMs don't have a pad token by default

        # prepare the prompts
        prepared_prompts = [get_prepared_prompt_as_text(self.prompt, abstract, [], tokenizer) for abstract in abstracts]
        model_inputs = tokenizer(prepared_prompts, padding=True, return_tensors="pt")
        model_inputs.to(model.device)

        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs.get("attention_mask", None)
        
        # to suppress the user warning when we set do_sample=False and a temperature is still set by the default generation config
        model.generation_config.temperature=None
        model.generation_config.top_p=None

        if self.with_confidence:
            # throw exception
            raise Exception("Confidence is not implemented for batched generation.")
        else:
            raise Exception("Not implemented yet.")
            outputs = model.generate(
                input_ids=input_ids, attention_mask=attention_mask, # token and attention input
                num_beams=1, do_sample=False  # make it deterministic -> greedy decoding
            )
            generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True) # batch decode
            print(generated_text)
            #generated_text = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
            #splitted_keywords = split_keywords(generated_text)
            #if self.extractive_keywords_only:
            #    splitted_keywords = [keyword for keyword in splitted_keywords if keyword in abstract]
            #result.append([(keyword, 1.0) for keyword in splitted_keywords])


            #outputs = model.generate(
            #    input_ids=input_ids, attention_mask=attention_mask, # token and attention input
            #    num_beams=1, do_sample=False  # make it deterministic -> greedy decoding
            #)


    def non_batched_computation(self, abstracts: List[str]) -> List[List[Tuple[str, float]]]:
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto"
        )
        result = []
        for abstract in abstracts:
            prepared_prompt = get_prepared_prompt_as_text(self.prompt, abstract, [], tokenizer)
            model_inputs = tokenizer(prepared_prompt, return_tensors="pt")
            model_inputs.to(model.device)

            input_ids = model_inputs["input_ids"]
            attention_mask = model_inputs.get("attention_mask", None)

            # to suppress the user warning when we set do_sample=False and a temperature is still set by the default generation config
            model.generation_config.temperature=None
            model.generation_config.top_p=None

            if self.with_confidence:
                outputs = model.generate(
                    input_ids=input_ids, attention_mask=attention_mask, # token and attention input
                    output_scores=True, return_dict_in_generate=True, # also return the scores
                    num_beams=1, do_sample=False  # make it deterministic -> greedy decoding
                )
                transition_scores = model.compute_transition_scores(outputs.sequences, outputs.scores, normalize_logits=True)
                transition_proba = torch.exp(transition_scores)

                input_length = 1 if model.config.is_encoder_decoder else input_ids.shape[1]
                generated_ids = outputs.sequences[:, input_length:]
                generated_tokens = tokenizer.convert_ids_to_tokens(generated_ids[0])

                # https://huggingface.co/spaces/joaogante/color-coded-text-generation/blob/main/app.py
                # underscore (_) is used to signify token that this token is at the beginning of words
                # https://stackoverflow.com/questions/78039649/huggingface-tokenizer-has-two-ids-for-the-same-token
                keyword_list = []
                current_keyword = ""
                current_confidence = 1.0
                for token, proba in zip(generated_tokens, transition_proba[0]):
                    replaced_token = token.replace(SPIECE_UNDERLINE, " ")
                    if "," in replaced_token or ";" in replaced_token or "*" in replaced_token or "\n" in replaced_token:
                        if len(replaced_token.strip()) > 1:
                            logger.warning(f"Separator token {replaced_token} does not only consist of the separator.")
                        keyword_to_be_added = current_keyword.replace('<0x0A>', '').strip()
                        if keyword_to_be_added:
                            keyword_list.append((keyword_to_be_added, current_confidence))
                        current_keyword = ""
                        current_confidence = 1.0
                    else:
                        current_keyword += replaced_token
                        # https://en.wikipedia.org/wiki/Chain_rule_(probability)
                        current_confidence *= proba.item()
                result.append(keyword_list)
            else:
                outputs = model.generate(
                    input_ids=input_ids, attention_mask=attention_mask, # token and attention input
                    num_beams=1, do_sample=False  # make it deterministic -> greedy decoding
                )
                generated_text = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
                result.append([(keyword, 1.0) for keyword in split_keywords(generated_text)])
        return result


    def predict(self, abstracts: List[str]) -> List[List[Tuple[str, float]]]:
        result = self.batched_computation(abstracts) if self.batched_generation else self.non_batched_computation(abstracts)
        if self.extractive_keywords_only:
            new_result = []
            for abstract, keywords in zip(abstracts, result):
                new_result.append([(keyword, confidence) for keyword, confidence in keywords if keyword in abstract])
            return new_result
        else:
            return result
    


class LLMRandomButFixedTraining(LLMBaseModel):
    
    def __init__(self, 
                 prompt: Union[str, List], 
                 model_name: str, 
                 revision: str = "main", 
                 with_confidence: bool = False, 
                 batched_generation: bool = False, 
                 extractive_keywords_only: bool = True,
                 number_of_examples: int = 5):
        super().__init__(prompt, model_name, revision, with_confidence, batched_generation, extractive_keywords_only)
        self.number_of_examples = number_of_examples

    def fit(self, abstracts: List[str], keyphrases: List[List[str]]) -> None:
        self.training_data = zip(abstracts, keyphrases)[:self.number_of_examples]

    def compute_one_training_data(self, prediction_abstract : str) -> List[Tuple[str, List[str]]]:
        return self.training_data
    

class LLMRandomTraining(LLMBaseModel):
    
    def __init__(self, 
                 prompt: Union[str, List], 
                 model_name: str, 
                 revision: str = "main", 
                 with_confidence: bool = False, 
                 batched_generation: bool = False, 
                 extractive_keywords_only: bool = True,
                 number_of_examples: int = 5):
        super().__init__(prompt, model_name, revision, with_confidence, batched_generation, extractive_keywords_only)
        self.number_of_examples = number_of_examples

    def fit(self, abstracts: List[str], keyphrases: List[List[str]]) -> None:
        self.training_data = zip(abstracts, keyphrases)

    def compute_one_training_data(self, prediction_abstract : str) -> List[Tuple[str, List[str]]]:
        return random.sample(self.training_data, self.number_of_examples)

class LLMClosestTraining(LLMBaseModel):
    
    def __init__(self, 
                 prompt: Union[str, List], 
                 model_name: str, 
                 revision: str = "main", 
                 with_confidence: bool = False, 
                 batched_generation: bool = False, 
                 extractive_keywords_only: bool = True,
                 number_of_examples: int = 5):
        super().__init__(prompt, model_name, revision, with_confidence, batched_generation, extractive_keywords_only)
        self.number_of_examples = number_of_examples
        self.embedder = SentenceTransformer('all-mpnet-base-v2')

    def fit(self, abstracts: List[str], keyphrases: List[List[str]]) -> None:
        # embed the corpus
        self.embedded_abstracts = self.embedder.encode(abstracts, convert_to_tensor=True)
    




class RestrictedLLMBasedGeneration(BaseModel):
    """This model is restricted to only words that appear in the abstract and some separator tokens."""

    def __init__(self, prompt: str, model_name: str):
        """
        Args:
            prompt (str): The prompt to be used for the model.
            model_name (str): The name of the model to be used.
        """
        self.prompt = prompt
        self.model_name = model_name

    def fit(self, abstracts: List[str], keyphrases: List[List[str]]) -> None:
        pass 

    def predict(self, abstracts: List[str]) -> List[List[Tuple[str, float]]]:
        
        # other work: 
        # https://github.com/MaartenGr/KeyBERT/blob/master/keybert/llm/_textgenerationinference.py
        # https://huggingface.co/docs/text-generation-inference/conceptual/guidance
        # https://huggingface.co/docs/text-generation-inference/basic_tutorials/using_guidance
        # https://github.com/huggingface/text-generation-inference/blob/main/clients/python/text_generation/inference_api.py

        # change the generate function
        # https://github.com/huggingface/transformers/blob/v4.18.0/src/transformers/generation_utils.py#L831
        # https://github.com/huggingface/transformers/blob/v4.18.0/src/transformers/generation_utils.py#L1254
        # https://github.com/huggingface/transformers/blob/v4.18.0/src/transformers/generation_utils.py#L1585
        # https://github.com/huggingface/transformers/blob/v4.18.0/src/transformers/generation_utils.py#L1673


        # TODO: implement
        
        return []






