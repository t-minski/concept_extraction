from keybert import KeyBERT
from torch import bfloat16
import transformers
from ctransformers import AutoModelForCausalLM as CAutoModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from torch import cuda
from conexion.models.base_model import BaseModel
from typing import List, Tuple
from keybert.llm import TextGeneration
from huggingface_hub import login
from models.prompts import templates
import openai
login("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")


class GPTEntities(BaseModel):
    
    def __init__(self, template_name="template_1"):
        self.template_name = template_name
        openai.api_key = "sk-proj-wdjLW7ovQqVBqpeiZkjoT3BlbkFJE48GT2KjFHrx9kmAJenK"
        self.model_name = "gpt-3.5-turbo"
        
        templates_gpt = {
            "template_1": {
                "system_prompt": """
                You are a helpful, respectful, and honest assistant for extracting concepts related to computer science from provided documents.
                """,
                "main_prompt": """
                I have the following document:
                {document}

                Extract all the concepts related to computer science present in this document and separate them with commas. Provide only the concepts and nothing else.
                """
            },
            "template_2": {
                "system_prompt": """
                You are a helpful, respectful, and honest assistant for extracting concepts related to computer science from provided documents.
                
                I have the following document:
                - The development of machine learning algorithms for data analysis involves complex computational techniques and models such as neural networks, decision trees, and ensemble methods, which are used to enhance pattern recognition and predictive analytics.

                Extract all the concepts related to computer science present in this document and separate them with commas. Provide only the concepts and nothing else.
                Example: machine learning algorithms, data analysis, complex computational techniques, neural networks, decision trees, ensemble methods, pattern recognition, predictive analytics
                """,
                "main_prompt": """
                I have the following document:
                {document}

                Extract all the concepts related to computer science present in this document and separate them with commas. Provide only the concepts and nothing else.
                """
            },
            "template_3": {
                "system_prompt": """
                You are an ontology expert in extracting concepts from documents in computer science.
                
                I have the following document:
                - The development of machine learning algorithms for data analysis involves complex computational techniques and models such as neural networks, decision trees, and ensemble methods, which are used to enhance pattern recognition and predictive analytics.

                Extract the concepts that can be represented as classes in ontology and are present in this document. Separate them with commas and provide no additional information.
                Example: machine learning algorithms, data analysis, complex computational techniques, neural networks, decision trees, ensemble methods, pattern recognition, predictive analytics
                """,
                "main_prompt": """
                I have the following document:
                {document}

                Extract the concepts that can be represented as classes in ontology and are present in this document. Separate them with commas and provide no additional information.
                """
            },
            "template_4": {
                "system_prompt": """
                You are an ontology expert in extracting concepts from documents in computer science.
                
                I have the following document:
                - A synergic analysis for Web-based enterprise resources planning systems. As the central nervous system for managing an organization's mission and critical business data, Enterprise Resource Planning (ERP) system has evolved to become the backbone of e-business implementation. Since an ERP system is multimodule application software that helps a company manage its important business functions, it should be versatile enough to automate every aspect of business processes, including e-business.

                Extract the concepts that can be represented as classes in ontology and are present in this document. Separate them with commas and provide no additional information.
                Example: synergic analysis, Web-based enterprise resources planning, Enterprise Resource Planning, ERP, e-business, customer relationship management
                """,
                "main_prompt": """
                I have the following document:
                {document}

                Extract the concepts that can be represented as classes in ontology and are present in this document. Separate them with commas and provide no additional information.
                """
            },
            "template_5": {
                "system_prompt": """
                You are an expert in extracting keyphrases from documents in computer science.
                
                I have the following document:
                - The development of machine learning algorithms for data analysis involves complex computational techniques and models such as neural networks, decision trees, and ensemble methods, which are used to enhance pattern recognition and predictive analytics.

                Extract the keyphrases related to computer science and separate them with commas. Ensure each keyphrase represents a main topic from the document and provide no additional information.
                Example: machine learning algorithms, data analysis, complex computational techniques, neural networks, decision trees, ensemble methods, pattern recognition, predictive analytics
                """,
                "main_prompt": """
                I have the following document:
                {document}

                Extract the keyphrases related to computer science and separate them with commas. Ensure each keyphrase represents a main topic from the document and provide no additional information.
                """
            },
            "template_6": {
                "system_prompt": """
                You are an expert in extracting concepts, keywords, and keyphrases from documents in computer science. A concept represents a set or class of noun phrases or entities within a domain. Keywords are concepts that are more important. Keyphrases are important multi or single words that cover main topics.
                
                I have the following document:
                - The development of machine learning algorithms for data analysis involves complex computational techniques and models such as neural networks, decision trees, and ensemble methods, which are used to enhance pattern recognition and predictive analytics.

                Extract the key concepts, keywords, and keyphrases related to computer science. Separate each with commas and provide no additional information.
                Example: machine learning algorithms, data analysis, complex computational techniques, neural networks, decision trees, ensemble methods, pattern recognition, predictive analytics
                """,
                "main_prompt": """
                I have the following document:
                {document}

                Extract the key concepts, keywords, and keyphrases related to computer science. Separate each with commas and provide no additional information.
                """
            }
        }
        self.kw_model = KeyBERT(model='BAAI/bge-small-en-v1.5')
        self.template = templates_gpt[self.template_name]
        
    def generate_response(self, document: str) -> str:
        prompt = self.template['system_prompt'] + self.template['main_prompt'].format(document=document)
        
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.template['system_prompt']},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.1
        )
        
        concepts = response.choices[0].message['content']
        return concepts.strip()

    def fit(self, abstracts: List[str], keyphrases: List[List[str]]) -> None:
        pass

    def predict(self, abstracts: List[str]) -> List[List[str]]:          
        entities = []
        for abstract in abstracts:
            concepts = self.generate_response(abstract)
            keywords = concepts.split(',')
            
            keywords_with_scores = [
                (keyword.strip().rstrip('.'), 1.0) 
                for keyword in keywords 
                if keyword.strip().rstrip('.').lower() in abstract.lower()
            ]
            entities.append(keywords_with_scores)
        
        return entities
    
    def get_template_name(self) -> str:
        return self.template_name


class MultiPax(BaseModel):
    
    def __init__(self):
        self.kw_model = KeyBERT()

    def fit(self, abstracts: List[str], keyphrases: List[List[str]]) -> None:
        pass

    def predict(self, abstracts: List[str]) -> List[List[str]]:        
        # Extract keywords using KeyBERT entities
        entities = []
        for abstract in abstracts:
            keywords = self.kw_model.extract_keywords([abstract], keyphrase_ngram_range=(1, 3), use_mmr=True, diversity=0.5,
                                         stop_words='english', top_n=10)
            keywords_with_scores = [
                                    (keyword.rstrip('.'), score) 
                                    for keyword, score in keywords 
                                    if keyword.rstrip('.').lower() in abstract.lower()
                                    ]
            entities.append(keywords_with_scores)
        
        return entities

class KeyBERTEntities(BaseModel):
    
    def __init__(self):
        self.kw_model = KeyBERT()

    def fit(self, abstracts: List[str], keyphrases: List[List[str]]) -> None:
        pass

    def predict(self, abstracts: List[str]) -> List[List[str]]:        
        # Extract keywords using KeyBERT entities
        entities = []
        for abstract in abstracts:
            keywords = self.kw_model.extract_keywords([abstract], threshold=0.5)
            keywords_with_scores = [
                                    (keyword.rstrip('.'), score) 
                                    for keyword, score in keywords 
                                    if keyword.rstrip('.').lower() in abstract.lower()
                                    ]
            entities.append(keywords_with_scores)
        
        return entities


class Llama2_7b_Entities(BaseModel):
    
    def __init__(self, template_name="template_1"):
        self.template_name = template_name
        model_id = 'meta-llama/Llama-2-7b-chat-hf'
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

        generator = pipeline(
            "text-generation",
            model=model, tokenizer=tokenizer, max_new_tokens=1000, temperature=0.1,
            model_kwargs={"torch_dtype": torch.float16, "use_cache": False}  # Adjusted to suit LLaMA model specifics
        )

        # Load the selected template
        template = templates[template_name]
        system_prompt = template["system_prompt"]
        example_prompt = template.get("example_prompt")
        main_prompt = template["main_prompt"]
        
        if example_prompt:
            prompt = system_prompt + example_prompt + main_prompt
        else:
            prompt = system_prompt + main_prompt
        
        llm = TextGeneration(generator, prompt=prompt)
        self.kw_model = KeyBERT(llm=llm, model='BAAI/bge-small-en-v1.5')

    def fit(self, abstracts: List[str], keyphrases: List[List[str]]) -> None:
        pass

    def predict(self, abstracts: List[str]) -> List[List[str]]:        
        # Extract keywords using llama2 entities
        entities = []
        for abstract in abstracts:
            keywords = self.kw_model.extract_keywords([abstract], keyphrase_ngram_range=(1, 3), use_mmr=True, diversity=0.5,
                                         stop_words='english', top_n=10)[0]
        
            keywords_with_scores = [
                                    (keyword.rstrip('.'), 1.) 
                                    for keyword in keywords 
                                    if keyword.rstrip('.').lower() in abstract.lower()
                                    ]
            entities.append(keywords_with_scores)
        
        return entities
    
    def get_template_name(self) -> str:
        return self.template_name

class Llama2_70b_Entities(BaseModel):
    
    def __init__(self, template_name="template_1"):
        self.template_name = template_name
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

        generator = pipeline(
            "text-generation",
            model=model, tokenizer=tokenizer, max_new_tokens=1000, temperature=0.1,
            model_kwargs={"torch_dtype": torch.float16, "use_cache": False}  # Adjusted to suit LLaMA model specifics
        )

        # Load the selected template
        template = templates[template_name]
        system_prompt = template["system_prompt"]
        example_prompt = template.get("example_prompt")
        main_prompt = template["main_prompt"]
        
        if example_prompt:
            prompt = system_prompt + example_prompt + main_prompt
        else:
            prompt = system_prompt + main_prompt
        
        llm = TextGeneration(generator, prompt=prompt)
        self.kw_model = KeyBERT(llm=llm, model='BAAI/bge-small-en-v1.5')

    def fit(self, abstracts: List[str], keyphrases: List[List[str]]) -> None:
        pass

    def predict(self, abstracts: List[str]) -> List[List[str]]:        
        # Extract keywords using llama2 entities
        entities = []
        for abstract in abstracts:
            keywords = self.kw_model.extract_keywords([abstract], keyphrase_ngram_range=(1, 3), use_mmr=True, diversity=0.5,
                                         stop_words='english', top_n=10)[0]
        
            keywords_with_scores = [
                                    (keyword.rstrip('.'), 1.) 
                                    for keyword in keywords 
                                    if keyword.rstrip('.').lower() in abstract.lower()
                                    ]
            entities.append(keywords_with_scores)
        
        return entities
    
    def get_template_name(self) -> str:
        return self.template_name

class Llama3_8b_Entities(BaseModel):
    
    def __init__(self, template_name="template_1"):
        self.template_name = template_name
        model_id = "meta-llama/Meta-Llama-3-8B"
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

        generator = pipeline(
            "text-generation",
            model=model, tokenizer=tokenizer, max_new_tokens=1000, temperature=0.1,
            model_kwargs={"torch_dtype": torch.float16, "use_cache": False}  # Adjusted to suit LLaMA model specifics
        )

        # Load the selected template
        template = templates[template_name]
        system_prompt = template["system_prompt"]
        example_prompt = template.get("example_prompt")
        main_prompt = template["main_prompt"]
        
        if example_prompt:
            prompt = system_prompt + example_prompt + main_prompt
        else:
            prompt = system_prompt + main_prompt
            
        llm = TextGeneration(generator, prompt=prompt)
        self.kw_model = KeyBERT(llm=llm, model='BAAI/bge-small-en-v1.5')

    def fit(self, abstracts: List[str], keyphrases: List[List[str]]) -> None:
        pass

    def predict(self, abstracts: List[str]) -> List[List[str]]:        
        # Extract keywords using llama2 entities
        entities = []
        for abstract in abstracts:
            keywords = self.kw_model.extract_keywords([abstract], keyphrase_ngram_range=(1, 3), use_mmr=True, diversity=0.5,
                                         stop_words='english', top_n=10)[0]
            keywords_with_scores = [
                                    (keyword.rstrip('.'), 1.) 
                                    for keyword in keywords 
                                    if keyword.rstrip('.').lower() in abstract.lower()
                                    ]
            entities.append(keywords_with_scores)
        
        return entities
    
    def get_template_name(self) -> str:
        return self.template_name

class Llama3_70b_Entities(BaseModel):
    
    def __init__(self, template_name="template_1"):
        self.template_name = template_name
        model_id = "meta-llama/Meta-Llama-3-70B"
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

        generator = pipeline(
            "text-generation",
            model=model, tokenizer=tokenizer, max_new_tokens=1000, temperature=0.1,
            model_kwargs={"torch_dtype": torch.float16, "use_cache": False}  # Adjusted to suit LLaMA model specifics
        )

        # Load the selected template
        template = templates[template_name]
        system_prompt = template["system_prompt"]
        example_prompt = template.get("example_prompt")
        main_prompt = template["main_prompt"]
        
        if example_prompt:
            prompt = system_prompt + example_prompt + main_prompt
        else:
            prompt = system_prompt + main_prompt
            
        llm = TextGeneration(generator, prompt=prompt)
        self.kw_model = KeyBERT(llm=llm, model='BAAI/bge-small-en-v1.5')

    def fit(self, abstracts: List[str], keyphrases: List[List[str]]) -> None:
        pass

    def predict(self, abstracts: List[str]) -> List[List[str]]:        
        # Extract keywords using llama3 entities
        entities = []
        for abstract in abstracts:
            keywords = self.kw_model.extract_keywords([abstract], keyphrase_ngram_range=(1, 3), use_mmr=True, diversity=0.5,
                                         stop_words='english', top_n=10)[0]
            keywords_with_scores = [
                                    (keyword.rstrip('.'), 1.) 
                                    for keyword in keywords 
                                    if keyword.rstrip('.').lower() in abstract.lower()
                                    ]
            entities.append(keywords_with_scores)
        
        return entities
    
    def get_template_name(self) -> str:
        return self.template_name

class Mistral_7b_Entities(BaseModel):
    
    def __init__(self, template_name="template_1"):
        self.template_name = template_name
        
        model_id = 'mistralai/Mistral-7B-Instruct-v0.1'
        model = AutoModelForCausalLM.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # Pipeline
        generator = pipeline(
                            "text-generation",
                            model=model, tokenizer=tokenizer, max_new_tokens=1000,
                            model_kwargs={"torch_dtype": torch.float16, "load_in_4bit": True},
                            )

        # Load the selected template
        template = templates[template_name]
        system_prompt = template["system_prompt"]
        example_prompt = template.get("example_prompt")
        main_prompt = template["main_prompt"]
        
        if example_prompt:
            prompt = system_prompt + example_prompt + main_prompt
        else:
            prompt = system_prompt + main_prompt
                
        llm = TextGeneration(generator, prompt=prompt)
        self.kw_model = KeyBERT(llm=llm, model='BAAI/bge-small-en-v1.5')

    def fit(self, abstracts: List[str], keyphrases: List[List[str]]) -> None:
        pass

    def predict(self, abstracts: List[str]) -> List[List[str]]:        
        # Extract keywords using KeyBERT entities
        entities = []
        for abstract in abstracts:
            keywords = self.kw_model.extract_keywords([abstract], keyphrase_ngram_range=(1, 3), use_mmr=True, diversity=0.5,
                                         stop_words='english', top_n=10)[0]
            keywords_with_scores = [
                                    (keyword.rstrip('.'), 1.) 
                                    for keyword in keywords 
                                    if keyword.rstrip('.').lower() in abstract.lower()
                                    ]
            entities.append(keywords_with_scores)
        
        return entities
    
    def get_template_name(self) -> str:
        return self.template_name

class Mixtral_7b_Entities(BaseModel):
    
    def __init__(self, template_name="template_1"):
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        self.template_name = template_name
        model_id = 'mistralai/Mixtral-8x7B-Instruct-v0.1'
        model = AutoModelForCausalLM.from_pretrained(model_id)#AutoTokenizer.from_pretrained(model_id)

        # Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id)#AutoTokenizer.from_pretrained(model_id)

        # Pipeline
        generator = pipeline(
                            "text-generation",
                            model=model, tokenizer=tokenizer, max_new_tokens=1000,
                            model_kwargs={"torch_dtype": torch.float16, "load_in_4bit": True},
                            )

        # Load the selected template
        template = templates[template_name]
        system_prompt = template["system_prompt"]
        example_prompt = template.get("example_prompt")
        main_prompt = template["main_prompt"]
        
        if example_prompt:
            prompt = system_prompt + example_prompt + main_prompt
        else:
            prompt = system_prompt + main_prompt
        
        llm = TextGeneration(generator, prompt=prompt)
        self.kw_model = KeyBERT(llm=llm, model='BAAI/bge-small-en-v1.5')

    def fit(self, abstracts: List[str], keyphrases: List[List[str]]) -> None:
        pass

    def predict(self, abstracts: List[str]) -> List[List[str]]:        
        # Extract keywords using KeyBERT entities
        entities = []
        for abstract in abstracts:
            keywords = self.kw_model.extract_keywords([abstract], keyphrase_ngram_range=(1, 3), use_mmr=True, diversity=0.5,
                                         stop_words='english', top_n=10)[0]
            keywords_with_scores = [
                                    (keyword.rstrip('.'), 1.) 
                                    for keyword in keywords 
                                    if keyword.rstrip('.').lower() in abstract.lower()
                                    ]
            entities.append(keywords_with_scores)
        
        return entities
    
    def get_template_name(self) -> str:
        return self.template_name
    
class Mixtral_22b_Entities(BaseModel):
    
    def __init__(self, template_name="template_1"):
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        self.template_name = template_name
        model_id = 'mistralai/Mixtral-8x22B-Instruct-v0.1'
        model = AutoModelForCausalLM.from_pretrained(model_id)#AutoTokenizer.from_pretrained(model_id)

        # Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id)#AutoTokenizer.from_pretrained(model_id)

        # Pipeline
        generator = pipeline(
                            "text-generation",
                            model=model, tokenizer=tokenizer, max_new_tokens=1000,
                            model_kwargs={"torch_dtype": torch.float16, "load_in_4bit": True},
                            )

        # Load the selected template
        template = templates[template_name]
        system_prompt = template["system_prompt"]
        example_prompt = template.get("example_prompt")
        main_prompt = template["main_prompt"]
        
        if example_prompt:
            prompt = system_prompt + example_prompt + main_prompt
        else:
            prompt = system_prompt + main_prompt
        
        llm = TextGeneration(generator, prompt=prompt)
        self.kw_model = KeyBERT(llm=llm, model='BAAI/bge-small-en-v1.5')

    def fit(self, abstracts: List[str], keyphrases: List[List[str]]) -> None:
        pass

    def predict(self, abstracts: List[str]) -> List[List[str]]:        
        # Extract keywords using KeyBERT entities
        entities = []
        for abstract in abstracts:
            keywords = self.kw_model.extract_keywords([abstract], keyphrase_ngram_range=(1, 3), use_mmr=True, diversity=0.5,
                                         stop_words='english', top_n=10)[0]
            keywords_with_scores = [
                                    (keyword.rstrip('.'), 1.) 
                                    for keyword in keywords 
                                    if keyword.rstrip('.').lower() in abstract.lower()
                                    ]
            entities.append(keywords_with_scores)
        
        return entities
    
    def get_template_name(self) -> str:
        return self.template_name
