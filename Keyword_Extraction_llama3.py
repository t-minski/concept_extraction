# %%
import csv
import os
from ctransformers import AutoModelForCausalLM as CAutoModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from keybert.llm import TextGeneration
from keybert import KeyLLM, KeyBERT
from sentence_transformers import SentenceTransformer
import bibtexparser
from yake import KeywordExtractor
from rake_nltk import Rake
from sklearn.metrics import precision_recall_fscore_support
from yake import KeywordExtractor
from rake_nltk import Rake
from gensim.models.phrases import Phrases, ENGLISH_CONNECTOR_WORDS
from pyate import combo_basic, basic, cvalues
from summa import keywords as summa_keywords
import torch
import spacy
import pandas as pd
from keybert import KeyBERT
from nltk.stem import PorterStemmer
import Levenshtein
from Levenshtein import distance
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
# %%
import nltk
nltk.download('stopwords')
spacy.load("en_core_web_lg")
from huggingface_hub import login
login("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")

from torch import cuda

model_id = 'meta-llama/Meta-Llama-3-8B'
device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

print(device)

from torch import bfloat16
import transformers

# set quantization configuration to load large model with less GPU memory
# this requires the `bitsandbytes` library

bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,  # 4-bit quantization
    bnb_4bit_quant_type='nf4',  # Normalized float 4
    bnb_4bit_use_double_quant=True,  # Second quantization after the first
    bnb_4bit_compute_dtype=bfloat16  # Computation type
)

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

model_id = "meta-llama/Meta-Llama-3-8B"
tokenizer_llama3 = AutoTokenizer.from_pretrained(model_id)
model_llama3 = transformers.AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                quantization_config=bnb_config,
                device_map='auto',
            )
model_llama3.eval()

generator_llama3 = pipeline(
    "text-generation",
    model=model_llama3, tokenizer=tokenizer_llama3, max_new_tokens=100, temperature=0.1,
    model_kwargs={"torch_dtype": torch.float16, "use_cache": False}  # Adjusted to suit LLaMA model specifics
)
# System prompt describes information given to all conversations
system_prompt = """
<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant for extracting concepts related to computer science from provided documents.
<</SYS>>
"""

example_prompt = """
I have a topic that contains the following documents:
- The development of machine learning algorithms for data analysis involves complex computational techniques and models such as neural networks, decision trees, and ensemble methods, which are used to enhance pattern recognition and predictive analytics.

Give me all the concepts that are present in this document and related to computer science and separate them with commas.
Make sure you only return the concepts and say nothing else. For example, don't say:
"Sure, I'd be happy to help! Based on the information provided in the document".
[/INST] machine learning algorithms,data analysis,complex computational techniques,neural networks, decision trees,ensemble methods,pattern recognition,predictive analytics,
"""

main_prompt = """
[INST]
I have the following document:
[DOCUMENT]

Give me the topics that are present in this document and related to computer science and separate them with commas.
Make sure you only return the concepts and say nothing else. For example, don't say:
"Sure, I'd be happy to help! Based on the information provided in the document".
[/INST]
"""

prompt_zero_shot = system_prompt +  main_prompt
prompt_one_shot = system_prompt + example_prompt + main_prompt

# %%
def extract_keywords_from_abstract(abstract):
    # Assuming kw_model_mistral is already initialized and configured correctly
    abstract = abstract.lower()

    # llama3_zero_shot
    llm_llama3 = TextGeneration(generator_llama3, prompt=prompt_zero_shot)
    kw_model_llama3 = KeyBERT(llm=llm_llama3, model='BAAI/bge-small-en-v1.5')
    raw_keywords = kw_model_llama3.extract_keywords([abstract], threshold=0.5)[0]
    cleaned_keywords_zero_shot = [keyword.rstrip('.') for keyword in raw_keywords if keyword.rstrip('.').lower() in abstract.lower()]
    
    # llama3_one_shot
    llm_llama3 = TextGeneration(generator_llama3, prompt=prompt_one_shot)
    kw_model_llama3 = KeyBERT(llm=llm_llama3, model='BAAI/bge-small-en-v1.5')
    raw_keywords = kw_model_llama3.extract_keywords([abstract], threshold=0.5)[0]
    cleaned_keywords_one_shot = [keyword.rstrip('.') for keyword in raw_keywords if keyword.rstrip('.').lower() in abstract.lower()]

    return {
        "llama3_KeyBERT_zero_shot": cleaned_keywords_zero_shot,
        "llama3_KeyBERT_one_shot": cleaned_keywords_one_shot,
    }
    

# %%
# Function to tokenize and stem text
def tokenize_and_stem(text):
    stemmer = PorterStemmer()
    if isinstance(text, str):
        tokens = [stemmer.stem(word) for word in text.split()]
        return ' '.join(tokens)
    else:
        return str(text)

# Function to evaluate keywords with Levenshtein threshold
def evaluate_keywords(ground_truth_keywords, extracted_keywords, threshold=0.8):
    ground_truth_keywords = list(set(ground_truth_keywords))
    extracted_keywords = list(set(extracted_keywords))
    
    # Initialize variables for evaluation metrics
    tp, fp, fn = 0, 0, 0

    # Tokenize and stem ground truth keywords
    ground_truth_stems = [tokenize_and_stem(keyword) for keyword in ground_truth_keywords]

    # Iterate over extracted keywords
    for extracted_keyword in extracted_keywords:
        # Tokenize and stem extracted keyword
        extracted_stem = tokenize_and_stem(extracted_keyword)

        # Check if extracted keyword matches any ground truth keyword within Levenshtein threshold
        matched = False
        for ground_truth_stem in ground_truth_stems:
            max_len = max(len(extracted_stem), len(ground_truth_stem))
            if Levenshtein.distance(extracted_stem, ground_truth_stem) / max_len <= 1 - threshold:
                matched = True
                break

        # Update evaluation metrics based on match status
        if matched:
            tp += 1
        else:
            fp += 1

    # Calculate false negatives (missed ground truth keywords)
    fn = len(ground_truth_keywords) - tp

    # Calculate precision, recall, and F1-score
    if tp + fp > 0:
        precision = tp / (tp + fp)
    else:
        precision = 0.0

    if tp + fn > 0:
        recall = tp / (tp + fn)
    else:
        recall = 0.0

    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0.0

    return precision, recall, f1_score

# %%
def read_files_from_directory(base_path, dataset):
    abstracts = []
    keywords = []

    docs_path = os.path.join(base_path, dataset, 'docsutf8')
    keys_path = os.path.join(base_path, dataset, 'keys')

    for filename in os.listdir(docs_path):
        if filename.endswith('.txt'):
            with open(os.path.join(docs_path, filename), 'r', encoding='utf-8') as file:
                if dataset == 'SemEval2010':
                    content = file.read()
                    abstract = extract_semeval_abstract(content)
                else:
                    abstract = file.read().strip()
                #abstracts[identifier] = abstract
                abstracts.append(abstract)

    for filename in os.listdir(keys_path):
        if filename.endswith('.key'):
            with open(os.path.join(keys_path, filename), 'r', encoding='utf-8') as file:
                keywords.append([line.strip() for line in file.readlines()])
    return abstracts, keywords

def extract_semeval_abstract(content):
    lines = content.split('\n')
    abstract_lines = []
    in_abstract = False
    for line in lines:
        if line.startswith('ABSTRACT'):
            in_abstract = True
        elif line.startswith('Categories and Subject Descriptors'):
            in_abstract = False
        elif in_abstract:
            abstract_lines.append(line.strip())
    return ' '.join(abstract_lines).strip()

# %%
def evaluate_keywords_from_data(base_path, datasets, extraction_functions, output_folder):

    for dataset in datasets:
        cumulative_precision = {method: 0 for method in extraction_functions}
        cumulative_recall = {method: 0 for method in extraction_functions}
        cumulative_f1_score = {method: 0 for method in extraction_functions}

        all_evaluation_results = []
        all_evaluation_results_avg = []
        total_abstracts = 0
        abstracts, keywords = read_files_from_directory(base_path, dataset)
        
        #for identifier, abstract in abstracts.items():
        for abstract, ground_truth_keywords in zip(abstracts, keywords):
            
            total_abstracts += 1

            for method, extraction_function in extraction_functions.items():
                extracted_keywords = extraction_function(abstract.lower())
                precision, recall, f1_score = evaluate_keywords(ground_truth_keywords, extracted_keywords)

                cumulative_precision[method] += precision
                cumulative_recall[method] += recall
                cumulative_f1_score[method] += f1_score

                all_evaluation_results.append((ground_truth_keywords, extracted_keywords, method, precision, recall, f1_score, len(ground_truth_keywords), len(extracted_keywords)))
    
        average_precision = {method: cumulative_precision[method] / total_abstracts for method in extraction_functions}
        average_recall = {method: cumulative_recall[method] / total_abstracts for method in extraction_functions}
        average_f1_score = {method: cumulative_f1_score[method] / total_abstracts for method in extraction_functions}

        # Print average scores
        print("Average Scores over all Abstracts:")
        for method in extraction_functions:
            print(f"Method      , Average Precision:                    , Average Recall:                    , Average F1-score:                    ")
            print(f"{method},{average_precision[method]},{average_recall[method]},{average_f1_score[method]}")
            all_evaluation_results_avg.append((method, average_precision[method], average_recall[method], average_f1_score[method]))

        # Write ground truth keywords, extracted keywords, and evaluation results to CSV files
        with open(os.path.join(output_folder, f'evaluation_results_llama3_{dataset}.csv'), 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Ground_truth Keywords', 'Extracted Keywords', 'Method', 'Precision', 'Recall', 'F1-score', 'n_gt_keywords', 'n_extraced_leywords'])
            writer.writerows(all_evaluation_results)
        
        with open(os.path.join(output_folder, f'evaluation_results_avg_llama3_{dataset}.csv'), 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Method', 'Precision', 'Recall', 'F1-score'])
            writer.writerows(all_evaluation_results_avg)

# Define extraction functions
extraction_functions = {
    "llama3_KeyBERT_zero_shot": lambda abstract: extract_keywords_from_abstract(abstract)["llama3_KeyBERT_zero_shot"],
    "llama3_KeyBERT_one_shot": lambda abstract: extract_keywords_from_abstract(abstract)["llama3_KeyBERT_one_shot"],
}

# %%
base_path = 'data_cs'
datasets = ['Inspec', 'SemEval2010', 'www']

output_folder = 'output'
evaluate_keywords_from_data(base_path, datasets, extraction_functions, output_folder)

# %%
