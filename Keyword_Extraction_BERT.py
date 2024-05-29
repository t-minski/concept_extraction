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

# %%
def extract_keywords_from_abstract(abstract):

    # Get the English stopwords
    stop_words = set(stopwords.words('english'))
    abstract = ' '.join([word for word in abstract.split() if word.lower() not in stop_words])
    
    # Extract keywords using KeyBERT
    keybert_model = KeyBERT()#KeyBERT(model="m3rg-iitd/matscibert")#KeyBERT()
    keybert_keywords = [keyword[0] for keyword in keybert_model.extract_keywords(abstract, keyphrase_ngram_range=(1, 3), stop_words='english')] #keyphrase_ngram_range=(1, 3),
    
    # Extract keywords using KeyBERT+MatSciBERT
    keybert_m_model = KeyBERT(model="m3rg-iitd/matscibert")#KeyBERT()
    keybert_m_keywords = [keyword[0] for keyword in keybert_m_model.extract_keywords(abstract, keyphrase_ngram_range=(1, 3), stop_words='english')] #keyphrase_ngram_range=(1, 3),

    return {
        "Keybert_keywords": keybert_keywords,
        "Keybert_m_keywords": keybert_m_keywords,
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
    cumulative_precision = {method: 0 for method in extraction_functions}
    cumulative_recall = {method: 0 for method in extraction_functions}
    cumulative_f1_score = {method: 0 for method in extraction_functions}

    all_evaluation_results = []
    all_evaluation_results_avg = []
    total_abstracts = 0
    for dataset in datasets:
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
    with open(os.path.join(output_folder, 'evaluation_results_bert.csv'), 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Ground_truth Keywords', 'Extracted Keywords', 'Method', 'Precision', 'Recall', 'F1-score', 'n_gt_keywords', 'n_extraced_leywords'])
        writer.writerows(all_evaluation_results)
    
    with open(os.path.join(output_folder, 'evaluation_results_avg_bert.csv'), 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Method', 'Precision', 'Recall', 'F1-score'])
        writer.writerows(all_evaluation_results_avg)


# %%
# Define extraction functions
extraction_functions = {
    "Keybert_keywords": lambda abstract: extract_keywords_from_abstract(abstract)["Keybert_keywords"],
    "Keybert_m_keywords": lambda abstract: extract_keywords_from_abstract(abstract)["Keybert_m_keywords"],
}

# %%
base_path = 'data_cs'
datasets = ['Inspec', 'SemEval2010', 'www']

output_folder = 'output'
evaluate_keywords_from_data(base_path, datasets, extraction_functions, output_folder)

# %%
