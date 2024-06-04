# %%
import csv
import os
import bibtexparser
from yake import KeywordExtractor
from rake_nltk import Rake
from sklearn.metrics import precision_recall_fscore_support
from yake import KeywordExtractor
from rake_nltk import Rake
from gensim.models.phrases import Phrases, ENGLISH_CONNECTOR_WORDS
from pyate import combo_basic, basic, cvalues
from summa import keywords as summa_keywords
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
# %%
import nltk
nltk.download('stopwords')
spacy.load("en_core_web_lg")
# %%
def extract_keywords_from_abstract(abstract):

    # Get the English stopwords
    stop_words = set(stopwords.words('english'))
    abstract = ' '.join([word for word in abstract.split() if word.lower() not in stop_words])

    # Initialize Spacy, YAKE, and RAKE keyword extractors
    nlp = spacy.load("en_core_web_lg")
    kw_extractor = KeywordExtractor(lan="en", n=2,
                                    dedupLim=0.81, dedupFunc='seqm',
                                    windowsSize=1, top=37)
    rake_nltk_var = Rake()

    # Extract keywords using Spacy entities
    doc = nlp(abstract)
    spacy_entities = [ent.text for ent in doc.ents]

    # Extract keywords using Spacy noun chunks
    doc = nlp(abstract)
    spacy_noun_chunks = [chunk.text for chunk in doc.noun_chunks]

    # Extract keywords using YAKE
    yake_keywords = kw_extractor.extract_keywords(abstract)
    yake_keywords = [keyword[0] for keyword in yake_keywords]

    # Extract keywords using RAKE
    rake_nltk_var.extract_keywords_from_text(abstract)
    rake_keywords = rake_nltk_var.get_ranked_phrases()

    # Extract keywords using Pyate
    pyate_combo_basic_keywords = combo_basic(abstract).sort_values(ascending=False).index.str.split().str[0].tolist()
    pyate_basic_keywords = basic(abstract).sort_values(ascending=False).index.str.split().str[0].tolist()
    pyate_cvalues_keywords = cvalues(abstract).sort_values(ascending=False).index.str.split().str[0].tolist()

    # Extract keywords using summa
    summa_keywords_ = [keyword[0] for keyword in summa_keywords.keywords(abstract, scores=True)]
    
    # Extract keywords using TF-IDF
    tfidf_vectorizer = TfidfVectorizer(
                                    stop_words='english',
                                    ngram_range=(1, 2),
                                    #max_df=max_df,
                                    min_df=0.49,
                                    #max_features=max_features,
                                    norm=None,
                                    use_idf=True,
                                    sublinear_tf=True)
    tfidf_matrix = tfidf_vectorizer.fit_transform([abstract])
    tfidf_keywords = tfidf_vectorizer.get_feature_names_out()

    # Extract keywords using LSA
    lsa_model = TruncatedSVD(n_components=10)  # Adjust the number of components as needed
    lsa_matrix = lsa_model.fit_transform(tfidf_matrix)
    lsa_keywords = [tfidf_keywords[i] for i in lsa_model.components_[0].argsort()[::-1]]

    # Extract keywords using LDA
    dictionary = Dictionary([abstract.split()])
    corpus = [dictionary.doc2bow(abstract.split())]
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in [abstract.split()]]
    lda_model = LdaModel(corpus=doc_term_matrix, num_topics=18, id2word=dictionary)  # Adjust the number of topics as needed
    lda_keywords = [word for word, _ in lda_model.show_topic(0)]

    return {
        "Spacy_entities": spacy_entities,
        "Spacy_noun_chunks": spacy_noun_chunks,
        "YAKE_keywords": yake_keywords,
        "RAKE_keywords": rake_keywords,
        "Pyate_combo_basic_keywords": pyate_combo_basic_keywords,
        "Pyate_basic_keywords": pyate_basic_keywords,
        "Pyate_cvalues_keywords": pyate_cvalues_keywords,
        "Summa_keywords": summa_keywords_,
        "TFIDF_keywords": tfidf_keywords,
        "LSA_keywords": lsa_keywords,
        "LDA_keywords": lda_keywords,
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
                if dataset == 'Krapivin2009':
                    content = file.read()
                    abstract = extract_krapivin_abstract(content)
                else:
                    abstract = file.read().strip()
                #abstracts[identifier] = abstract
                abstracts.append(abstract)

    for filename in os.listdir(keys_path):
        if filename.endswith('.key'):
            with open(os.path.join(keys_path, filename), 'r', encoding='utf-8') as file:
                keywords.append([line.strip() for line in file.readlines()])
    return abstracts, keywords

def extract_krapivin_abstract(content):
    lines = content.split('\n')
    abstract_lines = []
    in_abstract = False
    for line in lines:
        if line.strip() == '--A':
            in_abstract = True
        elif line.strip() == '--B':
            in_abstract = False
        elif in_abstract:
            abstract_lines.append(line.strip())
    return ' '.join(abstract_lines).strip()

# %%
def evaluate_keywords_from_data(base_path, datasets, extraction_functions, output_folder):

    for dataset in datasets:
        abstracts, keywords = read_files_from_directory(base_path, dataset)       
        for method in ['TFIDF', 'YAKE', 'LSA', 'LDA']:
            study = optuna.create_study(direction='maximize')
            study.optimize(lambda trial: objective(trial, method, abstracts, keywords, len(abstracts)), n_trials= 100)

            # Output the best parameters found
            print('method and dataset are: ', method, dataset)
            print("Best parameters are:", study.best_params)


# %%
# Define extraction functions
extraction_functions = {
    "Spacy_entities": lambda abstract: extract_keywords_from_abstract(abstract)["Spacy_entities"],
    "Spacy_noun_chunks": lambda abstract: extract_keywords_from_abstract(abstract)["Spacy_noun_chunks"],
    "YAKE_keywords": lambda abstract: extract_keywords_from_abstract(abstract)["YAKE_keywords"],
    "RAKE_keywords": lambda abstract: extract_keywords_from_abstract(abstract)["RAKE_keywords"],
    "Pyate_combo_basic_keywords": lambda abstract: extract_keywords_from_abstract(abstract)["Pyate_combo_basic_keywords"],
    "Pyate_basic_keywords": lambda abstract: extract_keywords_from_abstract(abstract)["Pyate_basic_keywords"],
    "Pyate_cvalues_keywords": lambda abstract: extract_keywords_from_abstract(abstract)["Pyate_cvalues_keywords"],
    "Summa_keywords": lambda abstract: extract_keywords_from_abstract(abstract)["Summa_keywords"],
    "TFIDF_keywords": lambda abstract: extract_keywords_from_abstract(abstract)["TFIDF_keywords"],
    "LSA_keywords": lambda abstract: extract_keywords_from_abstract(abstract)["LSA_keywords"],
    "LDA_keywords": lambda abstract: extract_keywords_from_abstract(abstract)["LDA_keywords"],
}

# %%
import optuna
import logging

# Setup logging to only show warning messages or higher (suppressing info messages)
logging.getLogger('optuna').setLevel(logging.WARNING)

# Define the optimization objective function for a given method and abstract
def objective(trial, method, abstracts, ground_truth_keywords_list, num_abstracts):
    
    total_score = 0
    for abstract, ground_truth_keywords in zip(abstracts, ground_truth_keywords_list):
        if method == "YAKE":
            max_ngram_size  = trial.suggest_int("n", 1, 3)
            dedupLim = trial.suggest_float("dedupLim", 0.1, 1.0)
            dedupFunc = trial.suggest_categorical("dedupFunc", ["seqm", "jaro", "levenshtein"])
            windowsSize = trial.suggest_int("windowsSize", 1, 3)
            top = trial.suggest_int("top", 10, 50)
            kw_extractor = KeywordExtractor(lan="en", n=max_ngram_size,
                                            dedupLim=dedupLim, dedupFunc=dedupFunc,
                                            windowsSize=windowsSize, top=top)
            keywords = [kw[0] for kw in kw_extractor.extract_keywords(abstract)]

        elif method == "TFIDF":
            ngram_range = trial.suggest_categorical("ngram_range", [(1, 1), (1, 2), (1, 3)])
            use_idf = trial.suggest_categorical("use_idf", [True, False])
            min_df = trial.suggest_float("min_df", 0, 1.)  # min document frequency (absolute counts)
            #max_df = trial.suggest_float("max_df", (min_df + 1) / num_abstracts, 1.0)  # max document frequency as a proportion
            #max_features = trial.suggest_categorical("max_features", [None, 500, 1000, 5000])  # number of max features
            norm = trial.suggest_categorical("norm", ['l1', 'l2', None])  # normalization
            sublinear_tf = trial.suggest_categorical("sublinear_tf", [True, False])  # sublinear term frequency scaling

            # Initialize TF-IDF Vectorizer with the suggested parameters
            tfidf_vectorizer = TfidfVectorizer(
                                            stop_words='english',
                                            ngram_range=ngram_range,
                                            #max_df=max_df,
                                            min_df=min_df,
                                            #max_features=max_features,
                                            norm=norm,
                                            use_idf=use_idf,
                                            sublinear_tf=sublinear_tf)
            tfidf_matrix = tfidf_vectorizer.fit_transform([abstract])
            keywords = tfidf_vectorizer.get_feature_names_out()
        
        elif method == "LSA":
            n_components = trial.suggest_int("n_components", 5, 50)
            ngram_range = trial.suggest_categorical("ngram_range", [(1, 1), (1, 2), (1, 3)])
            top_words = trial.suggest_int("top_words", 1, 20)
            lsa_model = TruncatedSVD(n_components=n_components)
            tfidf_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2),
                                            use_idf= True, min_df= 0.5875806866591319,
                                            norm= None, sublinear_tf= False)
            tfidf_matrix = tfidf_vectorizer.fit_transform([abstract])
            tfidf_keywords = tfidf_vectorizer.get_feature_names_out()
            lsa_model.fit_transform(tfidf_matrix)
            terms = [tfidf_keywords[i] for i in lsa_model.components_[0].argsort()[::-1]]
            # Accessing the terms and their respective weights
            component_terms = lsa_model.components_[0].argsort()[-top_words:][::-1]
            keywords = [terms[i] for i in component_terms]
        
        elif method == "LDA":
            dictionary = Dictionary([abstract.split()])
            corpus = [dictionary.doc2bow(abstract.split())]
            doc_term_matrix = [dictionary.doc2bow(doc) for doc in [abstract.split()]]
            
            num_topics = trial.suggest_int("num_topics", 2, 20)
            lda_model = LdaModel(corpus=doc_term_matrix, num_topics=num_topics, id2word=dictionary)
            keywords = [word for word, _ in lda_model.show_topic(0)]
    
        precision, recall, f1_score = evaluate_keywords(ground_truth_keywords, keywords)
    
        total_score += f1_score

    # Average score over all abstracts
    average_score = total_score / len(abstracts)
    return average_score

# %%
base_path = 'data_cs'
datasets = ['SemEval2017', 'SemEval2010', 'Krapivin2009']

output_folder = 'output'
evaluate_keywords_from_data(base_path, datasets, extraction_functions, output_folder)

# %%
